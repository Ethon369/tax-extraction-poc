"""Local-only model loading and identical greedy decoding for all experiments."""
from __future__ import annotations
import importlib.metadata
import time
from .common import ROOT, encode_supervised, load_config, messages_for, read_json, setup_runtime, sha256

def environment_info():
    import platform
    import torch
    names = ('torch','transformers','peft','accelerate','bitsandbytes','pydantic')
    return {'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in names},'gpu':torch.cuda.get_device_name(0),'total_vram_mib':torch.cuda.get_device_properties(0).total_memory/2**20,'cuda_runtime':torch.version.cuda}

class Engine:
    def __init__(self, adapter=None, train=False, reduced=False):
        setup_runtime()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model, LoraConfig
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Install the approved CUDA PyTorch wheel in the project environment.')
        self.config = load_config()
        set_seed(self.config['seed'])
        torch.set_num_threads(4)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.path = ROOT/'models/Qwen2.5-1.5B-Instruct'
        if not (self.path/'model.safetensors').exists():
            raise FileNotFoundError('Download and verify the model first: python scripts/download_model.py')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        q = self.config['quantization']
        quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type=q['type'],bnb_4bit_use_double_quant=q['double_quant'],bnb_4bit_compute_dtype=self.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(self.path,local_files_only=True,trust_remote_code=False,quantization_config=quant,torch_dtype=self.dtype,device_map={'':0},attn_implementation='sdpa')
        t = self.config['training'].copy()
        if reduced:
            t.update(self.config['fallback_training'])
        self.training_settings = t
        if train:
            self.model = prepare_model_for_kbit_training(self.model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
            self.model.config.use_cache = False
            self.model = get_peft_model(self.model,LoraConfig(r=t['lora_rank'],lora_alpha=t['lora_alpha'],lora_dropout=t['lora_dropout'],target_modules=t['target_modules'],bias='none',task_type='CAUSAL_LM'))
        elif adapter:
            self.model = PeftModel.from_pretrained(self.model,str(adapter),local_files_only=True,is_trainable=False)
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.eval()

    def encode_training(self, row):
        messages = row['messages']
        prefix,full = encode_supervised(self.tokenizer,messages)
        if full[:len(prefix)] != prefix:
            raise ValueError(f'{row["id"]}: assistant mask prefix mismatch')
        if len(full) > self.training_settings['max_length']:
            raise ValueError(f'{row["id"]}: {len(full)} tokens exceed {self.training_settings["max_length"]}; answer MUST NOT be truncated')
        labels = [-100]*len(prefix)+full[len(prefix):]
        if not any(value != -100 for value in labels):
            raise ValueError('No supervised assistant tokens')
        return {'input_ids':full,'attention_mask':[1]*len(full),'labels':labels,'id':row['id'],'tokens':len(full),'supervised_tokens':len(full)-len(prefix)}

    def generate(self, row, few_shot=False):
        import torch
        messages = messages_for(row,few_shot)
        inputs = self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
        if inputs.shape[1] > self.config['generation']['max_input_tokens']:
            raise ValueError(f'Input exceeds generation limit: {inputs.shape[1]}')
        self.model.eval()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=self.config['generation']['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=self.tokenizer.pad_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        generated = outputs[0,inputs.shape[1]:]
        return {'raw_output':self.tokenizer.decode(generated,skip_special_tokens=True),'latency_seconds':elapsed,'input_tokens':int(inputs.shape[1]),'output_tokens':int(len(generated)),'hit_token_limit':len(generated)==self.config['generation']['max_new_tokens']}

def provenance():
    return {'model_manifest_sha256':sha256(ROOT/'models/Qwen2.5-1.5B-Instruct/download_manifest.json'),'config_sha256':sha256(ROOT/'configs/experiment.json'),'data_sha256':{s:sha256(ROOT/'data'/f'{s}.json') for s in ('train','dev','test','smoke')}}
