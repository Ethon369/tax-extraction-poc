"""Assistant-only QLoRA training, with explicit evidence of parameter updates."""
from __future__ import annotations
import argparse
import copy
import math
import time
from .common import ROOT, read_split, reduced_smoke_rows, write_json
from .modeling import Engine, environment_info, provenance

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=['smoke','formal'],default='smoke')
    parser.add_argument('--reduced',action='store_true',help='Approved low-memory settings; no silent truncation')
    args=parser.parse_args()
    if args.reduced and args.profile!='smoke':
        parser.error('--reduced is a short smoke-workflow check; formal samples cannot be silently shortened to 256 tokens')
    import torch
    from torch.utils.data import DataLoader
    engine=Engine(train=True,reduced=args.reduced)
    settings=engine.training_settings
    rows=read_split('smoke' if args.profile=='smoke' else 'train')
    if args.reduced:
        rows=reduced_smoke_rows(rows)
    encoded=[engine.encode_training(row) for row in rows]
    dev=[engine.encode_training(row) for row in read_split('dev')] if args.profile=='formal' else []
    def collate(batch):
        assert len(batch)==1
        return {k:torch.tensor([batch[0][k]],device='cuda') for k in ('input_ids','attention_mask','labels')}
    loader=DataLoader(encoded,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(engine.config['seed']),collate_fn=collate,num_workers=0)
    parameters=[(n,p) for n,p in engine.model.named_parameters() if p.requires_grad]
    assert parameters and all('lora_' in name for name,_ in parameters),'Only adapter parameters should be trainable'
    initial={name:p.detach().cpu().clone() for name,p in parameters}
    optimizer=torch.optim.AdamW([p for _,p in parameters],lr=settings['learning_rate'],weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',enabled=engine.dtype==torch.float16)
    accumulation=settings['gradient_accumulation']
    limit=engine.config['smoke']['optimizer_steps'] if args.profile=='smoke' else math.ceil(len(encoded)/accumulation)*settings['epochs']
    out=ROOT/'runs'/args.profile
    out.mkdir(parents=True,exist_ok=True)
    audit=[{k:r[k] for k in ('id','tokens','supervised_tokens')} for r in encoded]
    write_json(out/'token_audit.json',audit)
    print('TOKEN_AUDIT',max(r['tokens'] for r in audit),'TRAINABLE',sum(p.numel() for _,p in parameters),flush=True)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); logs=[]; dev_logs=[]; step=0; epoch=0; best=math.inf
    optimizer.zero_grad(set_to_none=True)
    while step<limit:
        epoch+=1; engine.model.train(); micro=0; running=0.0
        for index,batch in enumerate(loader):
            with torch.autocast('cuda',dtype=engine.dtype):
                loss=engine.model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            # Final incomplete accumulation group is correctly normalized.
            group_start=(index//accumulation)*accumulation
            group_size=min(accumulation,len(encoded)-group_start)
            scaler.scale(loss/group_size).backward()
            running+=float(loss.detach()); micro+=1
            if micro==group_size:
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],settings['max_grad_norm'])
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite gradients')
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); step+=1
                entry={'step':step,'epoch':epoch,'loss':running/micro,'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-start}
                logs.append(entry); print('TRAIN',entry,flush=True)
                write_json(out/'training_log.json',logs)
                micro=0; running=0.0
                if step>=limit:
                    break
        if dev:
            engine.model.eval(); values=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=engine.dtype):
                for row in dev:
                    values.append(float(engine.model(**collate([row])).loss))
            value=sum(values)/len(values)
            if not math.isfinite(value):
                raise RuntimeError('Non-finite validation loss')
            dev_logs.append({'epoch':epoch,'loss':value})
            print('DEV_LOSS',epoch,value,flush=True)
            if value<best:
                best=value; engine.model.save_pretrained(out/'adapter'); engine.tokenizer.save_pretrained(out/'adapter')
                write_json(out/'selected_checkpoint.json',{'epoch':epoch,'dev_loss':value,'rule':'minimum mean dev assistant-only loss; final test not consulted'})
    torch.cuda.synchronize()
    changed={name:float((p.detach().cpu()-initial[name]).abs().max()) for name,p in parameters}
    assert any(v>0 for v in changed.values()),'Adapter parameters did not change'
    if args.profile=='smoke':
        engine.model.save_pretrained(out/'adapter'); engine.tokenizer.save_pretrained(out/'adapter')
        prediction=engine.generate(rows[0])
        write_json(out/'reload_expected.json',{'row_id':rows[0]['id'],'source_text':rows[0]['messages'][1]['content'],'raw_output':prediction['raw_output']})
    summary={'profile':args.profile,'settings':settings,'dtype':str(engine.dtype),'optimizer_steps':step,'epochs_started':epoch,'training_seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'trainable_parameters':sum(p.numel() for _,p in parameters),'maximum_parameter_update':max(changed.values()),'all_losses_finite':all(math.isfinite(e['loss']) for e in logs),'dev_losses':dev_logs,'environment':environment_info(),'provenance':provenance()}
    write_json(out/'summary.json',summary)
    print('TRAINING_COMPLETE',summary,flush=True)

if __name__=='__main__':
    main()
