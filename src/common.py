from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = '''只输出JSON:{"action":动作,"data":数据,"reason":原因}。
抽取动作extract，原因null。数据键invoice_date(YYYY-MM-DD),taxpayer_id(买方税号),buyer_name,items,missing_fields。items元素键name,amount_without_tax,tax_rate,tax_amount。未知值null，不推算；税率13%=0.13；明细保持原顺序。missing_fields列出空字段路径如items.0.tax_rate，空items写items；tax_amount为空不列。
计算动作call_tool，原因null，数据键tool_name,parameters。仅支持calculate_tax_amount(amount_without_tax,tax_rate)和calculate_vat_deduction(tax_amount,deductible_ratio)。参数必须明确提供，比例用小数。
计算参数不足时动作fallback，数据null，原因insufficient_information；无关或未注册请求原因unsupported_request。'''

def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def setup_runtime():
    for variable, subdirectory in (("HF_HOME", ".cache/huggingface"), ("TORCH_HOME", ".cache/torch"), ("XDG_CACHE_HOME", ".cache")):
        os.environ[variable] = str(ROOT / subdirectory)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

def load_config():
    return read_json(ROOT / "configs/experiment.json")

def read_split(split):
    return read_json(ROOT / "data" / f"{split}.json")

def messages_for(row, few_shot=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        train = read_split("train")
        for example_id in ("train-01", "train-21"):
            example = next(item for item in train if item["id"] == example_id)
            messages.extend(example["messages"][1:])
    messages.append({"role": "user", "content": row["messages"][1]["content"]})
    return messages

def encode_supervised(tokenizer, messages):
    """Tokenize prompt and completion separately to match the inference boundary.

    BPE can otherwise merge the prompt's terminal newline with the answer's '{'.
    Rendered strings must match; only the completion tokens receive labels.
    """
    prompt = tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    if not rendered.startswith(prompt):
        raise ValueError('Assistant rendering is not a continuation of the prompt')
    prefix = tokenizer.encode(prompt,add_special_tokens=False)
    completion = tokenizer.encode(rendered[len(prompt):],add_special_tokens=False)
    if not completion:
        raise ValueError('No assistant completion tokens')
    return prefix,prefix+completion

def reduced_smoke_rows(rows):
    """Shorter workflow-only instructions; preserve all source inputs and answers."""
    import copy
    shortened=copy.deepcopy(rows)
    prompt='只输出JSON，键action,data,reason。计算用action=call_tool，reason=null，data含tool_name,parameters。仅支持calculate_tax_amount(amount_without_tax,tax_rate)、calculate_vat_deduction(tax_amount,deductible_ratio)，百分比转小数。无关问题action=fallback,data=null,reason=unsupported_request；计算参数不足reason=insufficient_information。'
    for row in shortened:
        row['messages'][0]['content']=prompt
    return shortened
