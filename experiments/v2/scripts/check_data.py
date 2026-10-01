from __future__ import annotations
import difflib
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, PROJECT_ROOT, SYSTEM_PROMPT, encode_supervised, load_config, read_json, read_split, sha256, write_json
from src.schemas import validate_response
from src.tools import execute_checked

def normalize(text):
    text=re.sub(r'TEST\d+','<ID>',text)
    text=re.sub(r'calculate_[A-Za-z0-9_]+','<TOOL>',text)
    text=re.sub(r'\d+(?:\.\d+)?','<N>',text)
    text=re.sub(r'(?:模拟|演练|样例)[^，；。\s（）/｜、：]+?(?:公司|厂|中心|商行|商店|研究室|工作室|实验室|合作社|社|站|室|班|队|组|所|场|园|圃|库|团|馆)','<BUYER>',text)
    return re.sub(r'\s+','',text)

def audit_rows(splits, tokenizer=None):
    import json
    seen={};families={};lengths=[];maximum=0;closest=None
    for split,rows in splits.items():
        for row in rows:
            assert row['messages'][0]['content']==SYSTEM_PROMPT, row['id']
            answer=validate_response(row['messages'][2]['content'])
            text=row['messages'][1]['content'];normalized=normalize(text)
            assert normalized not in seen, f'Duplicate after normalization: {row["id"]}, {seen.get(normalized)}'
            seen[normalized]=row['id']
            family=row['template_family'];assert family not in families or families[family]==split
            families[family]=split
            gold=json.loads(row['messages'][2]['content'])
            if gold['action']=='call_tool':
                assert execute_checked(row['messages'][2]['content'],text)['status']=='executed', f'Gold tool cannot execute: {row["id"]}'
            if tokenizer:
                prefix,full=encode_supervised(tokenizer,row['messages'])
                assert len(prefix)<len(full)<=load_config()['training']['max_length'],f'Sequence too long: {row["id"]}, {len(full)}'
                lengths.append({'id':row['id'],'total':len(full),'assistant':len(full)-len(prefix)})
    names=list(splits)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            for a in splits[left]:
                for b in splits[right]:
                    similarity=difflib.SequenceMatcher(None,normalize(a['messages'][1]['content']),normalize(b['messages'][1]['content'])).ratio()
                    if similarity>maximum: maximum=similarity;closest=[a['id'],b['id']]
    assert maximum<.85, f'Cross-split near duplicate ({maximum:.3f}): {closest}'
    assert not set(row['messages'][1]['content'] for row in splits['test']) & set(row['messages'][1]['content'] for row in read_json(PROJECT_ROOT/'data/test.json'))
    return {'counts':{s:len(rows) for s,rows in splits.items()},'normalized_duplicates':0,'maximum_cross_split_similarity':maximum,'closest_pair':closest,'near_duplicate_threshold':.85,'lengths':lengths,'human_review':'Assistant-authored synthetic labels; user review pending.'}

def assert_preserved():
    protected=read_json(ROOT/'v1_preservation.json')['files']
    for path,digest in protected.items():
        assert sha256(PROJECT_ROOT/path)==digest, f'Original experiment changed: {path}'
    manifest=read_json(ROOT/'data/manifest.json')
    for split,digest in manifest['sha256'].items():
        assert sha256(ROOT/'data'/f'{split}.json')==digest, f'V2 frozen dataset changed: {split}'
    return {'v1_files_verified':len(protected),'v2_data_hashes_verified':True}

if __name__=='__main__':
    tokenizer=None
    if '--tokenizer' in sys.argv:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(PROJECT_ROOT/'models/Qwen2.5-1.5B-Instruct',local_files_only=True)
    result=audit_rows({s:read_split(s) for s in ('train','dev','test','smoke')},tokenizer)
    result.update(assert_preserved())
    write_json(ROOT/'runs/data_audit.json',result)
    print('V2_DATA_AUDIT', {key:value for key,value in result.items() if key!='lengths'}, flush=True)
