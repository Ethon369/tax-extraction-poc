from __future__ import annotations
import difflib
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, encode_supervised, read_split, reduced_smoke_rows, write_json
from src.schemas import validate_response

def normalize(text):
    text=re.sub(r'TEST\d+','<ID>',text)
    text=re.sub(r'\d+(?:\.\d+)?','<N>',text)
    text=re.sub(r'模拟[^，；。\s（）/｜]+?(?:公司|厂|中心|商行|商店|研究室|工作室|实验室|合作社|社)','<BUYER>',text)
    return re.sub(r'\s+','',text)

def check(tokenizer=None):
    splits={s:read_split(s) for s in ('train','dev','test','smoke')}
    seen={}; families={}; lengths=[]; maximum_similarity=0; pair=None
    for split,rows in splits.items():
        for row in rows:
            validate_response(row['messages'][2]['content'])
            text=row['messages'][1]['content']; canonical=normalize(text)
            assert canonical not in seen,f'Normalized duplicate: {row["id"]}, {seen.get(canonical)}'
            seen[canonical]=row['id']
            family=row['template_family']
            assert family not in families or families[family]==split
            families[family]=split
            if tokenizer:
                prefix,full=encode_supervised(tokenizer,row['messages'])
                assert full[:len(prefix)]==prefix
                assert len(full)>len(prefix)
                assert len(full)<=512,f'{row["id"]} has {len(full)} tokens; do not truncate'
                lengths.append({'id':row['id'],'total':len(full),'assistant':len(full)-len(prefix)})
    for left in ('train','dev','smoke'):
        for right in ('test','dev'):
            if left==right:
                continue
            for a in splits[left]:
                for b in splits[right]:
                    value=difflib.SequenceMatcher(None,normalize(a['messages'][1]['content']),normalize(b['messages'][1]['content'])).ratio()
                    if value>maximum_similarity:
                        maximum_similarity=value; pair=[a['id'],b['id']]
    result={'counts':{s:len(v) for s,v in splits.items()},'normalized_duplicates':0,'maximum_cross_split_similarity':maximum_similarity,'closest_pair':pair,'lengths':lengths,'human_review':'Assistant-authored examples; user review pending'}
    write_json(ROOT/'runs/data_audit.json',result)
    return result

if __name__=='__main__':
    tokenizer=None
    if '--tokenizer' in sys.argv:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-1.5B-Instruct',local_files_only=True)
    result=check(tokenizer)
    if tokenizer:
        reduced=[]
        for row in reduced_smoke_rows(read_split('smoke')):
            prefix,full=encode_supervised(tokenizer,row['messages'])
            assert len(full)<=256 and len(full)>len(prefix)
            reduced.append({'id':row['id'],'tokens':len(full)})
        result['reduced_smoke_lengths']=reduced
        write_json(ROOT/'runs/data_audit.json',result)
    print(result)
