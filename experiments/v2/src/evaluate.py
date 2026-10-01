from __future__ import annotations
import argparse
from pathlib import Path
from .common import ROOT, PROJECT_ROOT, dumps, read_json, read_split, sha256, write_json
from .metrics import score
from .modeling import Engine, environment_info, provenance
from .tools import execute_checked

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--mode',choices=['before','after'],required=True)
    args=parser.parse_args()
    import torch
    manifest=read_json(ROOT/'data/manifest.json')
    for split,expected in manifest['sha256'].items():
        if sha256(ROOT/'data'/f'{split}.json')!=expected:
            raise ValueError(f'Frozen dataset changed: {split}')
    adapter=(PROJECT_ROOT if args.mode=='before' else ROOT)/'runs/formal/adapter'
    if adapter and not adapter.exists():
        raise FileNotFoundError('Complete formal training first')
    engine=Engine(adapter=adapter)
    rows=read_split(args.split)
    engine.generate(rows[0])  # identical zero-shot prompts; warm-up excluded
    torch.cuda.reset_peak_memory_stats()
    out=ROOT/'runs'/f'{args.split}-{args.mode}'
    if (out/'predictions.jsonl').exists():
        raise FileExistsError(f'Results already exist: {out}; preserve them and choose a new experiment explicitly')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:
            prediction=engine.generate(row)
            source=row['messages'][1]['content']
            record={'id':row['id'],'category':row['category'],'scenario':row.get('scenario'),'source_text':source,'expected':read_json_answer(row),**prediction,'runtime_guard':execute_checked(prediction['raw_output'],source)}
            records.append(record); stream.write(dumps(record)+'\n'); stream.flush()
            print('EVAL',args.split,args.mode,row['id'],prediction['latency_seconds'],prediction['raw_output'][:100],flush=True)
    summary=score(records)
    from .boundary_metrics import boundary_score
    summary['boundaries']=boundary_score(records)
    summary.update({'mode':args.mode,'split':args.split,'adapter_sha256':sha256(adapter/'adapter_model.safetensors'),'adapter_path':str(adapter.relative_to(PROJECT_ROOT)).replace('\\','/'),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'environment':environment_info(),'provenance':provenance(),'runtime_executed':sum(r['runtime_guard']['status']=='executed' for r in records),'runtime_blocked':sum(r['runtime_guard']['status']=='blocked' for r in records)})
    write_json(out/'metrics.json',summary)
    print('EVALUATION_COMPLETE',summary,flush=True)

def read_json_answer(row):
    import json
    return json.loads(row['messages'][2]['content'])

if __name__=='__main__':
    main()
