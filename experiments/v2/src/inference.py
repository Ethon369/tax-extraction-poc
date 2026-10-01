from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import ROOT, read_json, write_json
from .modeling import Engine
from .tools import execute_checked

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--text')
    p.add_argument('--adapter',type=Path)
    p.add_argument('--output',type=Path)
    verification=p.add_mutually_exclusive_group()
    verification.add_argument('--verify-smoke',action='store_true')
    verification.add_argument('--verify-formal',action='store_true')
    a=p.parse_args()
    adapter=a.adapter
    if a.verify_smoke:
        expected=read_json(ROOT/'runs/smoke/reload_expected.json'); a.text=expected['source_text']; adapter=ROOT/'runs/smoke/adapter'
    elif a.verify_formal:
        expected=read_json(ROOT/'runs/formal/reload_expected.json')
        a.text=expected['source_text'];adapter=ROOT/'runs/formal/adapter'
    if not a.text:
        p.error('--text, --verify-smoke or --verify-formal is required')
    engine=Engine(adapter=adapter)
    result=engine.generate({'messages':[{}, {'role':'user','content':a.text}]})
    result['runtime_guard']=execute_checked(result['raw_output'],a.text)
    if a.verify_smoke or a.verify_formal:
        result['reload_matches']=result['raw_output']==expected['raw_output']
        write_json(ROOT/'runs'/('smoke' if a.verify_smoke else 'formal')/'reload_verified.json',result)
        if not result['reload_matches']:
            raise AssertionError('Fresh-process greedy output changed after reload')
    if a.output:
        write_json(a.output,result)
    print(result)

if __name__=='__main__':
    main()
