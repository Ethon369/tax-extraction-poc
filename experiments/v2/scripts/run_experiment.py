"""Reuse the approved root environment/config; keep all v2 outputs isolated."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=ROOT.parents[1]
PYTHON=PROJECT_ROOT/'.venv/Scripts/python.exe'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','train','compare'],required=True)
    args=parser.parse_args()
    if not PYTHON.exists(): raise FileNotFoundError('Prepare the approved project environment first')
    environment=dict(os.environ,PYTHONUTF8='0',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    commands={
      'prepare':[['scripts/build_data.py'],['scripts/check_data.py','--tokenizer'],['-m','unittest','discover','-s','tests','-v']],
      'train':[['scripts/check_data.py','--tokenizer'],['-m','src.train','--profile','formal'],['scripts/capture_adapter_reference.py'],['-m','src.inference','--verify-formal']],
      'compare':[['scripts/check_data.py','--tokenizer'],['-m','src.evaluate','--split','test','--mode','before'],['-m','src.evaluate','--split','test','--mode','after'],['scripts/build_report.py'],['scripts/check_data.py','--tokenizer']],
    }[args.stage]
    (ROOT/'runs').mkdir(exist_ok=True)
    log=ROOT/'runs'/f'{args.stage}_console.log'
    if log.exists(): raise FileExistsError(f'Stage evidence already exists: {log}')
    with log.open('w',encoding='utf-8') as stream:
        for command in commands:
            print('V2_COMMAND',command,flush=True)
            stream.write('COMMAND '+repr(command)+'\n');stream.flush()
            process=subprocess.Popen([str(PYTHON)]+command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            if process.wait()!=0: raise RuntimeError(f'V2 command failed: {command}; no automatic retry')

if __name__=='__main__': main()
