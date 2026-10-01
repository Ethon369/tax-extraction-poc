"""Package source, frozen data, adapter, and evidence; exclude multi-GB runtime files."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json, sha256

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    required=[ROOT/'REPORT.md',ROOT/'runs/smoke/reload_verified.json',ROOT/'runs/formal/reload_verified.json',ROOT/'runs/formal/adapter/adapter_model.safetensors']
    required.extend(ROOT/f'runs/test-{mode}/metrics.json' for mode in ('zero','few','finetuned'))
    for path in required:
        if not path.exists(): raise FileNotFoundError(f'Incomplete delivery: {path}')
    assert read_json(ROOT/'runs/smoke/reload_verified.json')['reload_matches']
    assert read_json(ROOT/'runs/formal/reload_verified.json')['reload_matches']
    args.output.mkdir(parents=True,exist_ok=True)
    directories=['src','scripts','tests','data','configs','runs']
    files=[]
    for directory in directories:
        files.extend(path for path in (ROOT/directory).rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'))
    files.extend(ROOT/name for name in ('.gitignore','README.md','REPORT.md','LEARNING.md','requirements.txt','requirements.lock.txt','DEVELOPMENT.md') if (ROOT/name).exists())
    files.extend([ROOT/'models/Qwen2.5-1.5B-Instruct/download_manifest.json',ROOT/'models/Qwen2.5-1.5B-Instruct/LICENSE'])
    manifest={'files':{str(path.relative_to(ROOT)).replace('\\','/'):sha256(path) for path in sorted(files)}}
    archive=args.output/'tax-extraction-poc.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in sorted(files): stream.write(path,arcname='tax-extraction-poc/'+str(path.relative_to(ROOT)).replace('\\','/'))
        stream.writestr('tax-extraction-poc/DELIVERY_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        for path,expected in manifest['files'].items():
            assert hashlib.sha256(stream.read('tax-extraction-poc/'+path)).hexdigest()==expected,path
    for name in ('REPORT.md','README.md','LEARNING.md'):
        (args.output/name).write_bytes((ROOT/name).read_bytes())
    (args.output/'delivery_sha256.txt').write_text(f'{sha256(archive)}  {archive.name}\n',encoding='utf-8')
    print('DELIVERY_READY',archive,'files',len(files),'size_MiB',archive.stat().st_size/2**20)

if __name__=='__main__': main()
