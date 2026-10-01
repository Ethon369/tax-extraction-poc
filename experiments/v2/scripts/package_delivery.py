"""Package both frozen versions and all adapters; omit environments/base weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, PROJECT_ROOT, read_json, sha256
from check_data import assert_preserved

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert_preserved()
    assert read_json(ROOT/'runs/formal/reload_verified.json')['reload_matches']
    for path in (ROOT/'REPORT.md',ROOT/'runs/test-before/metrics.json',ROOT/'runs/test-after/metrics.json'):
        if not path.exists():raise FileNotFoundError(path)
    files=[]
    excluded={'.cache','.venv','.python','.bootstrap','__pycache__','.git'}
    for directory in ('src','scripts','tests','data','configs','runs','experiments'):
        files.extend(p for p in (PROJECT_ROOT/directory).rglob('*') if p.is_file() and not excluded.intersection(p.parts) and p.suffix not in ('.pyc','.pyo'))
    files.extend(PROJECT_ROOT/name for name in ('.gitignore','README.md','REPORT.md','LEARNING.md','DEVELOPMENT.md','requirements.txt','requirements.lock.txt'))
    files.extend(PROJECT_ROOT/'models/Qwen2.5-1.5B-Instruct'/name for name in ('download_manifest.json','LICENSE'))
    manifest={'files':{str(p.relative_to(PROJECT_ROOT)).replace('\\','/'):sha256(p) for p in sorted(files)}}
    args.output.mkdir(exist_ok=True,parents=True)
    archive=args.output/'tax-extraction-poc-v2.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in sorted(files):stream.write(path,'tax-extraction-poc/'+str(path.relative_to(PROJECT_ROOT)).replace('\\','/'))
        stream.writestr('tax-extraction-poc/DELIVERY_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        for path,digest in manifest['files'].items():assert hashlib.sha256(stream.read('tax-extraction-poc/'+path)).hexdigest()==digest,path
    (args.output/'REPORT-v2.md').write_bytes((ROOT/'REPORT.md').read_bytes())
    (args.output/'delivery-v2-sha256.txt').write_text(f'{sha256(archive)}  {archive.name}\n',encoding='utf-8')
    print('V2_DELIVERY_READY',archive,'files',len(files),'MiB',archive.stat().st_size/2**20,flush=True)

if __name__=='__main__':main()
