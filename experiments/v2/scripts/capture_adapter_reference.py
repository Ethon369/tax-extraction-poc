from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_split, write_json
from src.modeling import Engine

output=ROOT/'runs/formal/reload_expected.json'
if output.exists(): raise FileExistsError('Preserve the existing reload reference')
row=read_split('dev')[0]
prediction=Engine(adapter=ROOT/'runs/formal/adapter').generate(row)
write_json(output,{'row_id':row['id'],'source_text':row['messages'][1]['content'],'raw_output':prediction['raw_output']})
print('SELECTED_ADAPTER_REFERENCE_CAPTURED',flush=True)
