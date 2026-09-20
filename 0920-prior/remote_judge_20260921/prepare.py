import ast,hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
B=R/'download/qwen32_resume_bundle'
h=json.loads((R/'handoff.json').read_text())
code=(B/'code/qwen32_dual_judge.py').read_bytes()
assert hashlib.sha256(code).hexdigest()==h['code_sha256']
const={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['RECON_SYSTEM','FULFILL_SYSTEM']}
audit=[]
for tag in h['assigned_models']:
 run=B/'models'/tag/'all'; src=run/'responses.jsonl'
 rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]
 assert len(rows)==10592 and len({x['key'] for x in rows})==10592
 expected={'schema':'qwen32_dual_judge/v1','model':h['model'],'revision':Path(h['model']).name,'responses_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'temperature':0,'seed':20260920,'reconstruction_rubric_sha256':hashlib.sha256(const['RECON_SYSTEM'].encode()).hexdigest(),'fulfillment_rubric_sha256':hashlib.sha256(const['FULFILL_SYSTEM'].encode()).hexdigest()}
 p=run/'judge_qwen32/protocol.json'
 if p.exists():
  old=json.loads(p.read_text())
  assert {k:v for k,v in old.items() if k!='model'}=={k:v for k,v in expected.items() if k!='model'},tag
  assert Path(old['model']).name==expected['revision']
  backup=R/'original_protocols'/tag/'protocol.json';backup.parent.mkdir(parents=True,exist_ok=True)
  if not backup.exists(): shutil.copy2(p,backup)
  p.write_text(json.dumps(expected,indent=2)+'\n')
  audit.append({'model':tag,'old_checkpoint_path':old['model'],'new_checkpoint_path':h['model'],'other_protocol_fields_verified':True})
 lookup={x['key']:x['response_sha256'] for x in rows}
 for stage in ['reconstruction','fulfillment']:
  f=run/'judge_qwen32'/f'{stage}.jsonl'
  if f.exists():
   for line in f.read_text().splitlines():
    row=json.loads(line);assert lookup[row['key']]==row['source_response_sha256']
(R/'migration_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Verified nine inputs, existing response hashes, rubric and revision; path-only migration recorded.')
