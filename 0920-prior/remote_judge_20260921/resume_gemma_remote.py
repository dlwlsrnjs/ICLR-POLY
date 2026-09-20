#!/usr/bin/env python3
"""Resume the merged Gemma handoff without rerunning valid judgments."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--batch',type=int,default=16);a=p.parse_args()
r=Path(__file__).resolve().parent;run=r/'all';revision='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
assert a.model.resolve().name==revision,'Use exact Qwen32 revision directory'
checks=json.loads((r/'SHA256SUMS.json').read_text())
for name in ['all/responses.jsonl','all/manifest.json','qwen32_dual_judge.py']:
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==checks[name],name
protocol=run/'judge_qwen32/protocol.json';value=json.loads(protocol.read_text())
assert value['revision']==revision
assert value['responses_sha256']==checks['all/responses.jsonl']
if value['model']!=str(a.model.resolve()):
 backup=protocol.with_name('protocol.before_remote_resume.json')
 if not backup.exists():backup.write_bytes(protocol.read_bytes())
 value['model']=str(a.model.resolve());protocol.write_text(json.dumps(value,indent=2)+'\n')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
raise SystemExit(subprocess.call([sys.executable,str(r/'qwen32_dual_judge.py'),'--run',str(run),'--model',str(a.model),'--batch',str(a.batch)],env=env))
