#!/usr/bin/env python3
"""GPU0 batch trial; persist descending batch fallback on CUDA OOM."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
control=R/'gpu0_batch_override.json'
args=sys.argv[1:]
idx=args.index('--batch')+1
batch=int(json.loads(control.read_text()).get('batch',4)) if control.exists() else 4
while True:
 args[idx]=str(batch)
 status_path=R/'gpu0.json'
 status=json.loads(status_path.read_text())
 status.update(batch=batch,stage='running',batch_override=True,time=time.time(),pid=os.getpid())
 tmp=status_path.with_suffix('.override.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(status_path)
 print(json.dumps({'event':'gpu0_batch_start','batch':batch,'time':time.time()}),flush=True)
 oom=False
 child=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in child.stdout:
  if 'out of memory' in line.lower():oom=True
  print(line,end='',flush=True)
 rc=child.wait()
 if rc and oom and batch>2:
  previous_batch=batch
  batch=max(2,batch//2)
  control.write_text(json.dumps({'batch':batch,'reason':f'CUDA OOM at batch {previous_batch}; user-authorized fallback','time':time.time()},indent=2)+'\n')
  print(json.dumps({'event':'gpu0_batch_fallback','batch':batch,'time':time.time()}),flush=True)
  time.sleep(3)
  continue
 sys.exit(rc if rc>=0 else 128-rc)
