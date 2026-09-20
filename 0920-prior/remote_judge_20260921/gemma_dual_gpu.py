#!/usr/bin/env python3
"""Split remaining Gemma judgments into eight native bb69581 shards on two GPUs."""
import concurrent.futures,fcntl,json,os,queue,signal,subprocess,threading,time
from pathlib import Path
R=Path(__file__).resolve().parent;B=R/'download/qwen32_resume_bundle';RUN=B/'models/gemma2_27b/all';OUT=RUN/'judge_qwen32';D=R/'gemma_dual_gpu';D.mkdir(exist_ok=True)
H=json.loads((R/'handoff.json').read_text());N=8;STOP=threading.Event();LOCK=threading.Lock();CHILDREN={}
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def valid(x,stage):return x.get('valid') is True and (isinstance(x.get('equivalent'),bool) if stage=='reconstruction' else x.get('label') in ['full','partial','none','uncertain'])
def info(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split()
 return {'pid':pid,'starttime':s[19],'state':s[0],'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
def shardpath(stage,i):return OUT/f'{stage}.shard-{i:02d}-of-{N:02d}.jsonl'
def check(i):
 expected={x['key'] for x in SOURCE[i::N]};result={}
 for stage in ['reconstruction','fulfillment']:
  seen=set()
  for x in rows(shardpath(stage,i)):
   assert x['key'] in expected and LOOKUP[x['key']]==x['source_response_sha256']
   if valid(x,stage):seen.add(x['key'])
  result[stage]=len(expected-seen)
 return result

def worker(gpu,jobs):
 while not STOP.is_set():
  try:i=jobs.get_nowait()
  except queue.Empty:return
  for attempt in range(1,4):
   if not any(check(i).values()):break
   while not STOP.is_set():
    free=float(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free>=62492.135+(2304 if gpu==0 else 6144):break
    save(D/f'gpu{gpu}.json',{'stage':'waiting_for_memory','shard':i,'free_mib':free,'time':time.time()});STOP.wait(20)
   if STOP.is_set():return
   logpath=D/f'shard{i:02d}.gpu{gpu}.attempt{attempt}.log'
   env=os.environ.copy();env.update(QWEN32_MODEL=H['model'],PYTHON_BIN=H['python'],QWEN32_BATCH=str(2 if gpu==0 else 16),PYTHONPATH=str(R/'runtime_deps'),HF_HOME='/home/ubuntu/342/jinkwon/hf_cache',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
   with logpath.open('a') as log:
    with LOCK:
     child=subprocess.Popen(['bash',str(B/'run_one_gpu.sh'),'gemma2_27b',str(gpu),str(N),str(i)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);CHILDREN[gpu]=child
    status={'stage':'running','model':'gemma2_27b','gpu':gpu,'pid':child.pid,'batch':2 if gpu==0 else 16,'shard':i,'num_shards':N,'attempt':attempt,'log':str(logpath),'time':time.time()}
    save(R/f'gpu{gpu}.json',status);save(D/f'gpu{gpu}.json',status)
    rc=child.wait()
    with LOCK:CHILDREN.pop(gpu,None)
   save(D/f'shard{i:02d}.json',{'shard':i,'gpu':gpu,'returncode':rc,'remaining':check(i),'attempt':attempt,'time':time.time()})
   if STOP.is_set():return
   if not any(check(i).values()):break
   STOP.wait(3)

def stop(sig,frame):
 STOP.set()
 with LOCK:
  for child in CHILDREN.values():
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass

def main():
 global SOURCE,LOOKUP
 lock=(D/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 parent=info(json.loads((R/'process.json').read_text())['pid']);assert parent and 'run_assigned.py' in parent['command']
 os.kill(parent['pid'],signal.SIGSTOP)
 try:
  old=[]
  for gpu in [0,1]:
   state=json.loads((R/f'gpu{gpu}.json').read_text());p=info(state['pid'])
   assert p and ('qwen32_dual_judge.py' in p['command'] or 'gpu0_batch_trial.py' in p['command'])
   assert os.getpgid(p['pid'])==p['pid'];old.append(p);os.killpg(p['pid'],signal.SIGTERM)
  save(D/'handoff.json',{'parent':parent,'stopped_children':old,'time':time.time()})
  for p in old:
   deadline=time.time()+45
   while (v:=info(p['pid'])) and v['state']!='Z':
    if time.time()>deadline:raise RuntimeError('Child did not terminate')
    time.sleep(1)
  SOURCE=rows(RUN/'responses.jsonl');LOOKUP={x['key']:x['response_sha256'] for x in SOURCE};assert len(SOURCE)==len(LOOKUP)==10592
  protocol=json.loads((OUT/'protocol.json').read_text())
  for stage in ['reconstruction','fulfillment']:
   source=OUT/f'{stage}.jsonl';data=rows(source)
   backup=D/f'{stage}.before_split.jsonl'
   if backup.exists():raise RuntimeError('Snapshot already exists; manual resume required')
   backup.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
   existing={x['key']:x for x in data if valid(x,stage)}
   for k,x in existing.items():assert LOOKUP[k]==x['source_response_sha256']
   for i in range(N):
    dst=shardpath(stage,i);assert not dst.exists()
    dst.write_text(''.join(json.dumps(existing[x['key']],ensure_ascii=False)+'\n' for x in SOURCE[i::N] if x['key'] in existing))
  for i in range(N):save(OUT/f'protocol.shard-{i:02d}-of-{N:02d}.json',protocol)
  save(D/'status.json',{'stage':'running','shards':N,'initial_remaining':{i:check(i) for i in range(N)},'time':time.time()})
  jobs=queue.Queue()
  for i in range(N):jobs.put(i)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(worker,gpu,jobs) for gpu in [0,1]]
   for f in futures:f.result()
  if STOP.is_set():raise RuntimeError('Stopped before merge; preserve shards for resume')
  coverage={}
  for stage in ['reconstruction','fulfillment']:
   merged={}
   for i in range(N):
    for x in rows(shardpath(stage,i)):
     assert LOOKUP[x['key']]==x['source_response_sha256']
     if valid(x,stage):merged[x['key']]=x
   tmp=OUT/f'{stage}.merged.tmp';tmp.write_text(''.join(json.dumps(merged[x['key']],ensure_ascii=False)+'\n' for x in SOURCE if x['key'] in merged));tmp.replace(OUT/f'{stage}.jsonl')
   coverage[stage]=len(merged)
  save(D/'status.json',{'stage':'complete' if all(n==10592 for n in coverage.values()) else 'needs_review','coverage':coverage,'time':time.time()})
 finally:
  current=info(parent['pid'])
  if current and current['starttime']==parent['starttime']:os.kill(parent['pid'],signal.SIGCONT)
  save(D/'parent_restored.json',{'time':time.time(),'parent':parent})
if __name__=='__main__':main()
