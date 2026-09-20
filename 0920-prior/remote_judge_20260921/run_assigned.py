#!/usr/bin/env python3
import concurrent.futures,fcntl,json,os,queue,signal,subprocess,sys,threading,time
from pathlib import Path
R=Path(__file__).resolve().parent
B=R/'download/qwen32_resume_bundle'
H=json.loads((R/'handoff.json').read_text())
MODELS=H['assigned_models']
LOCK=threading.Lock(); CHILDREN={}; STOP=threading.Event()

def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def state(gpu,**kw):
 save(R/f'gpu{gpu}.json',dict(time=time.time(),gpu=gpu,**kw))
def counts(tag):
 run=B/'models'/tag/'all'; source={r['key']:r['response_sha256'] for r in map(json.loads,(run/'responses.jsonl').read_text().splitlines())}
 result={}
 for stage in ['reconstruction','fulfillment']:
  p=run/'judge_qwen32'/f'{stage}.jsonl'; valid=set()
  if p.exists():
   for line in p.read_text().splitlines():
    row=json.loads(line)
    typed=isinstance(row.get('equivalent'),bool) if stage=='reconstruction' else row.get('label') in ['full','partial','none','uncertain']
    if row.get('valid') is True and typed and source.get(row.get('key'))==row.get('source_response_sha256'):valid.add(row['key'])
  result[stage]=len(valid)
 return result

def worker(gpu, jobs):
 batch=2 if gpu==0 else 16
 while not STOP.is_set():
  try:tag=jobs.get_nowait()
  except queue.Empty:return
  result=counts(tag)
  if all(v==10592 for v in result.values()):
   state(gpu,stage='already_complete',model=tag,valid_counts=result);continue
  for attempt in range(1,4):
   while not STOP.is_set():
    free=float(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    required=62492.135+(2304 if gpu==0 else 6144)
    if free>=required:break
    state(gpu,stage='waiting_for_memory',model=tag,free_mib=free,required_mib=required);STOP.wait(30)
   if STOP.is_set():return
   env=os.environ.copy();env.update(QWEN32_MODEL=H['model'],PYTHON_BIN=H['python'],QWEN32_BATCH=str(batch),PYTHONPATH=str(R/'runtime_deps'),HF_HOME='/home/ubuntu/342/jinkwon/hf_cache',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
   cmd=['bash',str(B/'run_one_gpu.sh'),tag,str(gpu)]
   logfile=R/'logs'/f'{tag}.attempt{attempt}.log';logfile.parent.mkdir(exist_ok=True)
   with logfile.open('a') as log:
    with LOCK:
     child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     CHILDREN[gpu]=child
    state(gpu,stage='running',model=tag,pid=child.pid,batch=batch,attempt=attempt,log=str(logfile))
    rc=child.wait()
    with LOCK:CHILDREN.pop(gpu,None)
   result=counts(tag)
   save(R/'results'/f'{tag}.json',dict(model=tag,gpu=gpu,returncode=rc,valid_counts=result,expected_per_stage=10592,complete=all(v==10592 for v in result.values()),time=time.time(),attempt=attempt))
   if rc==0 and all(v==10592 for v in result.values()):break
   if STOP.is_set():return
   
   if rc and 'out of memory' in logfile.read_text(errors='replace').lower(): batch=max(1,batch//2)
   state(gpu,stage='retry_pending',model=tag,returncode=rc,valid_counts=result,next_batch=batch)
   STOP.wait(30)
  state(gpu,stage='model_finished',model=tag,valid_counts=result)
 state(gpu,stage='stopped')

def proc(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return None
 return {'pid':pid,'starttime':(p/'stat').read_text().split(') ',1)[1].split()[19],'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
def handle_signal(sig,frame):
 STOP.set()
 with LOCK:
  for p in CHILDREN.values():
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass

def main():
 lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (R/'results').mkdir(exist_ok=True)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,handle_signal)
 paused=[]
 try:
  for pid,expected in [(3392312,'fulfillment_supervisor.py'),(3381900,'adaptive_queue_worker.py')]:
   info=proc(pid)
   if not info or expected not in info['command']:raise RuntimeError(f'Unexpected old queue PID {pid}')
   children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
   for c in children:
    if Path(f'/proc/{c}/stat').read_text().split(') ',1)[1].split()[0]!='Z':raise RuntimeError(f'Old queue {pid} has active child {c}')
   os.kill(pid,signal.SIGSTOP);paused.append(info)
  save(R/'paused_previous_queues.json',{'time':time.time(),'processes':paused,'restore':'SIGCONT on matching PID and starttime in finally'})
  jobs=queue.Queue()
  for tag in MODELS:jobs.put(tag)
  save(R/'queue_status.json',{'stage':'running','pid':os.getpid(),'models':MODELS,'gpus':[0,1],'started_at':time.time()})
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(worker,gpu,jobs) for gpu in [0,1]]
   for f in futures:f.result()
  results={tag:counts(tag) for tag in MODELS}
  save(R/'queue_status.json',{'stage':'complete' if all(all(v==10592 for v in c.values()) for c in results.values()) else 'incomplete','counts':results,'time':time.time()})
 finally:
  for info in paused:
   current=proc(info['pid'])
   if current and current['starttime']==info['starttime']:os.kill(info['pid'],signal.SIGCONT)
  save(R/'previous_queues_restored.json',{'time':time.time(),'processes':paused})

if __name__=='__main__':main()
