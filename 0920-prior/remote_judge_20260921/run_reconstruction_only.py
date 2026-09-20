#!/usr/bin/env python3
import concurrent.futures,fcntl,json,os,queue,signal,subprocess,threading,time
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'reconstruction_only';D.mkdir(exist_ok=True);H=json.loads((R/'handoff.json').read_text());STOP=threading.Event();LOCK=threading.Lock();CHILDREN={}
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def worker(gpu,jobs):
 while not STOP.is_set():
  try:tag=jobs.get_nowait()
  except queue.Empty:return
  run=R/'download/qwen32_resume_bundle/models'/tag/'all';p=run/'judge_qwen32/reconstruction.jsonl'
  valid={x['key'] for x in map(json.loads,p.read_text().splitlines()) if x.get('valid') is True and isinstance(x.get('equivalent'),bool)} if p.exists() else set()
  if len(valid)==10592:continue
  while not STOP.is_set():
   free=float(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
   if free>=62492.135+(2304 if gpu==0 else 6144):break
   save(D/f'gpu{gpu}.json',{'stage':'waiting_for_memory','model':tag,'free_mib':free,'time':time.time()});STOP.wait(20)
  if STOP.is_set():return
  env=os.environ.copy();env.update(PYTHONPATH=str(R/'runtime_deps'),HF_HOME='/home/ubuntu/342/jinkwon/hf_cache',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
  cmd=[H['python'],str(R/'judge_one_stage.py'),'--run',str(run),'--model',H['model'],'--stage','reconstruction','--batch',str(2 if gpu==0 else 16),'--gpu',str(gpu),'--code',str(R/'download/qwen32_resume_bundle/code/qwen32_dual_judge.py')]
  logpath=D/f'{tag}.log'
  with logpath.open('a') as log:
   with LOCK:child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);CHILDREN[gpu]=child
   status={'stage':'running','scope':'reconstruction_only','model':tag,'gpu':gpu,'pid':child.pid,'batch':2 if gpu==0 else 16,'log':str(logpath),'time':time.time()};save(D/f'gpu{gpu}.json',status);save(R/f'gpu{gpu}.json',status)
   rc=child.wait()
   with LOCK:CHILDREN.pop(gpu,None)
  save(D/f'{tag}.result.json',{'model':tag,'scope':'reconstruction_only','returncode':rc,'time':time.time()})
def stop(sig,frame):
 STOP.set()
 with LOCK:
  for child in CHILDREN.values():
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass

def main():
 lock=(D/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 jobs=queue.Queue()
 for tag in H['assigned_models']:jobs.put(tag)
 save(D/'status.json',{'stage':'running','scope':'reconstruction_only','pid':os.getpid(),'models':H['assigned_models'],'time':time.time()})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(worker,g,jobs) for g in [0,1]]
  for f in futures:f.result()
 save(D/'status.json',{'stage':'queue_exited_check_model_results','scope':'reconstruction_only','time':time.time()})
if __name__=='__main__':main()
