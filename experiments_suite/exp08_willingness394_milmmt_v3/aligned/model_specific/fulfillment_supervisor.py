#!/usr/bin/env python3
"""Borrow GPU0 at a stage boundary, audit completed responses, resume collection."""
import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path
from fulfillment_audit import save,sha
ROOT=Path(__file__).resolve().parent

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);p.add_argument('--other-queue',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--gpu',type=int,choices=[0,1],default=0);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 lock=(a.out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 original=json.loads((a.queue/'process.json').read_text());save(a.out/'collection_resume.json',original)
 def state(stage,**kw):save(a.out/'supervisor_status.json',dict(stage=stage,time=time.time(),**kw))
 def command(stage,args):
  state(stage)
  with (a.out/(stage+'.log')).open('a') as log:
   subprocess.run([sys.executable,str(ROOT/'fulfillment_audit.py'),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
 def prepare():command('prepare',['prepare','--out',a.out,'--sources',a.queue,a.recovery,a.other_queue])
 def audit(roundname):
  # Allocation safety is rechecked by the judge; wait instead of displacing other users.
  weights=json.loads((a.model/'model.safetensors.index.json').read_text())['metadata']['total_size']/1024**2
  while True:
   free=float(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
   if free>=weights+6144:break
   state('waiting_for_gpu',gpu=a.gpu,free_mib=free,required_mib=weights+6144);time.sleep(15)
  command(roundname,['judge','--out',a.out,'--model',a.model,'--gpu',str(a.gpu)])
 def resume():
  (a.queue/'STOP').unlink(missing_ok=True)
  with (a.queue/'queue.log').open('a') as log:child=subprocess.Popen(original['command'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(a.queue/'process.json',dict(pid=child.pid,command=original['command'],started_at=time.time(),resumed_after='fulfillment audit'))
  save(a.out/'collection_resumed.json',dict(pid=child.pid,time=time.time()));return child.pid
 state('waiting_for_collection_boundary',pid=original['pid'])
 stop=a.queue/'STOP'
 if stop.exists():raise RuntimeError('Another stage handoff already owns STOP')
 stop.write_text('Fulfillment audit owns GPU0 handoff; resume command recorded in audit directory.\n')
 while alive(original['pid']):time.sleep(2)
 resumed=None
 try:
  prepare();audit('judge_round1')
 except Exception as exc:
  state('audit_failed_collection_resuming',error=str(exc));raise
 finally:resumed=resume()
 state('collection_resumed_waiting_for_remaining',pid=resumed)
 while True:
  others=json.loads((a.other_queue/'process.json').read_text())['pid']
  if not alive(resumed) and not alive(others):break
  time.sleep(20)
 prepare();audit('judge_round2');state('finished')
if __name__=='__main__':main()
