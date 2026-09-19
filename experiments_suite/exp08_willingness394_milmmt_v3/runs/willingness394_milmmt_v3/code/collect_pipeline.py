import json,os,sys,time,subprocess,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R)
config='configs/legacy_394.json';c=json.loads(Path(config).read_text());out=R/'outputs'/c['name'];logs=R/'logs/full_collection';logs.mkdir(exist_ok=True)
lock=(out/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
py=str((R/'../../.venv_prior085/bin/python').resolve());started=time.time();state={'pid':os.getpid(),'started_unix':started,'config':config,'expected_responses':json.loads((out/'plan.json').read_text())['ready_jobs'],'state':'running','completed_stages':[]}
def save():
 tmp=out/'pipeline_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'pipeline_status.json')
steps=[('collect',['bash','run.sh','collect','--config',config]),('wildguard',['bash','judge.sh',config,'wildguard']),('behavior',['bash','judge.sh',config,'behavior']),('summarize',[py,'code/summarize.py','--config',config]),('export',[py,'code/export_prior.py','--config',config])]
try:
 for name,cmd in steps:
  state.update(stage=name,stage_started_unix=time.time());save();print('START',name,flush=True)
  with (logs/(name+'.log')).open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  state['completed_stages'].append(name);save();print('FINISH',name,flush=True)
 state.update(state='complete',finished_unix=time.time());save()
except BaseException as e:
 state.update(state='failed',error=type(e).__name__+': '+str(e),finished_unix=time.time());save();raise
