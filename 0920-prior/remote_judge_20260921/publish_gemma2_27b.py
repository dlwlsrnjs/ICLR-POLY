#!/usr/bin/env python3
"""Publish a frozen Gemma2-27B result after the judge queue finishes its retries."""
import argparse,fcntl,hashlib,json,os,shutil,time
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
R=Path(__file__).resolve().parent
REPO='jin-kwon/poly-qwen32-judge-resume-20260920'
PREFIX='results/gpu_server_20260921/gemma2_27b'
MODEL='gemma2_27b'
SOURCE=R/'download/qwen32_resume_bundle/models'/MODEL/'all'
PUB=R/'publish_gemma2_27b'
PUB.mkdir(exist_ok=True)
def save(name,data):
 p=PUB/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ready():
 return json.loads((R/'gemma_external_handoff.json').read_text())
def build(result):
 folder=PUB/'package';folder.mkdir(exist_ok=True)
 for src in SOURCE.rglob('*'):
  if not src.is_file():continue
  dst=folder/'all'/src.relative_to(SOURCE);dst.parent.mkdir(parents=True,exist_ok=True)
  before=sha(src);shutil.copy2(src,dst)
  if sha(src)!=before or sha(dst)!=before:raise RuntimeError('Source changed while copying')
 shutil.copy2(R/'download/qwen32_resume_bundle/code/qwen32_dual_judge.py',folder/'qwen32_dual_judge.py')
 rows=[json.loads(l) for l in (folder/'all/responses.jsonl').read_text().splitlines() if l.strip()]
 expected={x['key']:x['response_sha256'] for x in rows}
 assert len(rows)==len(expected)==10592
 coverage={}
 for stage in ['reconstruction','fulfillment']:
  p=folder/'all/judge_qwen32'/f'{stage}.jsonl';valid=set();seen=set();invalid_attempts=0
  if p.exists():
   for l in p.read_text().splitlines():
    x=json.loads(l);assert expected.get(x['key'])==x['source_response_sha256'];seen.add(x['key'])
    typed=isinstance(x.get('equivalent'),bool) if stage=='reconstruction' else x.get('label') in ['full','partial','none','uncertain']
    if x.get('valid') is True and typed:valid.add(x['key'])
    else:invalid_attempts+=1
  coverage[stage]={'valid_unique':len(valid),'seen_unique':len(seen),'unresolved':len(expected)-len(valid),'invalid_attempts_in_history':invalid_attempts}
 complete=all(x['unresolved']==0 for x in coverage.values())
 meta={'model':MODEL,'complete':complete,'coverage':coverage,'handoff':result,'source_dataset':REPO,'source_revision':'1168515989bc8ca9da902dc340670086d4eab39e','source_bundle_sha256':'b28d1cf8ce5afa17a03ac4d004e029873d1cf0f55027fdcd167ac2a94ed9820e','judge_code_commit':'bb695819dd7f106a399eb487399e207cece23b4c','judge_revision':'5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd','batch_history':{'GPU0':[2,4,8,4,2],'GPU1':[16]},'torch':'2.6.0+cu124','transformers':'4.57.6','accelerate':'1.10.1','published_at_unix':time.time()}
 (folder/'SUMMARY.json').write_text(json.dumps(meta,indent=2)+'\n')
 (folder/'README.md').write_text('# Gemma2-27B Qwen32 judge results\n\nStatus: '+('COMPLETE' if complete else 'PARTIAL HANDOFF: intentionally stopped locally; resume fulfillment on destination server')+'\n\nSee SUMMARY.json for unique valid coverage. Historical invalid attempts remain in JSONL. Deduplicate by key using a valid, schema-correct record whose source_response_sha256 matches responses.jsonl. Do not count invalid or missing labels as successes. Both stages must be checked separately.\n\nall/ includes the 10592 original responses, manifest, judge protocol, and reconstruction/fulfillment outputs. The exact judging code is also included. Input provenance is pinned in SUMMARY.json. This is the external-server 10592-row experiment, separate from the older 7503-response audit.\n\nVerify files against SHA256SUMS.json after downloading. The receiving server should preserve its own outputs and copy this result to a separate directory before merging.\n')
 shutil.copy2(R/'resume_gemma_remote.py',folder/'resume.py')
 shutil.copy2(R/'gemma_external_handoff.json',folder/'HANDOFF.json')
 with (folder/'README.md').open('a') as f:
  f.write('\n## Resume on another server\n\nUse a fresh directory. The canonical all/judge_qwen32 JSONLs contain the union of valid shard judgments. Use the unsharded command below to skip all finished reconstruction and fulfillment judgments. Do not resume old shard files in parallel with this command.\n\n```bash\npython resume.py --model /path/to/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --gpu 0 --batch 16\n```\n\nRequires compatible torch, transformers and accelerate. resume.py checks immutable input/code hashes and preserves the previous protocol before adapting only the checkpoint path. Original SHA256SUMS describes the delivered snapshot; output hashes change after further judging.\n')
 checks={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
 (folder/'SHA256SUMS.json').write_text(json.dumps(checks,indent=2)+'\n')
 return folder,meta

def once():
 if (PUB/'receipt.json').exists():return True
 pending=PUB/'uploaded_pending_verification.json'
 if pending.exists():info=json.loads(pending.read_text())
 else:
  result=ready()
  if result is None:
   save('status.json',{'stage':'waiting_for_model_and_retries','time':time.time(),'model':MODEL});return False
  folder,meta=build(result)
  save('status.json',{'stage':'uploading','time':time.time(),'coverage':meta['coverage']})
  api=HfApi();assert api.repo_info(REPO,repo_type='dataset').private
  commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(folder),path_in_repo=PREFIX,commit_message='Publish Gemma2-27B judge results from GPU server for external fulfillment continuation')
  info={'repo_id':REPO,'revision':commit.oid,'path':PREFIX,'complete':meta['complete'],'coverage':meta['coverage'],'url':f'https://huggingface.co/datasets/{REPO}/tree/{commit.oid}/{PREFIX}'}
  save('uploaded_pending_verification.json',info)
 folder=PUB/'package';checks=json.loads((folder/'SHA256SUMS.json').read_text());checks['SHA256SUMS.json']=sha(folder/'SHA256SUMS.json')
 for relative,digest in checks.items():
  downloaded=Path(hf_hub_download(repo_id=REPO,repo_type='dataset',revision=info['revision'],filename=PREFIX+'/'+relative))
  assert sha(downloaded)==digest,relative
 info.update(verified_files=len(checks),verified_at=time.time());save('receipt.json',info)
 save('status.json',{'stage':'uploaded_verified','time':time.time(),**info});return True

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');a=parser.parse_args()
 lock=(PUB/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  try:
   if once():return
  except Exception as exc:
   save('status.json',{'stage':'error_retrying','error_type':type(exc).__name__,'error':str(exc)[:600],'time':time.time()})
   print(type(exc).__name__,str(exc)[:600],flush=True)
   if a.once:raise
  if a.once:return
  time.sleep(30)
if __name__=='__main__':main()
