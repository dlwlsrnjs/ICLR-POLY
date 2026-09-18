#!/usr/bin/env python3
"""Restore pinned model snapshots into a dedicated cache; no inference."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--models',nargs='+',default=['qwen25_14b','qwen25_32b','gemma2_27b','mistral24b','phi3_medium_14b']);a=p.parse_args()
from huggingface_hub import snapshot_download
lock=json.loads((Path(__file__).parent/'model_revisions.json').read_text())
entries=[lock[t] for t in a.models]+[lock['qwen25_7b'],{'model':'Qwen/Qwen3Guard-Gen-8B','revision':'4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb'}]
for entry in entries:
 repo=entry['model'];rev=entry['revision'];assert rev
 model_dir=a.cache/('models--'+repo.replace('/','--'));ref=model_dir/'refs/main'
 if ref.exists() and ref.read_text().strip()!=rev:raise RuntimeError(f'Use a dedicated cache; mismatched main ref: {repo}')
 dest=Path(snapshot_download(repo_id=repo,revision=rev,cache_dir=str(a.cache)))
 for name,digest in entry.get('artifact_sha256',{}).items():
  assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest, (repo,name)
 ref.parent.mkdir(parents=True,exist_ok=True);ref.write_text(rev)
 print(repo,rev,'verified',flush=True)
