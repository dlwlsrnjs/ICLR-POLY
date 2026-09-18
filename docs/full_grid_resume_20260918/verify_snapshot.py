#!/usr/bin/env python3
"""Verify a restored snapshot without importing ML libraries."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
m=json.loads((a.root/'SNAPSHOT_MANIFEST.json').read_text());bad=[]
for rel,meta in m['files'].items():
 path=a.root/rel
 if not path.is_file() or path.stat().st_size!=meta['bytes']:bad.append(rel);continue
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=meta['sha256']:bad.append(rel)
print('verified',len(m['files'])-len(bad),'/',len(m['files']),'files')
if bad:print('mismatches:',bad)
raise SystemExit(bool(bad))
