import argparse,re,json,hashlib
from pathlib import Path
from scenarios import R,load,digest
p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--config',required=True);p.add_argument('--shards',type=int,default=1);a,_=p.parse_known_args()
if a.stage=='collect':
 c,*_=load(a.config);folder=R/'outputs'/c['name'];folder.mkdir(exist_ok=True)
 manifest=json.loads((folder/'build_manifest.json').read_text())
 assert manifest['config_sha256']==digest(c),'config changed; rebuild jobs first'
 for x in manifest['files']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'],'build inputs changed; rebuild jobs first'
 for f in folder.glob('responses.shard*-of-*.jsonl'):
  m=re.search(r'-of-(\d+)\.jsonl$',f.name);assert m and int(m.group(1))==a.shards,'Shard count changed: use a new config name, or resume original shard count'
 path=folder/'shard_layout.json'
 if path.exists():assert json.loads(path.read_text())['shards']==a.shards,'incompatible shard layout'
 path.write_text(json.dumps({'shards':a.shards}))
