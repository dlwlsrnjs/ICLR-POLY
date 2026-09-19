"""CPU-only consistency verification of the published complete dataset."""
from pathlib import Path
import json,hashlib,sys
from collections import Counter

p=Path(__file__).resolve().parent
sys.path.insert(0,str(p/'code'))
from grid_contract import digest
manifest=json.loads((p/'DATA_MANIFEST.json').read_text())
for name,entry in manifest['files'].items():
 raw=(p/name).read_bytes()
 assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
items=json.loads((p/'data/items.json').read_text())
assert len(items)==331 and len({r['id'] for r in items})==331
assert Counter(r['source_dataset'] for r in items)=={'falsereject':212,'orbench_hard':119}
assert all(r['bare_refuse_count']>=5 and r['bare_refuse_models']==9 for r in items)
originals={r['id']:r['prompt'] for r in items}
rows=[json.loads(l) for l in (p/'data/translations.jsonl').read_text().splitlines() if l.strip()]
assert len(rows)==993
assert {(r['item_id'],r['language']) for r in rows}=={(i,l) for i in originals for l in ['Norwegian','Finnish','Arabic']}
assert all(originals[r['item_id']]==r['english_original'] for r in rows)
configs=list((p/'configs').glob('*.json'));assert len(configs)==17
for f in configs:
 c=json.loads(f.read_text());assert c['languages']==['English','Arabic'] and c['cell']=='g3_ordered_n2'
 assert c['expected_items']==331 and c['corpus_sha256']==digest(items)
 assert c['sampling']['max_tokens']==1024
print('PASS: 331 unique originals; 993 translations; exact English identity; 17 frozen shared-language configs; all data checksums.')
