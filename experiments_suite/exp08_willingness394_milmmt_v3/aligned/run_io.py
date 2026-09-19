"""Run identity checks shared by collection, annotation and aggregation."""
import hashlib
import json
import re
from grid_contract import digest


def read_rows(path):
    if not path.exists(): return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def indexed(rows):
    result = {}
    for row in rows:
        if row['key'] in result: raise ValueError('Duplicate observation key')
        result[row['key']] = row
    return result


def load_run(root, complete=False):
    manifest = json.loads((root/'manifest.json').read_text())
    jobs = read_rows(root/'jobs.jsonl')
    if digest(jobs) != manifest['jobs_sha256'] or digest(manifest['config']) != manifest['config_sha256']:
        raise ValueError('Run manifest does not match inputs')
    ji = indexed(jobs)
    responses = read_rows(root/'responses.jsonl')
    ri = indexed(responses)
    if not set(ri) <= set(ji): raise ValueError('Unknown response key')
    for key, row in ri.items():
        if any(row.get(k) != v for k, v in ji[key].items()):
            raise ValueError('Response contains modified prompt/item/frame metadata')
        if row.get('response_sha256') != hashlib.sha256(row['response'].encode()).hexdigest():
            raise ValueError('Response hash mismatch')
    if complete and (not jobs or set(ri) != set(ji)):
        raise ValueError('Complete, nonempty five-frame response collection required')
    return manifest, jobs, responses


def answer_section(response):
    if response.count('[RECONSTRUCTED]') != 1 or response.count('[ANSWER]') != 1:
        return ''
    if response.index('[ANSWER]') < response.index('[RECONSTRUCTED]'):
        return ''
    return response.split('[ANSWER]', 1)[1].strip()
