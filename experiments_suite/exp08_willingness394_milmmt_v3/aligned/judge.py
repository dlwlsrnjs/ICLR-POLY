#!/usr/bin/env python3
"""Prepare auditable annotation inputs and optionally run one existing judge stage."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from grid_contract import digest
from run_io import load_run, answer_section


def reconstructed(response):
    match = re.search(r'\[RECONSTRUCTED\](.*?)(?:\[ANSWER\]|$)', response, flags=re.S)
    return match[1].strip() if match else ''


def prepare(run, repo):
    manifest, jobs, responses = load_run(run, complete=True)
    judge = run/'judge'
    for folder in ['inputs','outputs','code']: (judge/folder).mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]/'runs/wildguard_labeling_v1/code'
    hashes = {}
    for name in ['batch_label.py','label.py']:
        content = (source/name).read_bytes(); dest = judge/'code'/name
        if dest.exists() and dest.read_bytes() != content: raise ValueError('Judge code changed; use a new run')
        dest.write_bytes(content); hashes[name] = hashlib.sha256(content).hexdigest()
    hashes['judge_reconstruction_equivalence.py'] = hashlib.sha256((repo/'scripts/judge_reconstruction_equivalence.py').read_bytes()).hexdigest()
    inputs = [{'key': x['key'], 'original_request': x['original_request'], 'response': answer_section(x['response']),
               'finish_reason': x['finish_reason']} for x in responses]
    reconstruction_inputs = [{'key': x['key'], 'item_id': x['id'], 'condition': x['cell'],
                               'original': x['original_request'], 'reconstructed': reconstructed(x['response']),
                               'source_response_sha256': x['response_sha256']} for x in responses]
    provenance = {'jobs_sha256': manifest['jobs_sha256'], 'responses_sha256': digest(responses),
                  'judged_response_view': 'ANSWER section only; missing/invalid sections excluded by summarize', 'judge_sources_sha256': hashes, 'input_sha256': digest(inputs),
                  'reconstruction_input_sha256': digest(reconstruction_inputs)}
    mp = judge/'provenance.json'
    if mp.exists() and json.loads(mp.read_text()) != provenance: raise ValueError('Annotation inputs changed')
    (judge/'inputs/jobs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in inputs))
    (judge/'inputs/reconstruction.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in reconstruction_inputs))
    mp.write_text(json.dumps(provenance,indent=2)+'\n')
    return judge


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','reconstruction','wildguard','behavior'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--model-path', type=Path, help='Local judge snapshot, required for inference stages')
    a = p.parse_args()
    judge = prepare(a.run.resolve(), a.repo.resolve())
    if a.stage == 'prepare': print(judge); sys.exit(0)
    if not a.model_path or not a.model_path.exists(): p.error('--model-path must name a local judge snapshot')
    expected_revisions = {'reconstruction':'a09a35458c702b33eeacc393d103063234e8bc28',
                          'wildguard':'cbba4823f3e8020e5a74a5e29bf85072def6f2ff',
                          'behavior':'5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'}
    if a.model_path.resolve().name != expected_revisions[a.stage]:
        raise ValueError('Judge snapshot differs from the frozen protocol')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', VLLM_USE_V1='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    runtime = {'packages': {name: importlib.metadata.version(name) for name in (['torch','transformers'] if a.stage == 'reconstruction' else ['torch','transformers','vllm'])},
               'stage': a.stage, 'model_snapshot': str(a.model_path.resolve()), 'gpu': 1,
               'judge_provenance_sha256': digest(json.loads((judge/'provenance.json').read_text()))}
    if a.stage == 'reconstruction':
        # This is the same equivalence judge as the MJ/LG grid, independently loaded.
        if a.model_path.resolve().name != 'a09a35458c702b33eeacc393d103063234e8bc28':
            raise ValueError('Reconstruction judge must use the panel Qwen2.5-7B snapshot')
        cmd = [sys.executable, str(a.repo.resolve()/'scripts/judge_reconstruction_equivalence.py'),
               '--input', str(judge/'inputs/reconstruction.jsonl'), '--outdir', str(judge/'reconstruction'),
               '--model', str(a.model_path.resolve()), '--device', 'cuda:0', '--batch-size', '16']
    else:
        cmd = [sys.executable, str(judge/'code/batch_label.py'), a.stage, '--model', str(a.model_path.resolve())]
    mp = judge/(a.stage+'.runtime.json')
    if mp.exists() and json.loads(mp.read_text()) != runtime: raise ValueError('Judge runtime changed')
    mp.write_text(json.dumps(runtime,indent=2)+'\n')
    subprocess.run(cmd, env=env, check=True)
