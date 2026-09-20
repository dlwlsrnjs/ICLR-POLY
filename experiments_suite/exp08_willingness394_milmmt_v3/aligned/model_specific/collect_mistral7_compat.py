#!/usr/bin/env python3
"""Collect Mistral-7B with its missing HF head_dim supplied to vLLM 0.8.5."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid_contract import digest
from run_io import load_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--memory-utilization', type=float, default=.90)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--head-dim', type=int, default=128)
    a = p.parse_args()
    if a.head_dim < 1 or a.batch_size < 1 or not 0 < a.memory_utilization <= 1:
        p.error('Invalid runtime setting')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    with (a.run/'collection.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, jobs, previous = load_run(a.run)
        config = manifest['config']
        if config['model_tag'] != 'mistral7b':
            raise ValueError('This compatibility runner is restricted to mistral7b')
        if a.model_path.resolve().name != config['target_revision']:
            raise ValueError('Local model path is not the pinned snapshot')
        done = {row['key'] for row in previous}
        todo = [row for row in jobs if row['key'] not in done]
        if not todo:
            print('No pending jobs')
            return
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
        overrides = {'head_dim': a.head_dim}
        metadata = {
            'config_sha256': manifest['config_sha256'],
            'jobs_sha256': manifest['jobs_sha256'],
            'packages': {name: importlib.metadata.version(name) for name in ['vllm','torch','transformers']},
            'chat_template_sha256': digest(tokenizer.chat_template),
            'sampling': config['sampling'],
            'target_revision': config['target_revision'],
            'max_model_len': config['max_model_len'],
            'dtype': 'bfloat16',
            'tensor_parallel_size': 1,
            'system_message': None,
            'hf_overrides': overrides,
            'compatibility_reason': 'Pinned Mistral config has head_dim=null; 4096 hidden_size / 32 attention heads = 128.',
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        metadata_path = a.run/'collection_metadata.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError('Incompatible resume; use a new run')
        if previous and not metadata_path.exists():
            raise ValueError('Responses exist without runtime provenance')
        for job in todo:
            ids = tokenizer.apply_chat_template(job['messages'], tokenize=True, add_generation_prompt=True)
            if len(ids) + config['sampling']['max_tokens'] > config['max_model_len']:
                raise ValueError('Context overflow; do not truncate the frozen request')
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
        llm = LLM(
            model=str(a.model_path),
            dtype='bfloat16',
            tensor_parallel_size=1,
            max_model_len=config['max_model_len'],
            gpu_memory_utilization=a.memory_utilization,
            max_num_seqs=a.batch_size,
            enforce_eager=True,
            disable_log_stats=True,
            hf_overrides=overrides,
        )
        params = SamplingParams(**config['sampling'])
        with (a.run/'responses.jsonl').open('a') as output:
            for start in range(0, len(todo), a.batch_size):
                batch = todo[start:start+a.batch_size]
                results = llm.chat([row['messages'] for row in batch], params, use_tqdm=False)
                for job, result in zip(batch, results):
                    generation = result.outputs[0]
                    row = dict(
                        job,
                        response=generation.text,
                        response_sha256=hashlib.sha256(generation.text.encode()).hexdigest(),
                        finish_reason=generation.finish_reason,
                        output_tokens=len(generation.token_ids),
                        model=config['target_model'],
                        revision=config['target_revision'],
                    )
                    output.write(json.dumps(row, ensure_ascii=False)+'\n')
                output.flush()
                print('collected', min(start+a.batch_size, len(todo)), '/', len(todo), flush=True)


if __name__ == '__main__':
    main()
