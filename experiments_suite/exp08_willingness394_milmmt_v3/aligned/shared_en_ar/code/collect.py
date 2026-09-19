#!/usr/bin/env python3
"""Collect the frozen benign jobs with the panel's user-only message contract."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from grid_contract import digest
from run_io import load_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gpu', default='1', help='Comma-separated physical GPU ids exposed to vLLM')
    p.add_argument('--model-path', type=Path, help='Optional cached snapshot directory; basename must equal pinned revision')
    p.add_argument('--memory-utilization', type=float, default=.35)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-num-batched-tokens', type=int, default=2048)
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--cpu-offload-gb', type=float, default=0)
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--limit', type=int, help='Optional diagnostic response cap; partial runs cannot be summarized')
    a = p.parse_args()
    visible = [value.strip() for value in a.gpu.split(',') if value.strip()]
    if (not visible or any(not value.isdigit() for value in visible) or
            a.tensor_parallel_size < 1 or a.tensor_parallel_size > len(visible) or
            a.cpu_offload_gb < 0 or a.batch_size < 1 or
            not 0 < a.memory_utilization <= 1 or
            (a.limit is not None and a.limit < 1)):
        p.error('Invalid runtime limits')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(visible)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    with (a.run/'collection.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, jobs, previous = load_run(a.run)
        if not jobs: raise ValueError('No eligible items: prepare the required QA-accepted translations first')
        config = manifest['config']
        if a.model_path and a.model_path.resolve().name != config['target_revision']:
            raise ValueError('Local model path is not the pinned snapshot')
        done = {x['key'] for x in previous}
        todo = [x for x in jobs if x['key'] not in done]
        if a.limit: todo = todo[:a.limit]
        if not todo: print('No pending jobs'); return
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams
        model = str(a.model_path) if a.model_path else config['target_model']
        revision = {} if a.model_path else {'revision': config['target_revision'], 'tokenizer_revision': config['target_revision']}
        pinned_revision = config['target_revision'] if not a.model_path else None
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True,
                                            revision=pinned_revision,
                                            trust_remote_code=a.trust_remote_code)
        hf_config = AutoConfig.from_pretrained(model, local_files_only=True,
                                               revision=pinned_revision,
                                               trust_remote_code=a.trust_remote_code)
        hf_overrides = None
        if getattr(hf_config, 'head_dim', None) is None and getattr(hf_config, 'model_type', None) == 'mistral':
            hf_overrides = {'head_dim': hf_config.hidden_size // hf_config.num_attention_heads}
        metadata = {'config_sha256': manifest['config_sha256'], 'jobs_sha256': manifest['jobs_sha256'],
                    'packages': {n: importlib.metadata.version(n) for n in ['vllm','torch','transformers']},
                    'chat_template_sha256': digest(tok.chat_template), 'sampling': config['sampling'],
                    'target_revision': config['target_revision'], 'max_model_len': config['max_model_len'],
                    'dtype': 'bfloat16', 'tensor_parallel_size': a.tensor_parallel_size,
                    'physical_gpus': visible, 'cpu_offload_gb': a.cpu_offload_gb,
                    'trust_remote_code': a.trust_remote_code, 'hf_overrides': hf_overrides,
                    'system_message': None,
                    'max_num_batched_tokens': a.max_num_batched_tokens, 'memory_utilization': a.memory_utilization,
                    'max_num_seqs': a.batch_size,
                    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        mp = a.run/'collection_metadata.json'
        if mp.exists() and json.loads(mp.read_text()) != metadata: raise ValueError('Incompatible resume; use a new run')
        if previous and not mp.exists(): raise ValueError('Responses exist without runtime provenance')
        for job in todo:
            ids = tok.apply_chat_template(job['messages'], tokenize=True, add_generation_prompt=True)
            if len(ids) + config['sampling']['max_tokens'] > config['max_model_len']:
                raise ValueError('Context overflow; do not truncate the frozen request')
        mp.write_text(json.dumps(metadata, indent=2)+'\n')
        llm_kwargs = {}
        if a.cpu_offload_gb:
            llm_kwargs['cpu_offload_gb'] = a.cpu_offload_gb
        if hf_overrides:
            llm_kwargs['hf_overrides'] = hf_overrides
        llm = LLM(model=model, **revision, dtype='bfloat16',
                  tensor_parallel_size=a.tensor_parallel_size,
                  max_model_len=config['max_model_len'], gpu_memory_utilization=a.memory_utilization,
                  max_num_seqs=a.batch_size, max_num_batched_tokens=a.max_num_batched_tokens,
                  enforce_eager=True, disable_log_stats=True,
                  trust_remote_code=a.trust_remote_code, **llm_kwargs)
        params = SamplingParams(**config['sampling'])
        with (a.run/'responses.jsonl').open('a') as out:
            for start in range(0, len(todo), a.batch_size):
                batch = todo[start:start+a.batch_size]
                outputs = llm.chat([x['messages'] for x in batch], params, use_tqdm=False)
                for job, output in zip(batch, outputs):
                    g = output.outputs[0]
                    row = dict(job, response=g.text, response_sha256=hashlib.sha256(g.text.encode()).hexdigest(),
                               finish_reason=g.finish_reason, output_tokens=len(g.token_ids), model=config['target_model'],
                               revision=config['target_revision'])
                    out.write(json.dumps(row, ensure_ascii=False)+'\n')
                out.flush()
                print('collected', min(start+a.batch_size, len(todo)), '/', len(todo), flush=True)


if __name__ == '__main__': main()
