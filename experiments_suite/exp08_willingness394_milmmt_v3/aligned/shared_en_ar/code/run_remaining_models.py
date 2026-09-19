#!/usr/bin/env python3
"""Stage the final large models after earlier queues, collecting one at a time."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MODELS = [
    {
        'tag': 'phi3_medium_14b',
        'repo_id': 'microsoft/Phi-3-medium-4k-instruct',
        'revision': '48d87cd5e0523b430d77e93becd3655cd6897230',
        'gpu': '3', 'tp': 1, 'min_free_mib': 30000, 'trust_remote_code': True,
    },
    {
        'tag': 'mistral24b',
        'repo_id': 'mistralai/Mistral-Small-24B-Instruct-2501',
        'revision': '9527884be6e5616bdd54de542f9ae13384489724',
        'gpu': '2,3', 'tp': 2, 'min_free_mib': 40000, 'trust_remote_code': False,
    },
    {
        'tag': 'gemma2_27b',
        'repo_id': 'google/gemma-2-27b-it',
        'revision': 'aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0',
        'gpu': '2,3', 'tp': 2, 'min_free_mib': 40000, 'trust_remote_code': False,
    },
]


CLEANUP_AFTER_COMPLETE = [
    ('qwen25_3b', 'models--Qwen--Qwen2.5-3B-Instruct'),
    ('llama32_3b_it', 'models--meta-llama--Llama-3.2-3B-Instruct'),
    ('gemma2_2b_it', 'models--google--gemma-2-2b-it'),
    ('falcon3_3b', 'models--tiiuae--Falcon3-3B-Instruct'),
    ('mistral7b', 'models--mistralai--Mistral-7B-Instruct-v0.3'),
    ('falcon3_7b', 'models--tiiuae--Falcon3-7B-Instruct'),
    ('llama31_8b_it', 'models--meta-llama--Llama-3.1-8B-Instruct'),
    ('gemma2_9b_it', 'models--google--gemma-2-9b-it'),
    ('falcon3_10b', 'models--tiiuae--Falcon3-10B-Instruct'),
    ('glm4_9b', 'models--THUDM--glm-4-9b-chat-hf'),
    ('qwen25_14b', 'models--Qwen--Qwen2.5-14B-Instruct'),
    ('qwen25_32b', 'models--Qwen--Qwen2.5-32B-Instruct'),
]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def rows(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def complete(run, tag):
    root = run/'panel'/tag
    return rows(root/'jobs.jsonl') > 0 and rows(root/'jobs.jsonl') == rows(root/'responses.jsonl')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def safe_remove_cache(hub, cache):
    target = hub/cache
    if target.parent != hub or not target.name.startswith('models--'):
        raise ValueError(f'Unsafe cache target: {target}')
    if target.exists():
        shutil.rmtree(target)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--hf-cli', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--token-path', type=Path, required=True)
    parser.add_argument('--after-pid', type=int,
                        help='Wait for this predecessor when launched independently')
    args = parser.parse_args()

    run = args.run.resolve()
    repo = args.repo.resolve()
    python = Path(os.path.abspath(args.python))
    hf_cli = Path(os.path.abspath(args.hf_cli))
    hf_home = args.hf_home.resolve()
    token_path = args.token_path.resolve()
    hub = hf_home/'hub'
    if (not run.is_dir() or not repo.is_dir() or not python.is_file() or
            not hf_cli.is_file() or not hub.is_dir() or not token_path.is_file()):
        parser.error('Invalid run, repository, executable, HF cache, or token path')

    status_path = run/'pipeline_status_remaining.json'
    completed = []
    failures = []
    removed = []

    def status(stage, **extra):
        value = {
            'stage': stage,
            'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'completed_models': completed,
            'failed_models': failures,
            'removed_completed_model_caches': removed,
            **extra,
        }
        atomic_json(status_path, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    if args.after_pid is not None:
        status('waiting_for_predecessor', predecessor_pid=args.after_pid)
        while alive(args.after_pid):
            time.sleep(20)

    status('cleaning_completed_caches')
    for tag, cache in CLEANUP_AFTER_COMPLETE:
        if complete(run, tag) and safe_remove_cache(hub, cache):
            removed.append(tag)
            status('cleaning_completed_caches', last_removed=tag)

    downloader_environment = dict(
        os.environ,
        HF_HOME=str(hf_home),
        HF_TOKEN_PATH=str(token_path),
        HF_HUB_DISABLE_XET='0',
    )
    runner = Path(__file__).with_name('run_panel_queue.py')
    exclude = ['*.bin', 'original/*', '*.gguf', '*.h5', '*.msgpack', '*.ot',
               'consolidated*.safetensors']
    for model in MODELS:
        tag = model['tag']
        cache = 'models--' + model['repo_id'].replace('/', '--')
        snapshot = hub/cache/'snapshots'/model['revision']
        status('downloading', model=tag, repo_id=model['repo_id'],
               revision=model['revision'])
        command = [str(hf_cli), 'download', model['repo_id'], '--revision', model['revision'],
                   '--exclude', *exclude]
        download = subprocess.run(command, env=downloader_environment)
        if download.returncode or not snapshot.is_dir():
            failures.append({'model': tag, 'stage': 'download',
                             'returncode': download.returncode})
            status('download_failed', model=tag, returncode=download.returncode)
            continue

        command = [
            str(python), '-u', str(runner), '--run', str(run), '--repo', str(repo),
            '--python', str(python), '--gpu', model['gpu'],
            '--status-file', f'pipeline_status_{tag}.json',
            '--memory-utilization', '.85', '--batch-size', '8',
            '--max-num-batched-tokens', '2048', '--tensor-parallel-size', str(model['tp']),
            '--min-free-mib', str(model['min_free_mib']), '--continue-on-error',
            '--model', f'{tag}={snapshot}',
        ]
        if model['trust_remote_code']:
            command.extend(['--trust-remote-code-tag', tag])
        status('collecting', model=tag, gpu=model['gpu'], tensor_parallel_size=model['tp'])
        collection = subprocess.run(command)
        if complete(run, tag):
            completed.append(tag)
            if safe_remove_cache(hub, cache):
                removed.append(tag)
            status('model_complete', model=tag,
                   responses=rows(run/'panel'/tag/'responses.jsonl'))
        else:
            failures.append({'model': tag, 'stage': 'collection',
                             'returncode': collection.returncode,
                             'responses': rows(run/'panel'/tag/'responses.jsonl')})
            status('model_failed', model=tag, returncode=collection.returncode)
    status('remaining_queue_complete', requested_models=len(MODELS))
    return 0 if not failures else 2


if __name__ == '__main__':
    sys.exit(main())
