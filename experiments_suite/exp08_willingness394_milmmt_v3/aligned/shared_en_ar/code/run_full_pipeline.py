#!/usr/bin/env python3
"""Resume every target queue, stage large models, then run both judges.

The coordinator is intentionally restartable.  Each collector and judge owns its
own file lock and resumes from durable JSONL rows, so restarting this process
does not duplicate completed responses.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


GROUPS = [
    {
        'name': 'small', 'gpu': '3', 'tp': 1, 'min_free_mib': 30000,
        'tags': ['qwen25_3b', 'llama32_3b_it', 'gemma2_2b_it',
                 'falcon3_3b', 'phi35_mini'],
    },
    {
        'name': 'medium_a', 'gpu': '3', 'tp': 1, 'min_free_mib': 30000,
        'tags': ['mistral7b', 'falcon3_7b', 'llama31_8b_it', 'gemma2_9b_it'],
    },
    {
        'name': 'medium_b', 'gpu': '3', 'tp': 1, 'min_free_mib': 30000,
        'tags': ['falcon3_10b', 'glm4_9b', 'qwen25_14b'],
    },
    {
        'name': 'qwen32', 'gpu': '2,3', 'tp': 2, 'min_free_mib': 40000,
        'tags': ['qwen25_32b'],
    },
]


def rows(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def complete(run, tag):
    root = run/'panel'/tag
    jobs = rows(root/'jobs.jsonl')
    return jobs > 0 and jobs == rows(root/'responses.jsonl')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def cache_name(repo_id):
    return 'models--' + repo_id.replace('/', '--')


def locate_snapshot(cache_roots, repo_id, revision):
    for root in cache_roots:
        candidate = root/cache_name(repo_id)/'snapshots'/revision
        if candidate.is_dir():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--hf-cli', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--token-path', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, action='append', required=True,
                        help='Repeat with each Hugging Face hub directory')
    parser.add_argument('--reconstruction-model', type=Path, required=True)
    parser.add_argument('--wildguard-model', type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    repo = args.repo.resolve()
    python = Path(os.path.abspath(args.python))
    hf_cli = Path(os.path.abspath(args.hf_cli))
    hf_home = args.hf_home.resolve()
    token_path = args.token_path.resolve()
    cache_roots = [path.resolve() for path in args.cache_root]
    reconstruction_model = args.reconstruction_model.resolve()
    wildguard_model = args.wildguard_model.resolve()
    if (not run.is_dir() or not repo.is_dir() or not python.is_file() or
            not hf_cli.is_file() or not hf_home.is_dir() or
            not token_path.is_file() or not all(path.is_dir() for path in cache_roots) or
            not reconstruction_model.is_dir() or not wildguard_model.is_dir()):
        parser.error('Invalid run, repository, executable, cache, token, or judge model')

    package = Path(__file__).resolve().parents[1]
    configs = {
        path.stem: json.loads(path.read_text())
        for path in sorted((package/'configs').glob('*.json'))
    }
    status_path = run/'pipeline_status_full.json'
    failures = []

    def status(stage, **extra):
        value = {
            'stage': stage,
            'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'completed_models': sorted(tag for tag in configs if complete(run, tag)),
            'failed_stages': failures,
            **extra,
        }
        atomic_json(status_path, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    queue = Path(__file__).with_name('run_panel_queue.py')
    for group in GROUPS:
        specifications = []
        missing = []
        for tag in group['tags']:
            if complete(run, tag):
                continue
            config = configs[tag]
            snapshot = locate_snapshot(
                cache_roots, config['target_model'], config['target_revision'])
            if snapshot is None:
                missing.append(tag)
            else:
                specifications.append(f'{tag}={snapshot}')
        if missing:
            failures.append({'stage': group['name'], 'missing_snapshots': missing})
            status('snapshot_missing', group=group['name'], missing_models=missing)
            return 2
        if not specifications:
            status('group_already_complete', group=group['name'])
            continue
        command = [
            str(python), '-u', str(queue), '--run', str(run), '--repo', str(repo),
            '--python', str(python), '--gpu', group['gpu'],
            '--status-file', f"pipeline_status_{group['name']}.json",
            '--memory-utilization', '.85', '--batch-size', '8',
            '--max-num-batched-tokens', '2048',
            '--tensor-parallel-size', str(group['tp']),
            '--min-free-mib', str(group['min_free_mib']), '--continue-on-error',
        ]
        for specification in specifications:
            command.extend(['--model', specification])
        for tag in group['tags']:
            if tag in {'phi35_mini', 'glm4_9b'}:
                command.extend(['--trust-remote-code-tag', tag])
        status('running_group', group=group['name'], models=specifications,
               gpu=group['gpu'])
        result = subprocess.run(command)
        if result.returncode:
            failures.append({'stage': group['name'], 'returncode': result.returncode})
            status('group_failed', group=group['name'], returncode=result.returncode)

    remaining = Path(__file__).with_name('run_remaining_models.py')
    status('running_large_model_stager')
    result = subprocess.run([
        str(python), '-u', str(remaining), '--run', str(run), '--repo', str(repo),
        '--python', str(python), '--hf-cli', str(hf_cli), '--hf-home', str(hf_home),
        '--token-path', str(token_path),
    ])
    if result.returncode:
        failures.append({'stage': 'remaining_models', 'returncode': result.returncode})
        status('large_model_stager_failed', returncode=result.returncode)

    judges = Path(__file__).with_name('run_judge_queue.py')
    status('running_judges')
    result = subprocess.run([
        str(python), '-u', str(judges), '--run', str(run), '--repo', str(repo),
        '--python', str(python), '--reconstruction-model', str(reconstruction_model),
        '--wildguard-model', str(wildguard_model), '--reconstruction-gpu', '2',
        '--wildguard-gpu', '3',
    ])
    if result.returncode:
        failures.append({'stage': 'judges', 'returncode': result.returncode})
        status('judges_failed', returncode=result.returncode)

    status('pipeline_complete' if not failures else 'pipeline_complete_with_failures')
    return 0 if not failures else 2


if __name__ == '__main__':
    sys.exit(main())
