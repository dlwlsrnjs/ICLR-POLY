#!/usr/bin/env python3
"""Wait for translation QA, prepare the panel, and collect cached target models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def parse_models(values):
    result = []
    for value in values:
        if '=' not in value:
            raise ValueError(f'Expected TAG=/snapshot/path, got {value!r}')
        tag, raw_path = value.split('=', 1)
        path = Path(raw_path).resolve()
        if not tag or not path.is_dir():
            raise ValueError(f'Invalid model specification: {value!r}')
        result.append((tag, path))
    if len({tag for tag, _ in result}) != len(result):
        raise ValueError('Duplicate model tag')
    return result


def free_mib(gpu_ids):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        text=True,
    )
    observed = {}
    for line in output.splitlines():
        index, value = [part.strip() for part in line.split(',', 1)]
        observed[index] = int(value)
    return {gpu: observed[gpu] for gpu in gpu_ids}


def count_rows(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--gpu', default='3', help='Comma-separated physical GPU ids')
    parser.add_argument('--model', action='append', required=True,
                        help='Repeat TAG=/absolute/pinned/snapshot/path')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--after-pid', type=int,
                        help='Do not prepare or collect until this predecessor exits')
    parser.add_argument('--status-file', default='pipeline_status.json',
                        help='Run-relative status JSON basename')
    parser.add_argument('--memory-utilization', type=float, default=.85)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--cpu-offload-gb', type=float, default=0)
    parser.add_argument('--min-free-mib', type=int, default=30000)
    parser.add_argument('--trust-remote-code-tag', action='append', default=[])
    parser.add_argument('--continue-on-error', action='store_true')
    args = parser.parse_args()

    run = args.run.resolve()
    repo = args.repo.resolve()
    # Keep a virtualenv launcher path intact. Path.resolve() follows its symlink
    # to the system interpreter and silently drops the virtualenv site-packages.
    python = Path(os.path.abspath(args.python))
    package = Path(__file__).resolve().parents[1]
    models = parse_models(args.model)
    gpu_ids = [value.strip() for value in args.gpu.split(',') if value.strip()]
    if (not run.is_dir() or not repo.is_dir() or not python.is_file() or
            not gpu_ids or any(not value.isdigit() for value in gpu_ids) or
            args.tensor_parallel_size != len(gpu_ids) or
            not 0 < args.memory_utilization <= 1 or args.batch_size < 1 or
            args.max_num_batched_tokens < 1 or args.cpu_offload_gb < 0 or
            args.min_free_mib < 1):
        parser.error('Invalid run, executable, GPU, or runtime setting')
    environment_probe = subprocess.run(
        [str(python), '-c', 'import transformers, vllm'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if environment_probe.returncode:
        parser.error(f'Inference environment probe failed: {environment_probe.stderr.strip()}')

    configs = {}
    for path in sorted((package/'configs').glob('*.json')):
        configs[path.stem] = (path, json.loads(path.read_text()))
    unknown = sorted({tag for tag, _ in models} - set(configs))
    if unknown:
        parser.error(f'Unknown config tags: {unknown}')
    for tag, model_path in models:
        expected = configs[tag][1]['target_revision']
        if model_path.name != expected:
            parser.error(f'{tag}: snapshot {model_path.name} != pinned {expected}')

    if Path(args.status_file).name != args.status_file:
        parser.error('--status-file must be a basename')
    status_path = run/args.status_file
    completed = []
    failures = []

    def status(stage, **extra):
        value = {
            'stage': stage,
            'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'gpu_ids': gpu_ids,
            'completed_models': completed,
            'failed_models': failures,
            **extra,
        }
        atomic_json(status_path, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    if args.after_pid is not None:
        status('waiting_for_predecessor', predecessor_pid=args.after_pid)
        while process_alive(args.after_pid):
            time.sleep(20)

    summary_path = run/'qa/summary.json'
    status('waiting_for_qa', wait_pid=args.wait_pid)
    while not summary_path.exists():
        if args.wait_pid is not None and not process_alive(args.wait_pid):
            status('qa_failed', reason='QA process exited without summary')
            return 1
        time.sleep(20)
    summary = json.loads(summary_path.read_text())
    if summary.get('pass_count', 0) < 1:
        status('qa_failed', reason='No QA-accepted translations', qa_summary=summary)
        return 1

    for tag, (config_path, _) in configs.items():
        destination = run/'panel'/tag
        subprocess.run([
            str(python), str(package/'code/prepare.py'), '--repo', str(repo),
            '--config', str(config_path), '--items', str(run/'inputs/items.json'),
            '--translations', str(run/'qa/accepted_translations.json'),
            '--out', str(destination),
        ], check=True)
    status('panel_prepared', qa_summary=summary, panel_models=len(configs),
           jobs_per_model=summary['pass_count'] * 5)

    log_dir = run/'logs'
    log_dir.mkdir(exist_ok=True)
    for tag, model_path in models:
        while True:
            available = free_mib(gpu_ids)
            if min(available.values()) >= args.min_free_mib:
                break
            status('waiting_for_gpu', model=tag, free_mib=available,
                   required_free_mib=args.min_free_mib)
            time.sleep(30)
        destination = run/'panel'/tag
        status('collecting', model=tag, model_path=str(model_path),
               existing_rows=count_rows(destination/'responses.jsonl'),
               total_jobs=summary['pass_count'] * 5)
        command = [
            str(python), '-u', str(package/'code/collect.py'), '--run', str(destination),
            '--gpu', args.gpu, '--model-path', str(model_path),
            '--memory-utilization', str(args.memory_utilization),
            '--batch-size', str(args.batch_size),
            '--max-num-batched-tokens', str(args.max_num_batched_tokens),
            '--tensor-parallel-size', str(args.tensor_parallel_size),
            '--cpu-offload-gb', str(args.cpu_offload_gb),
        ]
        if tag in args.trust_remote_code_tag:
            command.append('--trust-remote-code')
        environment = dict(os.environ, VLLM_USE_V1='1', PYTHONUNBUFFERED='1')
        log_path = log_dir/f'{tag}.collect.log'
        with log_path.open('a') as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=environment)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(f'[{tag}] {line}', end='', flush=True)
            returncode = child.wait()
        rows = count_rows(destination/'responses.jsonl')
        if returncode == 0:
            completed.append(tag)
            status('model_complete', model=tag, rows=rows,
                   total_jobs=summary['pass_count'] * 5)
        else:
            failures.append({'model': tag, 'returncode': returncode, 'rows': rows,
                             'log': str(log_path)})
            status('model_failed', model=tag, returncode=returncode, rows=rows,
                   log=str(log_path))
            if not args.continue_on_error:
                return returncode
    status('queue_complete', qa_summary=summary, requested_models=len(models))
    return 0 if not failures else 2


if __name__ == '__main__':
    sys.exit(main())
