#!/usr/bin/env python3
"""Run reconstruction and refusal judges after all target collection queues."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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


def complete_collection(root):
    return rows(root/'jobs.jsonl') > 0 and rows(root/'jobs.jsonl') == rows(root/'responses.jsonl')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--reconstruction-model', type=Path, required=True)
    parser.add_argument('--wildguard-model', type=Path, required=True)
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--reconstruction-gpu', default='2')
    parser.add_argument('--wildguard-gpu', default='3')
    args = parser.parse_args()

    run = args.run.resolve()
    repo = args.repo.resolve()
    python = Path(os.path.abspath(args.python))
    reconstruction_model = args.reconstruction_model.resolve()
    wildguard_model = args.wildguard_model.resolve()
    if (not run.is_dir() or not repo.is_dir() or not python.is_file() or
            not reconstruction_model.is_dir() or not wildguard_model.is_dir() or
            not args.reconstruction_gpu.isdigit() or not args.wildguard_gpu.isdigit() or
            args.reconstruction_gpu == args.wildguard_gpu):
        parser.error('Invalid run, executable, judge model, or GPU assignment')

    aligned = Path(__file__).resolve().parents[2]
    judge = aligned/'judge.py'
    summarize = aligned/'summarize.py'
    expected = sorted(path.stem for path in (Path(__file__).resolve().parents[1]/'configs').glob('*.json'))
    status_path = run/'pipeline_status_judges.json'
    log_dir = run/'logs/judges'
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    failures = []

    def status(stage, **extra):
        value = {
            'stage': stage,
            'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'completed_models': completed,
            'failed_models': failures,
            **extra,
        }
        atomic_json(status_path, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    status('waiting_for_target_collection', predecessor_pid=args.after_pid)
    while alive(args.after_pid):
        time.sleep(20)

    for tag in expected:
        target = run/'panel'/tag
        if not complete_collection(target):
            failures.append({'model': tag, 'stage': 'target_incomplete',
                             'jobs': rows(target/'jobs.jsonl'),
                             'responses': rows(target/'responses.jsonl')})
            status('target_incomplete', model=tag)
            continue
        status('judging', model=tag, responses=rows(target/'responses.jsonl'),
               reconstruction_gpu=args.reconstruction_gpu,
               wildguard_gpu=args.wildguard_gpu)
        reconstruction_command = [
            str(python), '-u', str(judge), 'reconstruction', '--run', str(target),
            '--repo', str(repo), '--model-path', str(reconstruction_model),
            '--gpu', args.reconstruction_gpu, '--batch-size', '16',
        ]
        wildguard_command = [
            str(python), '-u', str(judge), 'wildguard', '--run', str(target),
            '--repo', str(repo), '--model-path', str(wildguard_model),
            '--gpu', args.wildguard_gpu, '--memory-utilization', '.75',
        ]
        with (log_dir/f'{tag}.reconstruction.log').open('a') as reconstruction_log, \
                (log_dir/f'{tag}.wildguard.log').open('a') as wildguard_log:
            reconstruction = subprocess.Popen(
                reconstruction_command, stdout=reconstruction_log, stderr=subprocess.STDOUT)
            wildguard = subprocess.Popen(
                wildguard_command, stdout=wildguard_log, stderr=subprocess.STDOUT)
            reconstruction_returncode = reconstruction.wait()
            wildguard_returncode = wildguard.wait()
        if reconstruction_returncode or wildguard_returncode:
            failures.append({
                'model': tag, 'stage': 'judging',
                'reconstruction_returncode': reconstruction_returncode,
                'wildguard_returncode': wildguard_returncode,
            })
            status('judge_failed', model=tag,
                   reconstruction_returncode=reconstruction_returncode,
                   wildguard_returncode=wildguard_returncode)
            continue
        summary_log = log_dir/f'{tag}.summarize.log'
        with summary_log.open('a') as output:
            aggregation = subprocess.run(
                [str(python), '-u', str(summarize), '--run', str(target)],
                stdout=output, stderr=subprocess.STDOUT)
        if aggregation.returncode:
            failures.append({'model': tag, 'stage': 'summarize',
                             'returncode': aggregation.returncode})
            status('summarize_failed', model=tag, returncode=aggregation.returncode)
            continue
        completed.append(tag)
        prior = json.loads((target/'willingness_prior.json').read_text())
        status('model_complete', model=tag, prior_status=prior['status'],
               common_items=prior['common_items'],
               willingness_vector=prior['willingness_vector'])
    status('judge_queue_complete', requested_models=len(expected))
    return 0 if not failures else 2


if __name__ == '__main__':
    sys.exit(main())
