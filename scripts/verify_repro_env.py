#!/usr/bin/env python3
"""Fail-fast check for the exact 2026-09-10 panel collection environment.

This prints only package versions, paths, hashes, git state, and behavior-changing environment
variables; it never prints benchmark text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_PACKAGES = {
    "accelerate": "1.15.0",
    "einops": "0.8.2",
    "huggingface-hub": "1.29.0",
    "ninja": "1.13.2",
    "numpy": "2.2.6",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "tokenizers": "0.23.1",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.16.1",
    "vllm": "0.28.0",
}
EXPECTED_MODELS = {
    "models--Qwen--Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "models--Qwen--Qwen3Guard-Gen-8B": "4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb",
}
EXPECTED_FILES = {
    "private_artifacts/multijail_v1/harm_grid.jsonl":
        "85c0cfeecfe59d3abee90640c076e0b66808e7f3cd54d524a76fae4b18faab75",
    "private_artifacts/multijail_v1/benign_probe.jsonl":
        "e6cd7458a3f45cc0aa9827ab2a4ab0836d21f33d5f40f941bde7135d9f5d8a69",
    "private_artifacts/multijail_v1/resource_order.json":
        "5e637378f209db2e656f28817416a42b4fd2624206c0cf28ddb83c7199e9e06d",
    "private_artifacts/panel_v2/harm_grid.jsonl":
        "14a69350e63d4d3c47f497de5087cf4ec2a3685085af37443a810a70c772ae62",
    "private_artifacts/panel_v2/benign_probe.jsonl":
        "28ecaf61a78a3da65218a71dd33ed02a1f246350f89097de0181a4e36787c2ce",
    "results/lang_rank_20260905/resource_order.json":
        "98fd729726e736d547b4a431fc827765d9516e9ddf70536adc21098589045446",
}
SAMPLES = {
    "MultiJail": ("private_artifacts/multijail_v1/harm_grid.jsonl", 64,
                  "186d1e180324bcd5dfc48115ccfaa8b4add8cc38b3bff5d9c6e1858bd65c6a8e"),
    "Lingua": ("private_artifacts/panel_v2/harm_grid.jsonl", 40,
               "c2f4a920bf4b9371e06f7842d4491415faa9b3a1a76c6eea11ca38420ab6f10d"),
}

# These knobs were unset in the known-working reference collection.  Setting any of them can change
# prompts, target generations, batching, or numerical execution.  Exact judge/target revisions are
# allowed explicitly because they are equivalent to the checked-in defaults.
REFERENCE_ENV = {
    "POLY_REP_PENALTY": {None, "", "1", "1.0"},
    "POLY_STRONG_RECON": {None, "", "0"},
    "POLY_MAX_NUM_SEQS": {None, ""},
    "VLLM_ENFORCE_EAGER": {None, "", "0"},
    "POLY_RECON_JUDGE_REV": {None, "", EXPECTED_MODELS["models--Qwen--Qwen2.5-7B-Instruct"]},
    "POLY_GUARD_JUDGE_REV": {None, "", EXPECTED_MODELS["models--Qwen--Qwen3Guard-Gen-8B"]},
    "POLY_TARGET_REV": {None, "", EXPECTED_MODELS["models--Qwen--Qwen2.5-7B-Instruct"]},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def report(label: str, actual: str, expected: str) -> bool:
    ok = actual == expected
    print(f"[{'OK' if ok else 'XX'}] {label}: {actual} (expected {expected})")
    return ok


def sample_fingerprint(path: Path, n: int) -> str:
    import numpy as np

    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    indices = np.random.default_rng(20260909).choice(len(rows), n, replace=False)
    item_ids = [str(rows[i].get("item_id", "")) for i in indices]
    return hashlib.sha256("\n".join(item_ids).encode()).hexdigest()


def check_reference_environment() -> bool:
    ok = True
    for name, allowed in REFERENCE_ENV.items():
        actual = os.environ.get(name)
        valid = actual in allowed
        shown = "UNSET" if actual is None or actual == "" else actual
        print(f"[{'OK' if valid else 'XX'}] env {name}: {shown}")
        ok = valid and ok
    return ok


def git_value(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument(
        "--strict-reference", action="store_true",
        help="require models/data and reject behavior-changing env overrides for the Qwen2.5-7B reference")
    parser.add_argument("--require-clean", action="store_true", help="fail if the git worktree is dirty")
    parser.add_argument(
        "--expected-commit", default="",
        help="fail unless HEAD starts with this pinned commit hash (short hashes are accepted)")
    args = parser.parse_args()

    if args.strict_reference:
        args.require_models = True
        args.require_data = True

    ok = report("python", f"{sys.version_info.major}.{sys.version_info.minor}", "3.10")
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "MISSING"
        ok = report(f"package {package}", actual, expected) and ok

    hf_home = args.hf_home or (Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else None)
    if args.require_models or hf_home:
        if hf_home is None:
            print("[XX] models: set HF_HOME or pass --hf-home")
            ok = False
        else:
            for repo_name, revision in EXPECTED_MODELS.items():
                model_root = hf_home / "hub" / repo_name
                snapshot = model_root / "snapshots" / revision
                present = snapshot.is_dir()
                print(f"[{'OK' if present else 'XX'}] model {repo_name}: {revision}")
                ok = present and ok
                # Old/ intentionally preserves the original code, which loaded the cache's main ref
                # without passing revision=.  Merely having the snapshot somewhere in the cache is
                # therefore insufficient for exact reproduction of that path.
                if args.strict_reference:
                    main_ref = model_root / "refs" / "main"
                    actual_ref = main_ref.read_text().strip() if main_ref.is_file() else "MISSING"
                    ok = report(f"model ref {repo_name} main", actual_ref, revision) and ok

    if args.require_data:
        for relative, expected in EXPECTED_FILES.items():
            path = args.repo / relative
            actual = digest(path) if path.is_file() else "MISSING"
            ok = report(f"data {relative}", actual, expected) and ok
        for name, (relative, n, expected) in SAMPLES.items():
            path = args.repo / relative
            actual = sample_fingerprint(path, n) if path.is_file() else "MISSING"
            ok = report(f"sample {name} n={n}", actual, expected) and ok

    if args.strict_reference:
        ok = check_reference_environment() and ok

    head = git_value(args.repo, "rev-parse", "HEAD")
    if head:
        print(f"[OK] git HEAD: {head}")
    elif args.expected_commit or args.require_clean:
        print("[XX] git HEAD: unavailable")
        ok = False

    if args.expected_commit and head:
        commit_ok = head.startswith(args.expected_commit)
        print(f"[{'OK' if commit_ok else 'XX'}] expected commit: {args.expected_commit}")
        ok = commit_ok and ok

    if args.require_clean:
        status = git_value(args.repo, "status", "--porcelain")
        clean = status == ""
        print(f"[{'OK' if clean else 'XX'}] git worktree: {'clean' if clean else 'dirty'}")
        ok = clean and ok

    print("REPRO_ENV_OK" if ok else "REPRO_ENV_MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
