#!/usr/bin/env python3
"""Run, audit, and privately publish the 160-setting MJ/LG L40S queue.

This supervisor deliberately excludes FalseReject and the four method baselines.
Each completed (model, dataset) unit is checked at row level, copied to an
immutable staging directory, uploaded to the private HF bucket, and followed by
a receipt.  Active JSONL files are never synced.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RUN_ID = "full_grid_20260917_v1"
OWNER = "l40s_small"
RUN_ROOT = REPO / RUN_ID / OWNER
UNIT_ROOT = RUN_ROOT / "units"
STAGING_ROOT = RUN_ROOT / "staging"
RECEIPT_ROOT = RUN_ROOT / "receipts"
LOG_ROOT = RUN_ROOT / "logs"
STATUS_PATH = RUN_ROOT / "status.jsonl"
LOCK_PATH = RUN_ROOT / "queue.lock"

PYTHON = Path("/home/jklee/safe_mental-dpo/ICLR-POLY/.venv/bin/python")
HF_BIN = Path("/home/jklee/safe_mental-dpo/ICLR-POLY/.hf-cli-venv/bin/hf")
HF_HOME = Path("/home/jklee/safe_mental-dpo/ICLR-POLY/bigmodel_l40s/hf_home")
HF_TOKEN_PATH = Path("/home/jklee/.cache/huggingface/token")
BUCKET_ROOT = f"hf://buckets/jin-kwon/poly/PolyJigsaw/{RUN_ID}/{OWNER}"

RECON_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
GUARD_REV = "4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb"
HANDOFF_COMMIT = "06f6f5ac325e3fefb37eec2c152d4739cd0e256b"
PROMPT_SEED = 20260828
PHYSICAL_GPUS = {"target": 1, "reconstruction_judge": 2, "guard_judge": 3}
VISIBLE_GPUS = "1,2,3"
EXECUTION_THREE_GPU = "three_process_gpu_pipeline"
EXECUTION_TWO_GPU = "two_gpu_staged_pipeline"
EXECUTION_GPUS = {
    EXECUTION_THREE_GPU: (1, 2, 3),
    EXECUTION_TWO_GPU: (1, 2),
}
EXTERNAL_ASSIGNMENTS = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "tag": "qwen25_3b",
        "owner": "other_gpu_server",
        "reason": "user confirmed active collection on the other GPU server on 2026-09-17",
    },
}

SETTINGS = tuple(
    f"g{fragments}_{arrangement}_n{languages}__{frame}"
    for fragments in (3, 5, 8, 12)
    for arrangement in ("ordered", "shuffled")
    for languages in (2, 4, 6, 8)
    for frame in ("plain", "persona", "fiction", "pap", "persona+fiction")
)


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    model: str
    tag: str
    revision: str
    util: float
    trust_remote_code: bool = False


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    key: str
    driver: str
    collection: str
    tlang: str
    items: int
    input_path: str
    input_sha256: str
    order_path: str
    order_sha256: str


# Start with the already-cached target; all following checkpoints are downloaded
# one at a time and removed only after both audited snapshots are receipted.
MODELS = (
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen25_7b", RECON_REV, 0.50),
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct", "llama32_3b_it",
              "0cb88a4f764b7a12671c53f0838cd831a0843b95", 0.30),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama31_8b_it",
              "0e9e39f249a16976918f6564b8830bc894c89659", 0.50),
    ModelSpec("google/gemma-2-2b-it", "gemma2_2b_it",
              "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8", 0.30),
    ModelSpec("google/gemma-2-9b-it", "gemma2_9b_it",
              "11c9b309abf73637e4b6f9a3fa1e92e615547819", 0.55),
    ModelSpec("microsoft/Phi-3.5-mini-instruct", "phi35_mini",
              "2fe192450127e6a83f7441aef6e3ca586c338b77", 0.35, True),
    ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", "mistral7b",
              "c170c708c41dac9275d15a8fff4eca08d52bab71", 0.50),
    ModelSpec("tiiuae/Falcon3-3B-Instruct", "falcon3_3b",
              "411bb94318f94f7a5735b77109f456b1e74b42a1", 0.30),
    ModelSpec("tiiuae/Falcon3-7B-Instruct", "falcon3_7b",
              "1e57a0ecd176c7c139f289c60a74e57f887c3dfb", 0.50),
    ModelSpec("tiiuae/Falcon3-10B-Instruct", "falcon3_10b",
              "8799bc6aec0152757221dc6b272d824642db6202", 0.60),
    ModelSpec("THUDM/glm-4-9b-chat-hf", "glm4_9b",
              "8599336fc6c125203efb2360bfaf4c80eef1d1bf", 0.55, True),
    # Reinstated 2026-09-19: excluded from this queue on 2026-09-17 pending confirmed
    # external collection; no output ever appeared under any known owner prefix in the
    # bucket, so it is collected here under a distinct owner (never l40s_small) to avoid
    # colliding with that external run if it resumes.
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", "qwen25_3b",
              "aa8e72537993ba99e69dfaafa59ed015b17504d1", 0.25),
)

DATASETS = (
    DatasetSpec(
        "mj", "collect_mj.py", "MultiJail", "Bengali", 315,
        "private_artifacts/multijail_v1/harm_grid.jsonl",
        "85c0cfeecfe59d3abee90640c076e0b66808e7f3cd54d524a76fae4b18faab75",
        "private_artifacts/multijail_v1/resource_order.json",
        "5e637378f209db2e656f28817416a42b4fd2624206c0cf28ddb83c7199e9e06d",
    ),
    DatasetSpec(
        "lg", "collect_lg.py", "Lingua-SafetyBench", "Norwegian", 250,
        "private_artifacts/panel_v2/harm_grid.jsonl",
        "14a69350e63d4d3c47f497de5087cf4ec2a3685085af37443a810a70c772ae62",
        "results/lang_rank_20260905/resource_order.json",
        "98fd729726e736d547b4a431fc827765d9516e9ddf70536adc21098589045446",
    ),
)

REQUIRED_ROW_FIELDS = {
    "item", "item_id", "original", "prompt_sha256", "protocol_hash",
    "reconstruction", "answer", "raw_output", "answer_section_present",
    "target_finish_reason", "target_stop_reason", "target_prompt_tokens",
    "target_completion_tokens", "target_truncated",
    "reconstruction_judge_value", "reconstruction_parse_valid",
    "reconstruction_confidence", "reconstruction_reason_code",
    "reconstruction_judge_output", "guard_judge_value", "guard_parse_valid",
    "guard_label", "guard_categories", "guard_refusal", "guard_judge_output",
    "R", "U", "J",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temp, mode)
    os.replace(temp, path)


def status(event: str, **fields: Any) -> None:
    row = {"time": utc_now(), "event": event, **fields}
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(STATUS_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def cache_dir(model: str) -> Path:
    return HF_HOME / "hub" / f"models--{model.replace('/', '--')}"


def cached_revision(model: ModelSpec) -> bool:
    snapshot = cache_dir(model.model) / "snapshots" / model.revision
    return snapshot.is_dir() and any(snapshot.iterdir())


def command_environment(online: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(HF_HOME),
        "HF_TOKEN_PATH": str(HF_TOKEN_PATH),
        "TOKENIZERS_PARALLELISM": "false",
    })
    if online:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    return env


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] COMMAND " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=REPO, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, text=True)
    return result.returncode


def retry_command(name: str, command: list[str], log_path: Path,
                  env: dict[str, str], attempts: int = 5) -> bool:
    for attempt in range(1, attempts + 1):
        status("command_start", name=name, attempt=attempt, log=str(log_path))
        return_code = run_logged(command, log_path, env)
        if return_code == 0:
            status("command_complete", name=name, attempt=attempt)
            return True
        status("command_failed", name=name, attempt=attempt, return_code=return_code)
        if attempt < attempts:
            time.sleep(120)
    return False


def ensure_download(model: ModelSpec) -> bool:
    if cached_revision(model):
        status("download_skip", model=model.model, revision=model.revision, reason="exact_revision_cached")
        return True
    usage = shutil.disk_usage(REPO)
    status("download_start", model=model.model, revision=model.revision,
           free_bytes=usage.free)
    command = [str(HF_BIN), "download", model.model, "--revision", model.revision,
               "--exclude", "original/**", "-q"]
    ok = retry_command(f"download:{model.tag}", command,
                       LOG_ROOT / f"download_{model.tag}.log",
                       command_environment(online=True))
    if ok and not cached_revision(model):
        status("download_invalid", model=model.model, revision=model.revision)
        return False
    return ok


def package_versions() -> dict[str, str]:
    names = ("vllm", "torch", "transformers", "tokenizers", "huggingface-hub")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "MISSING"
    try:
        import torch
        result["torch_runtime"] = torch.__version__
        result["torch_cuda"] = str(torch.version.cuda)
    except ImportError:
        result["torch_runtime"] = "MISSING"
        result["torch_cuda"] = "MISSING"
    return result


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   text=True).strip()


def gpu_inventory() -> list[dict[str, Any]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    inventory = []
    for line in output.splitlines():
        index, name, driver, memory = (part.strip() for part in line.split(",", 3))
        inventory.append({"index": int(index), "name": name, "driver": driver,
                          "memory_mib": int(memory)})
    return inventory


def gpu_free_mib() -> dict[int, int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
    ], text=True)
    free = {}
    for line in output.splitlines():
        index, memory = (part.strip() for part in line.split(",", 1))
        free[int(index)] = int(memory)
    return free


def wait_for_execution_mode(minimum_free_mib: int = 40000) -> str:
    """Prefer the fast three-GPU pipeline, otherwise use idle GPUs 1+2 staged."""
    while True:
        free = gpu_free_mib()
        if all(free.get(index, 0) >= minimum_free_mib for index in (1, 2, 3)):
            status("execution_slot_available", execution=EXECUTION_THREE_GPU,
                   free_mib={index: free[index] for index in (1, 2, 3)})
            return EXECUTION_THREE_GPU
        if all(free.get(index, 0) >= minimum_free_mib for index in (1, 2)):
            status("execution_slot_available", execution=EXECUTION_TWO_GPU,
                   free_mib={index: free[index] for index in (1, 2)},
                   unavailable_gpu3_mib=free.get(3, 0))
            return EXECUTION_TWO_GPU
        blocked = {index: free.get(index, 0) for index in (1, 2)
                   if free.get(index, 0) < minimum_free_mib}
        status("execution_slot_wait", minimum_free_mib=minimum_free_mib,
               required_any_of=[[1, 2, 3], [1, 2]], blocked=blocked,
               free_mib={index: free.get(index, 0) for index in (1, 2, 3)})
        time.sleep(60)


def wait_for_mode_gpus(execution: str, minimum_free_mib: int = 40000) -> None:
    wanted = EXECUTION_GPUS[execution]
    while True:
        free = gpu_free_mib()
        blocked = {index: free.get(index, 0) for index in wanted
                   if free.get(index, 0) < minimum_free_mib}
        if not blocked:
            status("assigned_gpus_available", execution=execution,
                   free_mib={index: free[index] for index in wanted})
            return
        status("assigned_gpus_wait", execution=execution,
               minimum_free_mib=minimum_free_mib, blocked=blocked)
        time.sleep(60)


def tokenizer_template_hash(model: ModelSpec) -> str:
    from transformers import AutoTokenizer
    snapshot = cache_dir(model.model) / "snapshots" / model.revision
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        trust_remote_code=model.trust_remote_code,
        local_files_only=True,
    )
    template = tokenizer.chat_template or ""
    return sha256_bytes(template.encode("utf-8"))


def verify_static_inputs() -> None:
    if len(SETTINGS) != 160 or len(set(SETTINGS)) != 160:
        raise RuntimeError("setting construction is not exactly 160 unique arms")
    for dataset in DATASETS:
        for relative, expected in ((dataset.input_path, dataset.input_sha256),
                                   (dataset.order_path, dataset.order_sha256)):
            observed = sha256_file(REPO / relative)
            if observed != expected:
                raise RuntimeError(f"input hash mismatch: {relative}: {observed} != {expected}")
        with (REPO / dataset.input_path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        ids = [str(row["item_id"]) for row in rows]
        if len(ids) != dataset.items or len(ids) != len(set(ids)):
            raise RuntimeError(f"invalid {dataset.key} rows/ids: {len(ids)}/{len(set(ids))}")


def make_protocol(model: ModelSpec, dataset: DatasetSpec,
                  execution: str = EXECUTION_THREE_GPU) -> dict[str, Any]:
    if execution not in EXECUTION_GPUS:
        raise ValueError(f"unsupported execution mode: {execution}")
    source_files = [
        "scripts/online_live.py",
        "scripts/closed_compare.py",
        "experiments_suite/common/engine.py",
        "experiments_suite/exp02_panel_collect/_common.py",
        f"experiments_suite/exp02_panel_collect/{dataset.driver}",
        "bigmodel_l40s/scripts/pipelined_full_grid_collect.py",
    ]
    if execution == EXECUTION_TWO_GPU:
        source_files.append("bigmodel_l40s/scripts/staged_two_gpu_full_grid_collect.py")
    if execution == EXECUTION_THREE_GPU:
        physical_gpus = PHYSICAL_GPUS
        visible_gpus = VISIBLE_GPUS
        logical_mapping = {
            "target_worker_cuda:0": 1,
            "reconstruction_worker_cuda:0": 2,
            "guard_worker_cuda:0": 3,
        }
        execution_label = EXECUTION_THREE_GPU
        max_inflight = 8
    else:
        physical_gpus = {"target": 1, "reconstruction_judge": 2, "guard_judge": 2}
        visible_gpus = "1,2"
        logical_mapping = {
            "target_worker_cuda:0": 1,
            "reconstruction_worker_cuda:0": 2,
            "guard_worker_cuda:0_after_reconstruction_exit": 2,
        }
        execution_label = EXECUTION_TWO_GPU
        max_inflight = 160
    core = {
        "schema": "poly_full_grid_unit_v1",
        "run_id": RUN_ID,
        "owner": OWNER,
        "handoff_commit": HANDOFF_COMMIT,
        "code": {
            "git_head": git_head(),
            "files_sha256": {name: sha256_file(REPO / name) for name in source_files},
        },
        "scope": {
            "dataset": dataset.key,
            "collection": dataset.collection,
            "items": dataset.items,
            "settings": list(SETTINGS),
            "setting_count": len(SETTINGS),
            "response_count": dataset.items * len(SETTINGS),
            "excluded": ["FalseReject", "m_aim", "m_deepinception", "m_pap", "m_translated"],
        },
        "input": {
            "path": dataset.input_path,
            "sha256": dataset.input_sha256,
            "order_path": dataset.order_path,
            "order_sha256": dataset.order_sha256,
            "translation_language": dataset.tlang,
        },
        "target": {
            "model": model.model,
            "revision": model.revision,
            "tokenizer_revision": model.revision,
            "chat_template_sha256": tokenizer_template_hash(model),
            "trust_remote_code": model.trust_remote_code,
            "dtype": "bfloat16",
            "tensor_parallel": 1,
            "gpu_memory_utilization": max(model.util, 0.70),
            "max_model_len": 8192,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 16384,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_new_tokens": 1024,
            "generation_seed": None,
            "prompt_seed": PROMPT_SEED,
            "enable_thinking": None,
            "strong_reconstruction_prompt": False,
        },
        "judges": {
            "reconstruction": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "revision": RECON_REV,
                "max_new_tokens": 80,
            },
            "guard": {
                "model": "Qwen/Qwen3Guard-Gen-8B",
                "revision": GUARD_REV,
                "max_new_tokens": 128,
                "strict_unsafe_labels": ["Unsafe", "Controversial"],
            },
            "batch_size": 32,
            "execution": execution_label,
            "max_inflight_arms": max_inflight,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "packages": package_versions(),
            "physical_gpus": physical_gpus,
            "cuda_visible_devices": visible_gpus,
            "logical_mapping": logical_mapping,
            "gpu_inventory": gpu_inventory(),
            "vllm_enforce_eager": False,
        },
    }
    return {**core, "protocol_hash": canonical_hash(core)}


def expected_ids(dataset: DatasetSpec) -> set[str]:
    with (REPO / dataset.input_path).open(encoding="utf-8") as handle:
        return {str(json.loads(line)["item_id"]) for line in handle if line.strip()}


def output_tag(model: ModelSpec, dataset: DatasetSpec) -> str:
    return f"{model.tag}_{dataset.key}"


def validate_arm(root: Path, tag: str, arm: str, dataset: DatasetSpec,
                 protocol: dict[str, Any], wanted_ids: set[str]) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    stats = {"rows": 0, "reconstruction_parse_failures": 0,
             "guard_parse_failures": 0, "target_truncations": 0}
    raw_path = root / "attack" / "_raw" / f"{tag}__{arm}.jsonl"
    aggregate_path = root / "attack" / f"{tag}__{arm}.json"
    if not raw_path.is_file() or not aggregate_path.is_file():
        return ["missing raw or aggregate"], stats
    observed_ids: list[str] = []
    try:
        with raw_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                stats["rows"] += 1
                missing = REQUIRED_ROW_FIELDS.difference(row)
                if missing:
                    errors.append(f"row {line_number}: missing {sorted(missing)}")
                    continue
                observed_ids.append(str(row["item_id"]))
                if row["protocol_hash"] != protocol["protocol_hash"]:
                    errors.append(f"row {line_number}: protocol hash mismatch")
                if row["target_finish_reason"] not in {"stop", "length"}:
                    errors.append(f"row {line_number}: invalid finish reason {row['target_finish_reason']!r}")
                if not isinstance(row["target_prompt_tokens"], int) or row["target_prompt_tokens"] <= 0:
                    errors.append(f"row {line_number}: invalid prompt token count")
                if not isinstance(row["target_completion_tokens"], int) or row["target_completion_tokens"] < 0:
                    errors.append(f"row {line_number}: invalid completion token count")
                if not row["reconstruction_parse_valid"]:
                    stats["reconstruction_parse_failures"] += 1
                if not row["guard_parse_valid"]:
                    stats["guard_parse_failures"] += 1
                if row["target_truncated"]:
                    stats["target_truncations"] += 1
                expected_joint = 1.0 if (bool(row["R"]) and bool(row["U"])) else 0.0
                if float(row["J"]) != expected_joint:
                    errors.append(f"row {line_number}: inconsistent J/R/U")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"raw parse error: {exc}")
    if stats["rows"] != dataset.items:
        errors.append(f"row count {stats['rows']} != {dataset.items}")
    if len(observed_ids) != len(set(observed_ids)):
        errors.append("duplicate item_id")
    if set(observed_ids) != wanted_ids:
        errors.append("item_id set mismatch")
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate.get("n") != dataset.items:
            errors.append(f"aggregate n {aggregate.get('n')} != {dataset.items}")
        if aggregate.get("method") != arm:
            errors.append("aggregate method mismatch")
        if aggregate.get("protocol_hash") != protocol["protocol_hash"]:
            errors.append("aggregate protocol hash mismatch")
        if aggregate.get("target_revision") != protocol["target"]["revision"]:
            errors.append("aggregate target revision mismatch")
        if aggregate.get("reconstruction_judge_revision") != RECON_REV:
            errors.append("aggregate reconstruction revision mismatch")
        if aggregate.get("guard_judge_revision") != GUARD_REV:
            errors.append("aggregate guard revision mismatch")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"aggregate parse error: {exc}")
    return errors, stats


def validate_unit(root: Path, model: ModelSpec, dataset: DatasetSpec,
                  protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    wanted_ids = expected_ids(dataset)
    invalid: dict[str, list[str]] = {}
    totals = {"rows": 0, "reconstruction_parse_failures": 0,
              "guard_parse_failures": 0, "target_truncations": 0}
    tag = output_tag(model, dataset)
    for arm in SETTINGS:
        errors, stats = validate_arm(root, tag, arm, dataset, protocol, wanted_ids)
        if errors:
            invalid[arm] = errors[:25]
        for key in totals:
            totals[key] += stats[key]
    aggregate_files = set((root / "attack").glob(f"{tag}__*.json")) if (root / "attack").exists() else set()
    raw_files = set((root / "attack" / "_raw").glob(f"{tag}__*.jsonl")) \
        if (root / "attack" / "_raw").exists() else set()
    expected_aggregates = {root / "attack" / f"{tag}__{arm}.json" for arm in SETTINGS}
    expected_raw = {root / "attack" / "_raw" / f"{tag}__{arm}.jsonl" for arm in SETTINGS}
    unexpected = sorted(str(path.relative_to(root)) for path in
                        (aggregate_files - expected_aggregates) | (raw_files - expected_raw))
    if unexpected:
        invalid["__unexpected_files__"] = unexpected
    audit = {
        "schema": "poly_full_grid_audit_v1",
        "model": model.model,
        "tag": tag,
        "dataset": dataset.key,
        "protocol_hash": protocol["protocol_hash"],
        "expected_settings": 160,
        "valid_settings": len(SETTINGS) - len([key for key in invalid if not key.startswith("__")]),
        "expected_rows": dataset.items * len(SETTINGS),
        **totals,
        "invalid": invalid,
        "complete": not invalid and totals["rows"] == dataset.items * len(SETTINGS),
    }
    atomic_json(root / "audit.json", audit)
    return audit, invalid


def quarantine_invalid(root: Path, model: ModelSpec, dataset: DatasetSpec,
                       invalid: dict[str, list[str]]) -> None:
    arms = [arm for arm in invalid if arm in SETTINGS]
    if not arms:
        return
    destination: Path | None = None
    moved = 0
    tag = output_tag(model, dataset)
    for arm in arms:
        for source in (root / "attack" / f"{tag}__{arm}.json",
                       root / "attack" / "_raw" / f"{tag}__{arm}.jsonl"):
            if source.exists():
                if destination is None:
                    destination = (root / "quarantine" /
                                   dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
                    destination.mkdir(parents=True, exist_ok=True)
                    os.chmod(destination, 0o700)
                relative = source.relative_to(root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
                moved += 1
    if destination is not None:
        status("invalid_outputs_quarantined", model=model.model, dataset=dataset.key,
               files=moved, path=str(destination))


def collector_command(model: ModelSpec, dataset: DatasetSpec, root: Path,
                      execution: str) -> list[str]:
    if execution == EXECUTION_THREE_GPU:
        runner = REPO / "bigmodel_l40s" / "scripts" / "pipelined_full_grid_collect.py"
    elif execution == EXECUTION_TWO_GPU:
        runner = REPO / "bigmodel_l40s" / "scripts" / "staged_two_gpu_full_grid_collect.py"
    else:
        raise ValueError(f"unsupported execution mode: {execution}")
    command = [
        str(PYTHON),
        str(runner),
        "--model", model.model,
        "--tag", output_tag(model, dataset),
        "--collection", dataset.collection,
        "--n-items", str(dataset.items),
        "--root", str(root),
        "--order", str(REPO / dataset.order_path),
        "--harm", str(REPO / dataset.input_path),
        "--tlang", dataset.tlang,
        "--util", str(max(model.util, 0.70)),
        "--max-tokens", "1024",
        "--max-model-len", "8192",
    ]
    if execution == EXECUTION_THREE_GPU:
        command.extend([
            "--max-inflight", "8",
            "--target-gpu", str(PHYSICAL_GPUS["target"]),
            "--recon-gpu", str(PHYSICAL_GPUS["reconstruction_judge"]),
            "--guard-gpu", str(PHYSICAL_GPUS["guard_judge"]),
        ])
    else:
        command.extend([
            "--max-inflight", "160",
            "--target-gpu", "1",
            "--judge-gpu", "2",
        ])
    if model.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def collection_environment(model: ModelSpec, protocol: dict[str, Any],
                           execution: str) -> dict[str, str]:
    env = command_environment()
    env.update({
        "CUDA_VISIBLE_DEVICES": "1,2,3" if execution == EXECUTION_THREE_GPU else "1,2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENFORCE_EAGER": "0",
        "POLY_MAX_NUM_SEQS": "64",
        "POLY_MAX_NUM_BATCHED_TOKENS": "16384",
        "POLY_JUDGE_BATCH_SIZE": "32",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "POLY_TARGET_REV": model.revision,
        "POLY_RECON_JUDGE_REV": RECON_REV,
        "POLY_GUARD_JUDGE_REV": GUARD_REV,
        "POLY_PROTOCOL_HASH": protocol["protocol_hash"],
        "POLY_REP_PENALTY": "1.0",
        "POLY_STRONG_RECON": "0",
    })
    return env


def collect_unit(model: ModelSpec, dataset: DatasetSpec,
                 protocol: dict[str, Any], execution: str) -> bool:
    root = UNIT_ROOT / model.tag / dataset.key
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    atomic_json(root / "protocol.json", protocol)
    audit, invalid = validate_unit(root, model, dataset, protocol)
    if audit["complete"]:
        status("collection_skip", model=model.model, dataset=dataset.key,
               reason="audited_complete", rows=audit["rows"])
        return True
    for attempt in range(1, 6):
        if invalid:
            quarantine_invalid(root, model, dataset, invalid)
        status("collection_start", model=model.model, dataset=dataset.key,
               attempt=attempt, expected_responses=dataset.items * len(SETTINGS),
               protocol_hash=protocol["protocol_hash"])
        wait_for_mode_gpus(execution)
        return_code = run_logged(
            collector_command(model, dataset, root, execution),
            LOG_ROOT / f"{model.tag}_{dataset.key}.log",
            collection_environment(model, protocol, execution),
        )
        audit, invalid = validate_unit(root, model, dataset, protocol)
        if return_code == 0 and audit["complete"]:
            status("collection_complete", model=model.model, dataset=dataset.key,
                   rows=audit["rows"], reconstruction_parse_failures=audit["reconstruction_parse_failures"],
                   guard_parse_failures=audit["guard_parse_failures"],
                   target_truncations=audit["target_truncations"])
            return True
        status("collection_retry", model=model.model, dataset=dataset.key,
               attempt=attempt, return_code=return_code, invalid_settings=len(invalid))
        if attempt < 5:
            time.sleep(120)
    status("collection_abandoned", model=model.model, dataset=dataset.key,
           invalid=invalid)
    return False


def snapshot_files(root: Path) -> list[Path]:
    included = []
    for relative in (Path("attack"), Path("MANIFEST.json"), Path("protocol.json"),
                     Path("audit.json")):
        path = root / relative
        if path.is_dir():
            included.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            included.append(path)
    return sorted(set(included))


def create_snapshot(root: Path, model: ModelSpec, dataset: DatasetSpec,
                    protocol: dict[str, Any]) -> tuple[str, Path, str]:
    snapshot_id = f"{model.tag}_{dataset.key}_{protocol['protocol_hash'][:16]}"
    manifest = {
        "schema": "poly_full_grid_snapshot_v1",
        "snapshot_id": snapshot_id,
        "run_id": RUN_ID,
        "owner": OWNER,
        "model": model.model,
        "model_revision": model.revision,
        "dataset": dataset.key,
        "settings": len(SETTINGS),
        "items_per_setting": dataset.items,
        "responses": len(SETTINGS) * dataset.items,
        "protocol_hash": protocol["protocol_hash"],
        "judging_complete": True,
        "files": [],
    }
    for path in snapshot_files(root):
        manifest["files"].append({
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    atomic_json(root / "snapshot_manifest.json", manifest)
    manifest_hash = sha256_file(root / "snapshot_manifest.json")
    stage = STAGING_ROOT / snapshot_id
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    os.chmod(stage, 0o700)
    for source in snapshot_files(root) + [root / "snapshot_manifest.json"]:
        relative = source.relative_to(root)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return snapshot_id, stage, manifest_hash


def upload_snapshot(root: Path, model: ModelSpec, dataset: DatasetSpec,
                    protocol: dict[str, Any]) -> bool:
    snapshot_id, stage, manifest_hash = create_snapshot(root, model, dataset, protocol)
    local_receipt = RECEIPT_ROOT / f"{snapshot_id}.json"
    if local_receipt.is_file():
        receipt = json.loads(local_receipt.read_text(encoding="utf-8"))
        if (receipt.get("protocol_hash") == protocol["protocol_hash"] and
                receipt.get("snapshot_manifest_sha256") == manifest_hash):
            status("upload_skip", snapshot_id=snapshot_id, reason="local_receipt_matches")
            shutil.rmtree(stage)
            return True
        status("immutable_snapshot_conflict", snapshot_id=snapshot_id,
               receipted_manifest=receipt.get("snapshot_manifest_sha256"),
               current_manifest=manifest_hash)
        shutil.rmtree(stage)
        return False
    destination = f"{BUCKET_ROOT}/snapshots/{snapshot_id}"
    command = [str(HF_BIN), "buckets", "sync", str(stage), destination]
    if not retry_command(f"upload:{snapshot_id}", command,
                         LOG_ROOT / f"upload_{snapshot_id}.log",
                         command_environment(online=True)):
        return False
    receipt = {
        "schema": "poly_full_grid_receipt_v1",
        "snapshot_id": snapshot_id,
        "snapshot_manifest_sha256": manifest_hash,
        "protocol_hash": protocol["protocol_hash"],
        "uploaded_at": utc_now(),
        "destination": destination,
    }
    receipt_stage = RECEIPT_ROOT / snapshot_id
    receipt_stage.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_stage / "receipt.json", receipt)
    receipt_destination = f"{BUCKET_ROOT}/receipts/{snapshot_id}"
    receipt_ok = retry_command(
        f"receipt:{snapshot_id}",
        [str(HF_BIN), "buckets", "sync", str(receipt_stage), receipt_destination],
        LOG_ROOT / f"receipt_{snapshot_id}.log",
        command_environment(online=True),
    )
    if receipt_ok:
        atomic_json(local_receipt, receipt)
        status("snapshot_receipted", snapshot_id=snapshot_id,
               manifest_sha256=manifest_hash, responses=dataset.items * len(SETTINGS))
        shutil.rmtree(stage)
        shutil.rmtree(receipt_stage)
    return receipt_ok


def cleanup_target_cache(model: ModelSpec) -> None:
    # Qwen-7B is also the reconstruction judge and must remain resident.
    if model.model == "Qwen/Qwen2.5-7B-Instruct":
        return
    path = cache_dir(model.model)
    resolved_hf = HF_HOME.resolve()
    try:
        resolved_path = path.resolve()
    except FileNotFoundError:
        return
    if not path.is_dir() or resolved_hf not in resolved_path.parents:
        return
    shutil.rmtree(path)
    status("target_cache_removed", model=model.model, path=str(path),
           reason="both_dataset_snapshots_receipted")


def preflight() -> None:
    verify_static_inputs()
    if not PYTHON.is_file() or not HF_BIN.is_file() or not HF_TOKEN_PATH.is_file():
        raise RuntimeError("required Python/HF CLI/token path is missing")
    versions = package_versions()
    expected = {
        "vllm": "0.8.5",
        "torch": "2.6.0",
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "huggingface-hub": "0.36.2",
    }
    for name, wanted in expected.items():
        observed = versions[name]
        if name == "torch":
            if not observed.startswith(wanted):
                raise RuntimeError(f"{name} {observed} != {wanted}.*")
        elif observed != wanted:
            raise RuntimeError(f"{name} {observed} != {wanted}")
    for model, revision in (("Qwen/Qwen2.5-7B-Instruct", RECON_REV),
                            ("Qwen/Qwen3Guard-Gen-8B", GUARD_REV)):
        spec = ModelSpec(model, "judge", revision, 0.0)
        if not cached_revision(spec):
            raise RuntimeError(f"judge cache missing exact revision: {model}@{revision}")
    status("preflight_complete", models=len(MODELS), datasets=len(DATASETS),
           settings=len(SETTINGS), total_responses=sum(d.items for d in DATASETS) * len(SETTINGS) * len(MODELS),
           versions=versions, false_reject=False,
           externally_assigned=EXTERNAL_ASSIGNMENTS)


def unit_receipted(model: ModelSpec, dataset: DatasetSpec) -> bool:
    prefix = f"{model.tag}_{dataset.key}_"
    if not RECEIPT_ROOT.is_dir():
        return False
    for path in RECEIPT_ROOT.glob(f"{prefix}*.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if str(receipt.get("snapshot_id", "")).startswith(prefix):
            return True
    return False


def durable_unit_protocol(root: Path, model: ModelSpec,
                          dataset: DatasetSpec) -> dict[str, Any] | None:
    """Reuse a protocol only after durable per-arm work exists.

    A supervisor can write protocol/audit metadata before entering a GPU wait.
    Those two files alone must not pin an unstarted unit to a now-unavailable
    execution topology.
    """
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        return None
    has_work = False
    attack = root / "attack"
    pipeline = root / "_pipeline"
    if attack.is_dir():
        has_work = any(attack.glob("*.json")) or any((attack / "_raw").glob("*.jsonl"))
    if not has_work and pipeline.is_dir():
        has_work = any(pipeline.rglob("*.jsonl"))
    if not has_work:
        return None
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (protocol.get("target", {}).get("model") != model.model or
            protocol.get("scope", {}).get("dataset") != dataset.key):
        raise RuntimeError(f"durable unit protocol identity mismatch: {root}")
    return protocol


def execution_from_protocol(protocol: dict[str, Any]) -> str:
    execution = protocol.get("judges", {}).get("execution")
    if execution == EXECUTION_TWO_GPU:
        return EXECUTION_TWO_GPU
    if execution == EXECUTION_THREE_GPU:
        return EXECUTION_THREE_GPU
    raise RuntimeError(f"unsupported durable execution mode: {execution!r}")


def run_queue(selected_models: set[str] | None = None,
              selected_datasets: set[str] | None = None) -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        status("queue_already_running", lock=str(LOCK_PATH))
        return 2
    preflight()
    failed = []
    for model in MODELS:
        if selected_models and model.tag not in selected_models and model.model not in selected_models:
            continue
        wanted_datasets = [dataset for dataset in DATASETS
                           if not selected_datasets or dataset.key in selected_datasets]
        if wanted_datasets and all(unit_receipted(model, dataset) for dataset in wanted_datasets):
            status("model_skip", model=model.model, reason="all_selected_units_receipted",
                   datasets=[dataset.key for dataset in wanted_datasets])
            continue
        if not ensure_download(model):
            failed.append({"model": model.model, "stage": "download"})
            continue
        unit_receipts = []
        for dataset in wanted_datasets:
            if unit_receipted(model, dataset):
                status("collection_skip", model=model.model, dataset=dataset.key,
                       reason="snapshot_already_receipted")
                unit_receipts.append(dataset.key)
                continue
            root = UNIT_ROOT / model.tag / dataset.key
            protocol = durable_unit_protocol(root, model, dataset)
            if protocol is None:
                execution = wait_for_execution_mode()
                protocol = make_protocol(model, dataset, execution)
            else:
                execution = execution_from_protocol(protocol)
                status("collection_resume_protocol", model=model.model,
                       dataset=dataset.key, execution=execution,
                       protocol_hash=protocol["protocol_hash"])
            if not collect_unit(model, dataset, protocol, execution):
                failed.append({"model": model.model, "dataset": dataset.key, "stage": "collection"})
                continue
            if not upload_snapshot(root, model, dataset, protocol):
                failed.append({"model": model.model, "dataset": dataset.key, "stage": "upload"})
                continue
            unit_receipts.append(dataset.key)
        expected_receipts = {dataset.key for dataset in wanted_datasets}
        if set(unit_receipts) == expected_receipts and expected_receipts == {"mj", "lg"}:
            cleanup_target_cache(model)
    status("queue_finished", failed=failed, success=not failed)
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "preflight", "plan"), nargs="?", default="run")
    parser.add_argument("--models", default="",
                        help="optional comma-separated model tags or exact HF IDs")
    parser.add_argument("--datasets", default="",
                        help="optional comma-separated subset: mj,lg")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps({
            "run_id": RUN_ID,
            "owner": OWNER,
            "false_reject": False,
            "externally_assigned": EXTERNAL_ASSIGNMENTS,
            "settings": len(SETTINGS),
            "models": [dataclasses.asdict(model) for model in MODELS],
            "datasets": [dataclasses.asdict(dataset) for dataset in DATASETS],
            "responses": sum(dataset.items for dataset in DATASETS) * len(SETTINGS) * len(MODELS),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight":
        preflight()
        return 0
    models = {item.strip() for item in args.models.split(",") if item.strip()} or None
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()} or None
    if datasets and not datasets.issubset({"mj", "lg"}):
        parser.error("--datasets accepts only mj,lg")
    return run_queue(models, datasets)


if __name__ == "__main__":
    raise SystemExit(main())
