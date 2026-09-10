#!/usr/bin/env python3
"""Collect fresh paired benign target observations with restartable output chunks."""
from __future__ import annotations
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import time

from polyjig_pilot import normalize, reconstruction_score


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--target-index", required=True, type=int)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cohort", type=Path)
    a = ap.parse_args()
    manifest = json.loads((a.data/"manifest.json").read_text())
    target = manifest["targets"][a.target_index]
    jobs = [json.loads(line) for line in (a.data/"prompts.jsonl").open()]
    if any(j["risk_type"] != "benign" for j in jobs):
        raise ValueError("Collector requires benign source provenance")
    a.outdir.mkdir(parents=True, exist_ok=True)
    collection_lock = (a.outdir/".collection.lock").open("a")
    fcntl.flock(collection_lock.fileno(), fcntl.LOCK_EX)
    run_config = {"target": target, "prompt_file_sha256": manifest["prompt_file_sha256"],
                  "rows": len(jobs), "temperature": 0, "max_tokens": a.max_tokens,
                  "seed": manifest["target_response_seed"], "batch_size": a.batch_size,
                  "enable_thinking": False, "dtype": "bfloat16",
                  "metric": manifest["metric"], "source_manifest": str(a.data/"manifest.json")}
    config_path = a.outdir/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("Resume configuration differs from existing run")
    write_json(config_path, run_config)
    done = {}
    for p in sorted(a.outdir.glob("chunk_*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            key = r["item_id"], r["config_id"]
            if key in done:
                raise ValueError("Duplicate completed observation")
            done[key] = r["prompt_sha256"]
    pending = []
    for job in jobs:
        key = job["item_id"], job["config_id"]
        if key in done:
            if done[key] != job["prompt_sha256"]:
                raise ValueError("Source prompt changed on resume")
        else:
            pending.append(job)
    if a.cohort:
        cohort = set(json.loads(a.cohort.read_text()))
        pending = [j for j in pending if j["item_id"] in cohort]
    if not pending:
        print(json.dumps({"status": "already_completed", "target": target["model_id"]}), flush=True)
        return
    if a.limit:
        pending = pending[:a.limit]
    began = time.monotonic()
    print(json.dumps({"stage": "loading", "target": target, "pending": len(pending)}), flush=True)
    from vllm import LLM, SamplingParams
    native_config = json.loads((Path(target["path"])/"config.json").read_text())
    context_length = min(8192, native_config.get("max_position_embeddings", 8192))
    llm = LLM(model=target["path"], dtype="bfloat16", trust_remote_code=target["family"] == "internlm",
              tokenizer_mode="slow" if target["family"] == "internlm" else "auto",
              enforce_eager=True, max_model_len=context_length, gpu_memory_utilization=.9,
              max_num_seqs=96, max_num_batched_tokens=8192, seed=manifest["target_response_seed"])
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=a.max_tokens,
                              seed=manifest["target_response_seed"])
    loading_seconds = time.monotonic()-began
    import torch, vllm, transformers, sentencepiece
    write_json(a.outdir/f"runtime_{int(time.time())}.json", {
        "torch":torch.__version__, "vllm":vllm.__version__, "transformers":transformers.__version__,
        "sentencepiece":sentencepiece.__version__, "max_model_len":context_length,
        "tokenizer_class":type(tokenizer).__name__, "gpu":torch.cuda.get_device_name(),
        "collector_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    measured_tokens = 0
    for start in range(0, len(pending), a.batch_size):
        batch = pending[start:start+a.batch_size]
        rendered = [tokenizer.apply_chat_template([{"role": "user", "content": j["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False) for j in batch]
        batch_started = time.monotonic()
        outputs = llm.generate(rendered, sampling, use_tqdm=False)
        seconds = time.monotonic()-batch_started
        chunk_number = len(list(a.outdir.glob("chunk_*.jsonl")))
        output_path = a.outdir/f"chunk_{chunk_number:05d}.jsonl"
        temp = output_path.with_suffix(".jsonl.tmp")
        with temp.open("w") as handle:
            for job, output, prompt in zip(batch, outputs, rendered):
                generated = output.outputs[0]
                raw = generated.text
                prediction = raw.strip()
                score = reconstruction_score(prediction, job["original"])
                row = {"item_id": job["item_id"], "config_id": job["config_id"],
                       "item_split": job["item_split"], "source_sha256": job["source_sha256"],
                       "prompt_sha256": job["prompt_sha256"],
                       "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "target_model": target["model_id"], "revision": target["revision"],
                       "raw_output": raw, "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                       "reconstruction_proxy": score, "normalized_exact": int(normalize(prediction) == normalize(job["original"])),
                       "input_tokens": len(output.prompt_token_ids), "output_tokens": len(generated.token_ids),
                       "finish_reason": generated.finish_reason, "seed": manifest["target_response_seed"],
                       "batch_seconds": seconds, "risk_type": "benign"}
                measured_tokens += row["output_tokens"]
                handle.write(json.dumps(row, ensure_ascii=False)+"\n")
        temp.replace(output_path)
        completed = len(done)+start+len(batch)
        progress = {"status": "completed" if completed == len(jobs) else "collecting",
                    "target": target["model_id"], "completed": completed, "total": len(jobs),
                    "this_process_seconds": time.monotonic()-began, "loading_seconds": loading_seconds,
                    "this_process_output_tokens": measured_tokens, "last_batch_seconds": seconds,
                    "last_batch_rows": len(batch)}
        write_json(a.outdir/"progress.json", progress)
        print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
