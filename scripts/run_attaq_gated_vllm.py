#!/usr/bin/env python3
"""vLLM-backed AttaQ-multilingual gated-ASR generation.

Mirrors run_attaq_gated.py's conditions and output schema (so HF and vLLM runs
are directly comparable) but generates with vLLM, which is ~40x faster on this
box and makes the full 1402-item AttaQ set tractable on a single GPU.

Runs in .vllm_env. Raw completions are controlled content: written 0600.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polyjig_pilot import reconstruction_score  # noqa: E402
from run_polyjig_gated import section, secure_write  # noqa: E402
from run_attaq_gated import build_conditions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--interleave-ns", type=int, nargs="+", default=[4, 7])
    ap.add_argument("--interleave-conditions", nargs="+", default=["ordered"],
                    choices=["ordered", "shuffled"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    records = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    records = [r for idx, r in enumerate(records) if idx % shard_n == shard_i]
    jobs = [(r, c) for r in records
            for c in build_conditions(r, args.seed, args.interleave_ns, args.interleave_conditions)]

    llm = LLM(model=args.target, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=8192, enforce_eager=False)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    started = time.time()
    results = llm.chat([[{"role": "user", "content": c["prompt"]}] for _, c in jobs],
                       sampling, use_tqdm=True)
    outputs = [r.outputs[0].text for r in results]

    rows = []
    for (record, cond), output in zip(jobs, outputs):
        if cond["is_direct"]:
            reconstructed, answer, rscore = cond["reconstructed"], output, 1.0
        else:
            reconstructed = section(output, "RECONSTRUCTED", "ANSWER")
            answer = section(output, "ANSWER")
            rscore = reconstruction_score(reconstructed, record["original"])
        rows.append({
            "item_id": record["item_id"], "scenario": record["scenario"],
            "risk_type": record.get("risk_type", ""), "condition": cond["condition"],
            "is_direct": int(cond["is_direct"]), "original": record["original"],
            "prompt": cond["prompt"], "raw_output": output,
            "reconstructed": reconstructed, "answer_section": answer,
            "reconstruction_score": round(rscore, 4),
            "reconstruction_pass_080": int(rscore >= 0.8),
            "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "target_model": args.target, "engine": "vllm",
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o700)
    output_path = outdir / "restricted_target_outputs.jsonl"
    secure_write(output_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    elapsed = time.time() - started
    summary = {
        "design": "AttaQ-multilingual gated-ASR (7 langs, translations reused verbatim), vLLM",
        "n_items": len(records), "n_generations": len(rows),
        "conditions": sorted({r["condition"] for r in rows}), "target": args.target,
        "decoding": {"temperature": 0, "max_new_tokens": args.max_new_tokens},
        "elapsed_seconds": round(elapsed, 2),
        "throughput_gen_per_s": round(len(rows) / elapsed, 2) if elapsed else None,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (outdir / "target_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
