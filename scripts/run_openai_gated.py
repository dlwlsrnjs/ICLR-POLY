#!/usr/bin/env python3
"""Frontier-model gated-ASR generation via the OpenAI API (e.g., GPT-4o-mini).

Mirrors run_text_gated_vllm.py's conditions (english / translation / CSRT /
interleave / slot / no-game) and output schema exactly, so the SAME reconstruction
gate, Qwen3Guard, MD-Judge, HR judges and compare_methods.py apply unchanged. This
adds a strong-alignment commercial target to complement the open models.

Cost control: restrict to a split (--split-file/--split), cap items
(--max-items), and/or keep only headline conditions (--conditions regex). Only
aggregate ASR is summarized; raw completions are controlled content -> 0600.

The API key is read from $OPENAI_API_KEY only; it is never logged, stored, or
echoed. Responses are requested with store=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polyjig_pilot import reconstruction_score  # noqa: E402
from run_polyjig_gated import build_conditions, section, secure_write  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--interleave-ns", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--interleave-conditions", nargs="+", default=["ordered"])
    ap.add_argument("--translated-langs", nargs="*", default=["Finnish"])
    ap.add_argument("--no-slots", action="store_true")
    ap.add_argument("--with-nogame", action="store_true")
    ap.add_argument("--with-csrt", action="store_true")
    ap.add_argument("--csrt-ks", type=int, nargs="+", default=[2])
    ap.add_argument("--conditions", default=None,
                    help="regex; keep only conditions whose name matches (cost control)")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--split", default=None, choices=["dev", "test"])
    ap.add_argument("--max-items", type=int, default=None,
                    help="cap number of items (after split filter) for cost control")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: set OPENAI_API_KEY in the environment (never pass it on the "
              "command line or commit it).", flush=True)
        return 2
    from openai import OpenAI
    client = OpenAI(api_key=key)

    records = [json.loads(l) for l in Path(args.data).open(encoding="utf-8")]
    if args.split_file and args.split:
        keep = {json.loads(l)["item_id"] for l in Path(args.split_file).open(encoding="utf-8")
                if json.loads(l).get("split") == args.split}
        records = [r for r in records if r["item_id"] in keep]
    if args.max_items:
        records = records[: args.max_items]

    cond_re = re.compile(args.conditions) if args.conditions else None
    jobs = []
    for record in records:
        for cond in build_conditions(record, args.seed, args.interleave_ns,
                                     list(record.get("questions", {})),
                                     translated_langs=args.translated_langs,
                                     interleave_conditions=args.interleave_conditions,
                                     include_slots=not args.no_slots,
                                     include_nogame=args.with_nogame,
                                     include_csrt=args.with_csrt, csrt_ks=args.csrt_ks):
            if cond_re and not cond_re.search(cond["condition"]):
                continue
            jobs.append((record, cond))

    def call(prompt: str) -> str:
        delay = 2.0
        for attempt in range(args.max_retries):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=args.max_new_tokens, store=False)
                return r.choices[0].message.content or ""
            except Exception as exc:  # rate limit / transient
                if attempt == args.max_retries - 1:
                    return f"__API_ERROR__: {type(exc).__name__}"
                time.sleep(delay); delay = min(delay * 2, 30)
        return "__API_ERROR__"

    started = time.time()
    outputs: list[str] = [None] * len(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut = {ex.submit(call, c["prompt"]): i for i, (_, c) in enumerate(jobs)}
        for f in as_completed(fut):
            outputs[fut[f]] = f.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(json.dumps({"stage": "openai", "completed": done,
                                  "total": len(jobs)}), flush=True)

    rows, api_errors = [], 0
    for (record, cond), output in zip(jobs, outputs):
        if output.startswith("__API_ERROR__"):
            api_errors += 1
            continue
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
            "target_model": args.model, "engine": "openai",
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o700)
    op = outdir / "restricted_target_outputs.jsonl"
    secure_write(op, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    elapsed = time.time() - started
    summary = {
        "design": "frontier gated-ASR (OpenAI API), schema-compatible with the open-model runs",
        "model": args.model, "n_items": len(records), "n_generations": len(rows),
        "api_errors": api_errors, "conditions": sorted({r["condition"] for r in rows}),
        "split": args.split or "all", "elapsed_seconds": round(elapsed, 2),
        "output_sha256": hashlib.sha256(op.read_bytes()).hexdigest(),
    }
    (outdir / "target_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
