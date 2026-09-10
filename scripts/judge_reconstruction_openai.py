#!/usr/bin/env python3
"""Second reconstruction judge via the OpenAI API (GPT-4o / GPT-4o-mini) for gate
validation. Judges only genuine reconstruction rows (is_direct==0), using the same
system prompt and JSON schema as judge_reconstruction_equivalence.py, so its verdicts
are directly comparable to the primary Qwen2.5-7B gate. Aggregate agreement only."""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_reconstruction_equivalence import SYSTEM, parse  # noqa: E402
from run_polyjig_gated import secure_write  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sample", type=int, default=0, help="stratified sample size (0 = all non-direct)")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY not set"); return 2
    from openai import OpenAI
    client = OpenAI(api_key=key)

    rows = [json.loads(l) for l in Path(args.input).open(encoding="utf-8")]
    idx = [i for i, r in enumerate(rows) if not r.get("is_direct")]
    if args.sample and args.sample < len(idx):
        from collections import defaultdict
        buckets = defaultdict(list)
        for i in idx:
            buckets[(rows[i].get("scenario"), rows[i]["condition"])].append(i)
        rng = random.Random(args.seed); sel = []
        keys = sorted(buckets); per = max(1, args.sample // len(keys))
        for k in keys:
            b = buckets[k]; rng.shuffle(b); sel += b[:per]
        rng.shuffle(sel); idx = sel[:args.sample]

    def call(i):
        content = f"Original:\n{rows[i]['original']}\n\nReconstruction:\n{rows[i].get('reconstructed','')}"
        delay = 2.0
        for attempt in range(args.max_retries):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                    temperature=0.0, max_tokens=80, store=False)
                return i, r.choices[0].message.content or ""
            except Exception as exc:
                if attempt == args.max_retries - 1: return i, f"__API_ERROR__:{type(exc).__name__}"
                time.sleep(delay); delay = min(delay*2, 30)
        return i, "__API_ERROR__"

    started = time.time(); done = 0; out = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(call, i) for i in idx]):
            i, txt = fut.result(); out[i] = txt; done += 1
            if done % 200 == 0 or done == len(idx):
                print(json.dumps({"stage": "openai_recon", "completed": done, "total": len(idx)}), flush=True)

    audit = []
    errors = 0
    for i in idx:
        raw = out.get(i, "")
        if raw.startswith("__API_ERROR__"): errors += 1; equiv = None
        else: equiv, _, _ = parse(raw)
        r = rows[i]
        audit.append({"item_id": r["item_id"], "condition": r["condition"],
                      "is_direct": int(r.get("is_direct", 0)),
                      "reconstruction_parse_valid": int(equiv is not None),
                      "semantic_reconstruction_equivalent": int(equiv) if equiv is not None else "",
                      "judge_model": args.model})
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True); os.chmod(outdir, 0o700)
    ap_path = outdir / "restricted_reconstruction_audit.jsonl"
    secure_write(ap_path, "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))
    summary = {"model": args.model, "n_judged": len(idx), "api_errors": errors,
               "sample": args.sample or "all_nondirect", "elapsed_seconds": round(time.time()-started, 2),
               "audit_sha256": hashlib.sha256(ap_path.read_bytes()).hexdigest()}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
