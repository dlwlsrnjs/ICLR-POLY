#!/usr/bin/env python3
"""Panel + held-out evaluation of two SOTA decomposition/obfuscation attacks (DrAttack, FlipAttack)
under OUR reconstruction-gated verified scoring, so they become first-class rows in the headline table
(Table~\\ref{tab:main}) rather than a single-target contrast. Both HIDE the request, so they are scored
STRICTLY (reconstruction-gated), exactly like our hidden-request configs.

Two-phase to fit every model size on one card:
  --phase gen   : load the target only, build DrAttack + FlipAttack(FCS) prompts on the harm items,
                  generate, write raw outputs (0600) to <outdir>/_raw_<tag>.json.
  --phase judge : load the two judges only (recon Qwen2.5-7B + safety Qwen3Guard-8B), read raw, gate
                  by reconstruction-equivalence AND answer-unsafe, write <outdir>/<tag>.json.

Prompts reuse scripts/build_sota_baselines.py verbatim. Issues real harmful queries -> authorized use only.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_polyjig_gated import section  # noqa: E402
from build_sota_baselines import flipattack_prompt, drattack_prompt  # noqa: E402

CONDS = ["drattack", "flipattack_fcs"]


def build_prompt(cond, req):
    return drattack_prompt(req) if cond == "drattack" else flipattack_prompt(req, "FCS")


def load_items(harm, n_items):
    rows = [json.loads(l) for l in open(harm)]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])
    return rows[:n_items] if n_items else rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["gen", "judge"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--harm", required=True)
    ap.add_argument("--n-items", type=int, default=0)
    ap.add_argument("--util", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    raw_path = out / f"_raw_{a.tag}.json"
    items = load_items(a.harm, a.n_items)

    if a.phase == "gen":
        from online_live import LiveTarget
        target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
        rec = []
        for cond in CONDS:
            prompts = [build_prompt(cond, r["original"]) for r in items]
            outs = target.generate(prompts)
            for r, o in zip(items, outs):
                rec.append({"item_id": r["item_id"], "cond": cond, "original": r["original"], "output": o})
        fd = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as h:
            json.dump(rec, h)
        print(json.dumps({"stage": "gen_done", "tag": a.tag, "n": len(rec)}))
        return 0

    # judge phase
    from online_live import Judges
    rec = json.loads(raw_path.read_text())
    judges = Judges("cuda:0")
    res = {}
    for cond in CONDS:
        sub = [x for x in rec if x["cond"] == cond]
        originals = [x["original"] for x in sub]
        recons = [section(x["output"], "RECONSTRUCTED", "ANSWER") for x in sub]
        answers = [section(x["output"], "ANSWER") or x["output"] for x in sub]
        R = judges.recon_equivalent(originals, recons)
        R = [bool(r) if r is not None else False for r in R]
        U = judges.answer_unsafe(originals, answers)
        U = [bool(u) if u is not None else False for u in U]
        gated = [1.0 if (r and u) else 0.0 for r, u in zip(R, U)]
        res[cond] = {"unsafe": round(float(np.mean(U)), 3),
                     "recon": round(float(np.mean(R)), 3),
                     "gated": round(float(np.mean(gated)), 3)}
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump({"target": a.tag, "conds": res}, h, indent=1)
    print(json.dumps({"stage": "judge_done", "tag": a.tag, **res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
