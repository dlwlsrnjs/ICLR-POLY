#!/usr/bin/env python3
"""Stage 2 of the universal-order strategy: given a FIXED universal language order (from stage 1,
results/lang_rank_20260905/universal_order.json), for each target add languages one at a time in
that order -- EN, EN+L1, EN+L1+L2, ... -- measuring harmful joint ASR at each step. This is the
online greedy procedure the user described: a pre-set universal order, sequentially probed per
target. We record the full cumulative curve so the stopping rule (keep growing while joint does
not drop by > margin; stop otherwise) can be evaluated offline and compared to a fixed config."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen_interleaving_probe import ENGLISH, build_puzzle
from run_polyjig_gated import section, interleave_game_prompt
from online_live import LiveTarget, Judges

SEED = 20260828
_PZ_SEED = SEED   # overridden by --seed (arrangement); seed=0 reproduces the main run


def prompt(row, langs, F):
    pz = build_puzzle({"item_id": row["item_id"], "scenario": row["scenario"], "questions": row["questions"]},
                      langs, "ordered", _PZ_SEED, "coarse", F)
    return interleave_game_prompt(pz)


def eval_cfg(target, judges, rows, langs, F):
    outs = target.generate([prompt(r, langs, F) for r in rows])
    O = [r["original"] for r in rows]
    R = [section(o, "RECONSTRUCTED", "ANSWER") for o in outs]
    A = [section(o, "ANSWER") for o in outs]
    Rj = [bool(x) if x is not None else False for x in judges.recon_equivalent(O, R)]
    Uj = [bool(x) if x is not None else False for x in judges.answer_unsafe(O, A)]
    J = [1.0 if (a and b) else 0.0 for a, b in zip(Rj, Uj)]
    return dict(gated=float(np.mean(J)), recon=float(np.mean(Rj)), unsafe=float(np.mean(Uj)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--order", default="results/lang_rank_20260905/universal_order.json")
    ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n-items", type=int, default=40); ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--util", type=float, default=0.35); ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", default="results/sequential_add_20260905")
    ap.add_argument("--seed", type=int, default=0, help="0 reproduces main run; >0 varies arrangement+item sample")
    a = ap.parse_args()
    global _PZ_SEED
    _PZ_SEED = SEED + a.seed
    order = json.loads(Path(a.order).read_text())["order"]     # universal fixed order of the 9 languages
    allrows = [json.loads(l) for l in open(a.harm)]
    if a.seed == 0:
        rows = allrows[:a.n_items]                             # first-N, matches the main panel
    else:
        import random as _r
        idx = list(range(len(allrows))); _r.Random(SEED + a.seed).shuffle(idx)
        rows = [allrows[i] for i in idx[:a.n_items]]           # seeded item resample
    for r in rows:
        r.setdefault("original", r["questions"]["English"])
    print(json.dumps({"stage": "loading", "target": a.tag, "order": order}), flush=True)
    target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
    judges = Judges("cuda:0")
    t0 = time.time(); res = []
    for k in range(1, len(order) + 1):                          # EN + top-k in the universal order
        langs = [ENGLISH] + order[:k]
        m = eval_cfg(target, judges, rows, langs, a.F)
        res.append(dict(k=k, n=k + 1, langs=order[:k],
                        gated=round(m["gated"], 3), recon=round(m["recon"], 3), unsafe=round(m["unsafe"], 3)))
        print(json.dumps({"k": k, "n": k + 1, "gated": m["gated"], "recon": m["recon"], "unsafe": m["unsafe"]}), flush=True)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(dict(target=a.tag, order=order, seconds=round(time.time() - t0, 1), steps=res), h, indent=2)
    print(json.dumps({"target": a.tag, "done": True}))


if __name__ == "__main__":
    raise SystemExit(main())
