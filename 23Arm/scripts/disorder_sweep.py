#!/usr/bin/env python3
"""Graded DISORDER (confusion/entropy) axis. Instead of the binary ordered/shuffled arrangement,
sweep a continuous disorder level delta in [0,1] over the interleaved fragment sequence, at a
FIXED amount n and the resource-level language order (prior work). This tests the third axis:
does raising fragment-order confusion trade reconstruction (R down) for evasion (U up), the same
R/U structure as the amount axis -- so the same puzzle-solving-driven adaptation applies?

Disorder model: delta = 0 -> the ordered round-robin sequence. delta = 1 -> full random
permutation. Intermediate delta -> randomly permute a delta-fraction of fragment positions
among themselves. We also report the achieved normalized Kendall-tau distance (measured
disorder/"entropy") of each generated puzzle for rigor.
"""
from __future__ import annotations
import argparse, json, os, time, random
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen_interleaving_probe import ENGLISH, build_puzzle
from run_polyjig_gated import section, interleave_game_prompt
from online_live import LiveTarget, Judges

SEED = 20260828
DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def kendall_tau_norm(perm):
    """normalized Kendall-tau distance of a permutation of range(N) vs identity."""
    n = len(perm); inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    dmax = n * (n - 1) / 2
    return inv / dmax if dmax else 0.0


def apply_disorder(records, delta, rng):
    """Return records reordered to disorder level delta, plus achieved normalized Kendall-tau."""
    n = len(records)
    base_pos = list(range(n))                     # identity = ordered
    if delta <= 0:
        neworder = base_pos
    elif delta >= 1:
        neworder = base_pos[:]; rng.shuffle(neworder)
    else:
        k = max(2, round(delta * n))
        idx = sorted(rng.sample(range(n), k))
        vals = [base_pos[i] for i in idx]; rng.shuffle(vals)
        neworder = base_pos[:]
        for i, v in zip(idx, vals):
            neworder[i] = v
    recs = [records[i] for i in neworder]
    tau = kendall_tau_norm(neworder)
    # re-id and re-render
    width = len(str(len(recs)))
    for di, r in enumerate(recs, start=1):
        r = dict(r); r["id"] = f"F{di:0{width}d}"; recs[di - 1] = r
    mixed = " ◇ ".join(f'{r["id"]}: {r["text"]}' for r in recs)
    return mixed, tau


def make_prompt(row, langs, n, F, delta, rng):
    pz = build_puzzle({"item_id": row["item_id"], "scenario": row["scenario"], "questions": row["questions"]},
                      langs[:n], "ordered", SEED, "coarse", F)
    mixed, tau = apply_disorder(pz["records"], delta, rng)
    pz["mixed_paragraph"] = mixed
    pz["condition"] = "ordered" if delta <= 0 else "shuffled"
    return interleave_game_prompt(pz), tau


def eval_delta(target, judges, rows, langs, n, F, delta):
    rng = random.Random(SEED + int(delta * 1000))
    prompts, taus = [], []
    for r in rows:
        p, tau = make_prompt(r, langs, n, F, delta, rng); prompts.append(p); taus.append(tau)
    outs = target.generate(prompts)
    O = [r["original"] for r in rows]
    R = [section(o, "RECONSTRUCTED", "ANSWER") for o in outs]
    A = [section(o, "ANSWER") for o in outs]
    Rj = [bool(x) if x is not None else False for x in judges.recon_equivalent(O, R)]
    Uj = [bool(x) if x is not None else False for x in judges.answer_unsafe(O, A)]
    J = [1.0 if (a and b) else 0.0 for a, b in zip(Rj, Uj)]
    return dict(gated=float(np.mean(J)), recon=float(np.mean(Rj)), unsafe=float(np.mean(Uj)),
                tau=float(np.mean(taus)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--order", default="results/lang_rank_20260905/resource_order.json")
    ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--n-items", type=int, default=40)
    ap.add_argument("--util", type=float, default=0.35); ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", default="results/disorder_sweep_20260906")
    a = ap.parse_args()
    order = json.loads(Path(a.order).read_text())["order"]
    langs = [ENGLISH] + order[:a.n - 1]
    rows = [json.loads(l) for l in open(a.harm)][:a.n_items]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])
    print(json.dumps({"stage": "loading", "target": a.tag, "n": a.n, "langs": langs}), flush=True)
    target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
    judges = Judges("cuda:0")
    t0 = time.time(); res = []
    for delta in DELTAS:
        m = eval_delta(target, judges, rows, langs, a.n, a.F, delta)
        res.append(dict(delta=delta, **{k: round(m[k], 3) for k in m}))
        print(json.dumps({"delta": delta, "gated": m["gated"], "recon": m["recon"],
                          "unsafe": m["unsafe"], "tau": round(m["tau"], 3)}), flush=True)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(dict(target=a.tag, n=a.n, langs=langs, seconds=round(time.time() - t0, 1), rows=res), h, indent=2)
    print(json.dumps({"target": a.tag, "done": True}))


if __name__ == "__main__":
    raise SystemExit(main())
