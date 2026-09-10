#!/usr/bin/env python3
"""Two-phase (gen target-only / judge judges-only) driver for the amount (sequential) and disorder
arm families, so 32B/27B bf16 survive on a single GPU (target 64GB gen-only, judges 31GB separately).
Reuses the exact prompt logic from sequential_add and disorder_sweep and writes output files in their
schemas (steps / rows), so the bandit reads them unchanged."""
from __future__ import annotations
import argparse, json, os, time, random
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qwen_interleaving_probe as RQ
RQ.CJK_LANGUAGES.add("Thai")
from run_qwen_interleaving_probe import ENGLISH, build_puzzle
from run_polyjig_gated import section, interleave_game_prompt
import sequential_add as SA
import disorder_sweep as DZ


def amount_prompts(rows, order, F, seed):
    SA._PZ_SEED = SA.SEED + seed
    steps = []
    for k in range(1, len(order) + 1):
        langs = [ENGLISH] + order[:k]
        steps.append(dict(k=k, n=k + 1, langs=order[:k],
                          prompts=[SA.prompt(r, langs, F) for r in rows]))
    return steps


def disorder_prompts(rows, order, n, F):
    langs = [ENGLISH] + order[:n - 1]
    out = []
    for delta in DZ.DELTAS:
        rng = random.Random(DZ.SEED + int(delta * 1000))
        ps, taus = [], []
        for r in rows:
            p, tau = DZ.make_prompt(r, langs, n, F, delta, rng); ps.append(p); taus.append(tau)
        out.append(dict(delta=delta, tau=float(np.mean(taus)), prompts=ps))
    return out


def score(judges, rows, outs):
    O = [r["original"] for r in rows]
    R = [section(o, "RECONSTRUCTED", "ANSWER") for o in outs]
    A = [section(o, "ANSWER") for o in outs]
    Rj = [bool(x) if x is not None else False for x in judges.recon_equivalent(O, R)]
    Uj = [bool(x) if x is not None else False for x in judges.answer_unsafe(O, A)]
    J = [1.0 if (a and b) else 0.0 for a, b in zip(Rj, Uj)]
    return dict(gated=round(float(np.mean(J)), 3), recon=round(float(np.mean(Rj)), 3), unsafe=round(float(np.mean(Uj)), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["gen", "judge"])
    ap.add_argument("--kind", required=True, choices=["amount", "disorder"])
    ap.add_argument("--target", default=""); ap.add_argument("--tag", required=True)
    ap.add_argument("--order", required=True); ap.add_argument("--harm", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--n-items", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--util", type=float, default=0.85); ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    raw = out / f"_raw_ad_{a.tag}.json"
    order = json.loads(Path(a.order).read_text())["order"]
    rows = [json.loads(l) for l in open(a.harm)][:a.n_items]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])

    if a.phase == "gen":
        from online_live import LiveTarget
        print(json.dumps({"phase": "gen", "kind": a.kind, "tag": a.tag}), flush=True)
        target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
        if a.kind == "amount":
            groups = amount_prompts(rows, order, a.F, a.seed)
            for g in groups:
                g["outputs"] = target.generate(g.pop("prompts")); print(json.dumps({"gen_k": g["k"]}), flush=True)
            rec = {"kind": "amount", "order": order, "groups": groups}
        else:
            groups = disorder_prompts(rows, order, a.n, a.F)
            for g in groups:
                g["outputs"] = target.generate(g.pop("prompts")); print(json.dumps({"gen_delta": g["delta"]}), flush=True)
            rec = {"kind": "disorder", "n": a.n, "langs": [ENGLISH] + order[:a.n - 1], "groups": groups}
        fd = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as h:
            json.dump(rec, h)
        print(json.dumps({"phase": "gen", "saved": str(raw)}))
        return

    from online_live import Judges
    t0 = time.time(); rec = json.loads(raw.read_text()); judges = Judges("cuda:0")
    if rec["kind"] == "amount":
        steps = []
        for g in rec["groups"]:
            s = score(judges, rows, g["outputs"])
            steps.append(dict(k=g["k"], n=g["n"], langs=g["langs"], **s))
            print(json.dumps({"k": g["k"], **s}), flush=True)
        doc = dict(target=a.tag, order=rec["order"], seconds=round(time.time() - t0, 1), steps=steps)
    else:
        rows_out = []
        for g in rec["groups"]:
            s = score(judges, rows, g["outputs"])
            rows_out.append(dict(delta=g["delta"], **s, tau=round(g["tau"], 3)))
            print(json.dumps({"delta": g["delta"], **s}), flush=True)
        doc = dict(target=a.tag, n=rec["n"], langs=rec["langs"], seconds=round(time.time() - t0, 1), rows=rows_out)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(doc, h, indent=2)
    print(json.dumps({"target": a.tag, "kind": a.kind, "done": True}))


if __name__ == "__main__":
    raise SystemExit(main())
