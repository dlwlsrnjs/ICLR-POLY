#!/usr/bin/env python3
"""Generic two-phase (gen target-only / judge judges-only) driver for the single-phase MJ evals
(combo, method) so 14B survives shared-GPU contention. Reuses the exact prompt builders from
combo_eval and method_baselines_eval and writes output files in their original schema, so the
bandit reads them unchanged. Peak memory = max(target, judges) instead of the sum."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qwen_interleaving_probe as RQ
RQ.CJK_LANGUAGES.add("Thai")
from run_qwen_interleaving_probe import ENGLISH
from run_polyjig_gated import section
import combo_eval as CE
import method_baselines_eval as ME


def build_prompts(kind, rows, n, F, order, tlang):
    """Return dict {key: [prompt,...]} for the given eval kind."""
    if kind == "combo":
        langs = [ENGLISH] + order[:n - 1]
        return {v: [CE.prompt(r, langs, n, F, v) for r in rows] for v in CE.VARIANTS}
    return {m: ME.build_prompts(rows, m, tlang) for m in ME.METHODS}


def score(kind, rows, outs_by_key, judges):
    O = [r["original"] for r in rows]
    res = {}
    for key, outs in outs_by_key.items():
        R = [section(o, "RECONSTRUCTED", "ANSWER") or r["original"] for o, r in zip(outs, rows)]
        A = [section(o, "ANSWER") or o for o in outs]
        Rj = [bool(x) if x is not None else False for x in judges.recon_equivalent(O, R)]
        Uj = [bool(x) if x is not None else False for x in judges.answer_unsafe(O, A)]
        J = [1.0 if (a and b) else 0.0 for a, b in zip(Rj, Uj)]
        res[key] = dict(gated=round(float(np.mean(J)), 3), recon=round(float(np.mean(Rj)), 3),
                        unsafe=round(float(np.mean(Uj)), 3))
        print(json.dumps({"key": key, **res[key]}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["gen", "judge"])
    ap.add_argument("--kind", required=True, choices=["combo", "method"])
    ap.add_argument("--target", default=""); ap.add_argument("--tag", required=True)
    ap.add_argument("--order", default="private_artifacts/multijail_v1/resource_order.json")
    ap.add_argument("--harm", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--n-items", type=int, default=64); ap.add_argument("--tlang", default=ME.TLANG)
    ap.add_argument("--util", type=float, default=0.45); ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    raw = out / f"_raw2p_{a.tag}.json"
    order = json.loads(Path(a.order).read_text())["order"]
    rows = [json.loads(l) for l in open(a.harm)][:a.n_items]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])

    if a.phase == "gen":
        from online_live import LiveTarget
        print(json.dumps({"phase": "gen", "kind": a.kind, "tag": a.tag}), flush=True)
        target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
        prompts = build_prompts(a.kind, rows, a.n, a.F, order, a.tlang)
        outputs = {}
        for key, ps in prompts.items():
            outputs[key] = target.generate(ps)
            print(json.dumps({"gen_done": key}), flush=True)
        rec = {"kind": a.kind, "n": a.n, "outputs": outputs}
        fd = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as h:
            json.dump(rec, h)
        print(json.dumps({"phase": "gen", "saved": str(raw)}))
        return

    from online_live import Judges
    t0 = time.time()
    rec = json.loads(raw.read_text())
    judges = Judges("cuda:0")
    res = score(a.kind, rows, rec["outputs"], judges)
    if a.kind == "combo":
        doc = dict(target=a.tag, n=rec["n"], variants=res, seconds=round(time.time() - t0, 1))
    else:
        doc = dict(target=a.tag, methods={k: dict(unsafe=v["unsafe"], recon=v["recon"], gated=v["gated"])
                                          for k, v in res.items()}, seconds=round(time.time() - t0, 1))
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(doc, h, indent=2)
    print(json.dumps({"target": a.tag, "kind": a.kind, "done": True}))


if __name__ == "__main__":
    raise SystemExit(main())
