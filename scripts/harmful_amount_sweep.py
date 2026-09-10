#!/usr/bin/env python3
"""Does INCREASING the amount of multilingual benign text (number of translation copies n)
improve jailbreak, a la Deng et al. ACL 2024 (multilingual combination raises unsafe rate)?

Unlike harmful_weaklang_test (which picked ONE weak language at small n), this scales the
NUMBER of language translations n = 2..10, using the target's benign-best-decoded languages
(so reconstruction has the best chance to survive). It reports the three components
separately -- reconstruction R (does more text break decoding?) and raw-unsafe U (does more
multilingual obfuscation raise evasion?) and the joint gated ASR J = R AND U -- so we can see
the R-down / U-up tradeoff as amount grows. Compared at each n against the FIXED language order.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen_interleaving_probe import ENGLISH, select_languages, build_puzzle
from run_polyjig_gated import section, interleave_game_prompt
from online_live import LiveTarget, Judges

SEED = 20260828
NS = (2, 3, 4, 6, 8, 10)


def prompt(row, langs, n, F, arr):
    pz = build_puzzle({"item_id": row["item_id"], "scenario": row["scenario"], "questions": row["questions"]},
                      langs[:n], arr, SEED, "coarse", F)
    return interleave_game_prompt(pz)


def eval_cfg(target, judges, rows, langs, n, F, arr):
    outs = target.generate([prompt(r, langs, n, F, arr) for r in rows])
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
    ap.add_argument("--benign-sweep", default="results/benign_recon_sweep_20260904")
    ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n-items", type=int, default=40); ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--util", type=float, default=0.35); ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", default="results/harmful_amount_20260905")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.harm)][:a.n_items]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])
    bs = json.loads((Path(a.benign_sweep) / f"{a.tag}.json").read_text())
    langd = {c["name"][3:]: c["recon_pass"] for c in bs["configs"]
             if c["name"].startswith("EN+") and c["n"] == 2 and "CJK" not in c["name"]}
    weak_ranked = sorted(langd, key=langd.get, reverse=True)     # best-decoded single langs first
    print(json.dumps({"stage": "loading", "target": a.tag, "weak_ranked": weak_ranked}), flush=True)
    target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
    judges = Judges("cuda:0")
    t0 = time.time(); res = []
    for n in NS:
        if n - 1 > len(weak_ranked):
            break
        fixed_langs = select_languages(n)
        weak_langs = [ENGLISH] + weak_ranked[:n - 1]
        f = eval_cfg(target, judges, rows, fixed_langs, n, a.F, "ordered")
        w = eval_cfg(target, judges, rows, weak_langs, n, a.F, "ordered")
        res.append(dict(n=n, weak_langs=weak_langs[:n],
                        fixed_gated=round(f["gated"], 3), weak_gated=round(w["gated"], 3),
                        fixed_recon=round(f["recon"], 3), weak_recon=round(w["recon"], 3),
                        fixed_unsafe=round(f["unsafe"], 3), weak_unsafe=round(w["unsafe"], 3),
                        delta=round(w["gated"] - f["gated"], 3)))
        print(json.dumps({"n": n, "fixed_gated": f["gated"], "weak_gated": w["gated"],
                          "weak_recon": w["recon"], "weak_unsafe": w["unsafe"]}), flush=True)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(dict(target=a.tag, weak_ranked=weak_ranked, seconds=round(time.time() - t0, 1), rows=res), h, indent=2)
    print(json.dumps({"target": a.tag, "done": True}))


if __name__ == "__main__":
    raise SystemExit(main())
