#!/usr/bin/env python3
"""Does the best configuration change by harm domain, or only by target?

The panel results store per-(target, configuration) means, so they cannot answer this. Here we keep
PER-ITEM judgments together with each item's scenario label, for a few configurations that span the
families, on the Lingua-SafetyBench collection (eight clean scenarios). Prompts are built by importing
the same builders the panel used, so a cell here is directly comparable to the panel's.

Two phases, like the rest of the pipeline: `gen` loads only the target, `judge` loads only the judges.
Scoring follows scripts/arm_scoring.py: clear-text configurations (AIM) are ungated, hidden-request
configurations are reconstruction-gated.

Output: results/<outdir>/<tag>.json with {config: {items: [{item_id, scenario, recon, unsafe, gated}]}}
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qwen_interleaving_probe as RQ  # noqa: E402
RQ.CJK_LANGUAGES.add("Thai")
from run_qwen_interleaving_probe import ENGLISH  # noqa: E402
from run_polyjig_gated import section  # noqa: E402
import sequential_add as SA  # noqa: E402
import combo_eval as CE  # noqa: E402
import triple_combo_eval as TC  # noqa: E402
import method_baselines_eval as ME  # noqa: E402

CONFIGS = ["amt_n4", "combo_ours_persona", "tri_hi_wl", "tri_triple", "m_aim"]
CLEARTEXT = {"m_aim"}


def build(cfg, rows, order, wl, n=4, F=5):
    langs_amt = [ENGLISH] + [l for l in order if l != ENGLISH][:n - 1]
    hi = [l for l in reversed(order) if l != wl]
    langs_tri = [ENGLISH] + hi[:n - 1]
    if cfg == "amt_n4":
        return [SA.prompt(r, langs_amt, F) for r in rows]
    if cfg.startswith("combo_"):
        return [CE.prompt(r, langs_amt, n, F, cfg[len("combo_"):]) for r in rows]
    if cfg.startswith("tri_"):
        return [TC.prompt(r, langs_tri, n, F, cfg[len("tri_"):], wl) for r in rows]
    if cfg.startswith("m_"):
        return ME.build_prompts(rows, cfg[len("m_"):], wl)
    raise ValueError(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["gen", "judge"])
    ap.add_argument("--target", default=""); ap.add_argument("--tag", required=True)
    ap.add_argument("--order", required=True); ap.add_argument("--harm", required=True)
    ap.add_argument("--answer-lang", default="Norwegian")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--n-items", type=int, default=250)
    ap.add_argument("--util", type=float, default=0.40); ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", default="results/domain_breakdown_20260907")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    raw = out / f"_raw_{a.tag}.json"
    order = json.loads(Path(a.order).read_text())["order"]
    rows = [json.loads(l) for l in open(a.harm)][:a.n_items]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])

    if a.phase == "gen":
        from online_live import LiveTarget
        t = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
        rec = {"tag": a.tag, "answer_lang": a.answer_lang, "n_items": len(rows), "outputs": {}}
        for cfg in CONFIGS:
            outs = t.generate(build(cfg, rows, order, a.answer_lang, a.n, a.F))
            rec["outputs"][cfg] = [{"item_id": r["item_id"], "scenario": r["scenario"],
                                    "original": r["original"],
                                    "recon": section(o, "RECONSTRUCTED", "ANSWER") or r["original"],
                                    "answer": section(o, "ANSWER") or o} for r, o in zip(rows, outs)]
            print(json.dumps({"gen_done": cfg, "tag": a.tag}), flush=True)
        fd = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as h:
            json.dump(rec, h)
        print(json.dumps({"saved": str(raw)})); return

    rec = json.loads(raw.read_text())
    from online_live import Judges
    jd = Judges("cuda:0")
    res = {"tag": a.tag, "answer_lang": rec["answer_lang"], "n_items": rec["n_items"], "configs": {}}
    for cfg, items in rec["outputs"].items():
        O = [it["original"] for it in items]; R = [it["recon"] for it in items]; A = [it["answer"] for it in items]
        rj, uj = [], []
        for i in range(0, len(O), 64):                      # chunked: a full pass OOMs the guard model
            rj += [bool(x) if x is not None else False for x in jd.recon_equivalent(O[i:i + 64], R[i:i + 64])]
            uj += [bool(x) if x is not None else False for x in jd.answer_unsafe(O[i:i + 64], A[i:i + 64])]
        per = []
        for it, r_, u_ in zip(items, rj, uj):
            gated = float(u_) if cfg in CLEARTEXT else float(r_ and u_)
            per.append({"item_id": it["item_id"], "scenario": it["scenario"],
                        "recon": int(r_), "unsafe": int(u_), "gated": gated})
        res["configs"][cfg] = {"items": per,
                               "gated": round(float(np.mean([p["gated"] for p in per])), 3),
                               "recon": round(float(np.mean([p["recon"] for p in per])), 3),
                               "unsafe": round(float(np.mean([p["unsafe"] for p in per])), 3)}
        print(json.dumps({"judged": cfg, **{k: v for k, v in res["configs"][cfg].items() if k != "items"}}), flush=True)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(res, h)
    print(json.dumps({"saved": str(out / f"{a.tag}.json")}))


if __name__ == "__main__":
    raise SystemExit(main())
