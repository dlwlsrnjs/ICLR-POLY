#!/usr/bin/env python3
"""Final-ASR baselines per panel model: the reference points our adaptive method is compared to.
  english_direct     : plain English harmful question, no obfuscation -> raw unsafe rate (ASR floor,
                       = alignment strength). Reconstruction is trivial, so gated == unsafe.
  translated_direct  : the SAME question asked in ONE foreign language (MultiJail 'unintentional'
                       single-language attack). Answer judged for unsafe; recon trivial.
We collect these for every panel model so the definitive table reads
  english_direct < translated_direct < fixed PolyJigsaw < adaptive(amount) < adaptive(amount+disorder).
The interleaved/fixed/adaptive/nogame numbers already exist in the other result dirs; this fills
the two missing anchors. translated_direct uses the resource-order top language by default."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_polyjig_gated import section
from online_live import LiveTarget, Judges


def unsafe_rate(target, judges, prompts, originals):
    outs = target.generate(prompts)
    # answers may or may not carry an [ANSWER] tag; use the tagged section if present else whole text
    ans = [section(o, "ANSWER") or o for o in outs]
    Uj = [bool(x) if x is not None else False for x in judges.answer_unsafe(originals, ans)]
    return float(np.mean(Uj)), outs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--order", default="results/lang_rank_20260905/resource_order.json")
    ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n-items", type=int, default=40)
    ap.add_argument("--util", type=float, default=0.35); ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", default="results/baselines_20260906")
    a = ap.parse_args()
    order = json.loads(Path(a.order).read_text())["order"]
    tlang = order[0]                                   # resource-order top language (e.g. Norwegian)
    rows = [json.loads(l) for l in open(a.harm)][:a.n_items]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])
    O = [r["original"] for r in rows]
    print(json.dumps({"stage": "loading", "target": a.tag, "translated_lang": tlang}), flush=True)
    target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
    judges = Judges("cuda:0")
    t0 = time.time()
    ed_prompts = [r["questions"]["English"] for r in rows]
    td_prompts = [r["questions"].get(tlang, r["questions"]["English"]) for r in rows]
    ed_u, _ = unsafe_rate(target, judges, ed_prompts, O)
    td_u, _ = unsafe_rate(target, judges, td_prompts, O)
    res = dict(target=a.tag, translated_lang=tlang,
               english_direct=round(ed_u, 3), translated_direct=round(td_u, 3),
               seconds=round(time.time() - t0, 1))
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(res, h, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    raise SystemExit(main())
