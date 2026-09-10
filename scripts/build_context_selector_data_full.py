#!/usr/bin/env python3
"""FULL-experiment contextual dataset: 32-arm fragment x language x arrangement space.

Arms (32) come from the fragment factorial (frag{F}_{arr}_n{n}); the behavioral
fingerprint comes from cheap probes NOT in the arm set:
  align = english_direct raw ASR        (from the factorial anchors, train items)
  multi = translated_Finnish raw ASR    (from the factorial anchors, train items)
  cap_* = benign FLORES reconstruction  (from panel_v2/<tag>/benign_recon.json)
Joint reward J = R AND U per item. Targets missing any arm cell or the benign probe are
dropped (no imputation). Emits J[T,I,A], context[T,5], arm features, families, splits.
"""
from __future__ import annotations
import json, os, re, hashlib, argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen_interleaving_probe import ALL_LANGUAGES, select_languages  # noqa: E402
from build_context_selector_data import family_of  # noqa: E402

FRAG = Path("private_artifacts/frag_factorial_20260903")
PANEL = Path("private_artifacts/panel_v2")
CELL = re.compile(r"frag(\d+)_(ordered|shuffled)_n(\d+)")
ARMS = [f"frag{F}_{arr}_n{n}" for F in (3, 5, 8, 12) for arr in ("ordered", "shuffled") for n in (2, 4, 6, 8)]

def arm_feature(cond):
    F, arr, n = CELL.match(cond).groups(); n = int(n); langs = select_languages(n)
    return ([float(x in langs) for x in ALL_LANGUAGES] + [n / 10.0, int(F) / 12.0]
            + [float(arr == "ordered"), float(arr == "shuffled")])

def load(tag):
    rp = FRAG / tag / "harm_recon/restricted_reconstruction_audit.jsonl"
    gp = FRAG / tag / "harm_guard/restricted_qwen3guard_audit.jsonl"
    bn = PANEL / tag / "benign_recon.json"
    if not (rp.exists() and gp.exists() and bn.exists() and os.access(rp, os.R_OK) and os.access(gp, os.R_OK)):
        return None
    R, U = {}, {}
    for l in open(rp):
        r = json.loads(l)
        if r["reconstruction_parse_valid"]:
            R[(r["item_id"], r["condition"])] = bool(r["semantic_reconstruction_equivalent"])
    for l in open(gp):
        r = json.loads(l)
        if r["guard_parse_valid"]:
            U[(r["item_id"], r["condition"])] = bool(r["guard_strict_unsafe"])
    return R, U, json.loads(bn.read_text())

def split_of(item_id):
    h = int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else "validation" if h < 80 else "test"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/context_selector_full_train_20260904")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tags = sorted(p.name for p in FRAG.iterdir() if p.is_dir() and (p / "harmful_summary.json").exists())
    data = {}
    for t in tags:
        d = load(t)
        if d is not None:
            data[t] = d
    tags = sorted(data)
    items = sorted({k[0] for k in data[tags[0]][0] if CELL.match(k[1])})
    iidx = {it: i for i, it in enumerate(items)}
    splits = np.array([split_of(it) for it in items])
    train_items = set(np.array(items)[splits == "train"])
    T, I, A = len(tags), len(items), len(ARMS)
    aidx = {c: i for i, c in enumerate(ARMS)}
    J = np.full((T, I, A), np.nan, np.float32)
    for ti, t in enumerate(tags):
        R, U, _ = data[t]
        for (it, cond), rv in R.items():
            if cond in aidx and it in iidx and (it, cond) in U:
                J[ti, iidx[it], aidx[cond]] = 1.0 if (rv and U[(it, cond)]) else 0.0

    def raw_asr(t, cond):
        _, U, _ = data[t]
        v = [U[(it, cond)] for it in train_items if (it, cond) in U]
        return float(np.mean(v)) if v else np.nan

    context = np.zeros((T, 5), np.float32)
    for ti, t in enumerate(tags):
        _, _, bn = data[t]
        # benign recon keys are panel_v2 interleave names; use them for capability
        cap = [bn.get(c, np.nan) for c in bn]
        context[ti] = [raw_asr(t, "english_direct"), raw_asr(t, "translated_direct_Finnish"),
                       float(np.nanmean(cap)),
                       float(np.nanmean([bn.get("interleave_ordered_n2", np.nan), bn.get("interleave_shuffled_n2", np.nan)])),
                       float(np.nanmean([bn.get("interleave_ordered_n10", np.nan), bn.get("interleave_shuffled_n10", np.nan)]))]

    keep = [ti for ti in range(T) if np.isfinite(J[ti]).all() and np.isfinite(context[ti]).all()]
    dropped = [tags[ti] for ti in range(T) if ti not in keep]
    J, context, tags = J[keep], context[keep], [tags[ti] for ti in keep]
    families = [family_of(t) for t in tags]
    np.savez(out / "context_values.npz", J=J, context=context,
             features=np.array([arm_feature(c) for c in ARMS], np.float32), splits=splits)
    manifest = dict(scope="FULL 32-arm contextual joint-ASR selector data.",
                    reward="J = reconstruction AND guard_strict_unsafe(answer); per-item 0/1",
                    arms=ARMS, context_features=["align_english_direct_asr", "multilingual_finnish_asr",
                                                 "benign_recon_mean", "benign_recon_n2", "benign_recon_n10"],
                    targets=tags, families=families, family_counts={f: families.count(f) for f in set(families)},
                    n_targets=len(tags), n_items=I, n_arms=A,
                    item_splits={s: int((splits == s).sum()) for s in ("train", "validation", "test")},
                    dropped_incomplete=dropped)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"targets": tags, "n_targets": len(tags), "n_arms": A,
                      "families": manifest["family_counts"], "dropped": dropped,
                      "item_splits": manifest["item_splits"]}, indent=2))

if __name__ == "__main__":
    main()
