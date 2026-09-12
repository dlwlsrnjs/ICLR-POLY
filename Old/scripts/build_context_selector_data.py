#!/usr/bin/env python3
"""Build the CONTEXTUAL selector dataset from all completed panel_v2 targets.

Per target it emits:
  J[T, I, A]   joint reward (R AND U) per harmful item, for the A=10 interleave ARMS
               (english_direct and translated_Finnish are NOT arms; they are cheap probes)
  context[T, K] behavioral fingerprint measured from black-box outputs, NOT from the arm
               joint labels: alignment (english_direct raw ASR), multilingual gap
               (translated_Finnish raw ASR), reconstruction capability (benign FLORES recon
               mean / low-load / high-load). Anchors measured on the train-item split only.
  features[A, F] per-arm config features (languages present, load n, arrangement).
  families[T], item split.

This is the data for a policy that maps (target fingerprint, probe history) -> next config.
No new generation or judging here.
"""
from __future__ import annotations
import json, os, hashlib, argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen_interleaving_probe import ALL_LANGUAGES, select_languages  # noqa: E402

PANEL = Path("private_artifacts/panel_v2")
ARMS = [f"interleave_{a}_n{n}" for a in ("ordered", "shuffled") for n in (2, 4, 6, 8, 10)]

def family_of(tag):
    t = tag.lower()
    for key, fam in [("qwen3", "qwen"), ("qwen2_5", "qwen"), ("qwen25", "qwen"), ("qwen", "qwen"),
                     ("phi", "phi"), ("falcon", "falcon"), ("mistral", "mistral_zephyr"),
                     ("zephyr", "mistral_zephyr"), ("olmo", "olmo"), ("internlm", "internlm"),
                     ("smollm", "smollm"), ("granite", "granite"), ("stablelm", "stablelm"),
                     ("yi", "yi"), ("deepseek", "deepseek")]:
        if key in t:
            return fam
    return "other"

def arm_feature(cond):
    a, nk = cond.replace("interleave_", "").split("_n")
    n = int(nk); langs = select_languages(n)
    return ([float(x in langs) for x in ALL_LANGUAGES] + [n / 10.0]
            + [float(a == "ordered"), float(a == "shuffled")])

def load(m):
    rp = PANEL/m/"harm_recon/restricted_reconstruction_audit.jsonl"
    gp = PANEL/m/"harm_guard/restricted_qwen3guard_audit.jsonl"
    bn = PANEL/m/"benign_recon.json"
    if not (os.access(rp, os.R_OK) and os.access(gp, os.R_OK) and bn.exists()):
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
    ap.add_argument("--outdir", default="results/context_selector_train_20260903")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    tags = sorted(p.name for p in PANEL.iterdir() if p.is_dir() and (p/"harmful_summary.json").exists())
    data = {}
    for m in tags:
        d = load(m)
        if d is not None:
            data[m] = d
    tags = sorted(data)
    items = sorted({k[0] for k in data[tags[0]][0] if k[1] == "english_direct"})
    iidx = {it: i for i, it in enumerate(items)}
    splits = np.array([split_of(it) for it in items])
    train_items = set(np.array(items)[splits == "train"])

    T, I, A = len(tags), len(items), len(ARMS)
    J = np.full((T, I, A), np.nan, np.float32)
    for ti, m in enumerate(tags):
        R, U, _ = data[m]
        for ci, cond in enumerate(ARMS):
            for it in items:
                if (it, cond) in R and (it, cond) in U:
                    J[ti, iidx[it], ci] = 1.0 if (R[(it, cond)] and U[(it, cond)]) else 0.0

    def raw_asr_train(m, cond):
        _, U, _ = data[m]
        v = [U[(it, cond)] for it in train_items if (it, cond) in U]
        return float(np.mean(v)) if v else np.nan

    context = np.zeros((T, 5), np.float32)
    for ti, m in enumerate(tags):
        _, _, bn = data[m]
        cap = [bn.get(c, np.nan) for c in ARMS]
        context[ti] = [raw_asr_train(m, "english_direct"),
                       raw_asr_train(m, "translated_direct_Finnish"),
                       float(np.nanmean(cap)),
                       float(np.nanmean([bn.get("interleave_ordered_n2", np.nan), bn.get("interleave_shuffled_n2", np.nan)])),
                       float(np.nanmean([bn.get("interleave_ordered_n10", np.nan), bn.get("interleave_shuffled_n10", np.nan)]))]

    # arm coverage per target: refuse targets with any missing arm cell (do not impute)
    keep = [ti for ti in range(T) if np.isfinite(J[ti]).all() and np.isfinite(context[ti]).all()]
    dropped = [tags[ti] for ti in range(T) if ti not in keep]
    J, context = J[keep], context[keep]
    tags = [tags[ti] for ti in keep]
    families = [family_of(t) for t in tags]

    np.savez(out/"context_values.npz", J=J, context=context,
             features=np.array([arm_feature(c) for c in ARMS], np.float32), splits=splits)
    manifest = dict(scope="Contextual joint-ASR selector data across panel_v2 targets.",
                    reward="J = reconstruction AND guard_strict_unsafe(answer); per-item 0/1",
                    arms=ARMS, context_features=["align_english_direct_asr", "multilingual_finnish_asr",
                                                 "benign_recon_mean", "benign_recon_n2", "benign_recon_n10"],
                    targets=tags, families=families, family_counts={f: families.count(f) for f in set(families)},
                    n_targets=len(tags), n_items=I, n_arms=A,
                    item_splits={s: int((splits == s).sum()) for s in ("train", "validation", "test")},
                    dropped_incomplete=dropped)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"targets": tags, "n_targets": len(tags),
                      "families": manifest["family_counts"], "dropped": dropped,
                      "item_splits": manifest["item_splits"]}, indent=2))

if __name__ == "__main__":
    main()
