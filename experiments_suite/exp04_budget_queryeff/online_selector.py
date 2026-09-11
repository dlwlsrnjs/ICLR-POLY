#!/usr/bin/env python3
"""EXP04f ONLINE selector live-run demo: benign probe -> DROP saturated axis -> spend a small harmful
budget with structured GP-UCB + pi-BO, logging every step. Demonstrates that axis-dropping is applied
DURING the run (decided from the harmless probe, before/while spending harmful queries), not post-hoc.

Modes:
  --mode replay   y(arm) read from the stored full matrix (fast, validates the loop; no GPU)
  --mode live     y(arm) evaluated LIVE (vLLM target + local judges) on --n-items harmful items
Logs per step: arm, verified, running best, and the axis-drop decision. Also runs a no-drop control
so the effect is auditable in the same trajectory file.

Usage (replay):  python online_selector.py --root <exp02 results> --tag qwen25_7b_mj --mode replay
Usage (live):    CUDA_VISIBLE_DEVICES=1,0 python online_selector.py --collection MultiJail \
                   --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b_mj --mode live --budget 8 --n-items 64"""
import argparse, json, glob, time
from pathlib import Path
import numpy as np
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import search_sota as S


def load_matrix(root, tag):
    ver = {}
    for f in glob.glob(str(Path(root) / "attack" / f"{tag}__*.json")):
        d = json.loads(Path(f).read_text())
        if d["method"].startswith("ours[") or d["method"] in (
                "plain", "translated", "cipher_base64", "aim", "deepinception", "pap"):
            continue
        ver[d["method"]] = d["verified"]
    return ver


def online_run(arms, X_c, X_w, y_fn, prior, drop_axis, budget, beta=2.0, gamma=3.0, log=None):
    """Sequential structured GP-UCB + pi-BO. If drop_axis given, that feature block is zeroed so the
    kernel/prior ignore it and exploration concentrates on the informative axis."""
    n = len(arms)
    Xc = X_c.copy(); Xw = X_w.copy()
    if drop_axis == "willingness":
        Xw[:] = 0.0
    elif drop_axis == "comprehension":
        Xc[:] = 0.0
    K = 0.5 * S.rbf(Xc, Xc, np.array([0.6, 0.6, 0.9])) + 0.5 * S.rbf(Xw, Xw, np.array([0.4, 0.4, 0.4, 0.5, 0.5]))
    pv = np.array([prior.get(a, 0.5) for a in arms]); pv = pv / (pv.max() + 1e-9)
    m = np.full(n, 0.5); rng = np.random.default_rng(0)
    obs, ys = [], []; best = 0.0; traj = []
    for t in range(budget):
        if not obs:
            mu = m.copy(); sd = np.ones(n)
        else:
            I = np.array(obs); Kinv = np.linalg.inv(K[np.ix_(I, I)] + 0.02 * np.eye(len(I)))
            Ks = K[:, I]; mu = m + Ks @ Kinv @ (np.array(ys) - m[I])
            sd = np.sqrt(np.clip(1 - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks), 1e-6, None))
        acq = mu + beta * sd + (gamma / (t + 1)) * np.log(pv + 1e-6)
        for j in obs:
            acq[j] = -1e9
        pick = int(rng.integers(n)) if not obs else int(np.argmax(acq))
        y = y_fn(arms[pick]); obs.append(pick); ys.append(y); best = max(best, y)
        step = dict(step=t + 1, arm=arms[pick], verified=round(float(y), 3), running_best=round(best, 3))
        traj.append(step)
        if log is not None:
            print(json.dumps({"stage": "online", "drop": drop_axis or "none", **step}), flush=True)
    return dict(drop_axis=drop_axis or "none", budget=budget, best=round(best, 3),
                selected=arms[int(np.argmax([s["verified"] for s in traj]))], trajectory=traj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="experiments_suite/exp02_panel_collect/results")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--collection"); ap.add_argument("--model"); ap.add_argument("--n-items", type=int, default=64)
    ap.add_argument("--judge-device", default="cuda:1"); ap.add_argument("--util", type=float, default=0.3)
    a = ap.parse_args()
    bj = json.loads((Path(a.root) / "benign" / f"{a.tag}.json").read_text())
    ver = load_matrix(a.root, a.tag)   # ORACLE reference only (not used as observations in live mode)
    if a.mode == "live":
        # COLD run: arm list comes from the configuration space, NOT from any pre-collected matrix.
        sys.path.insert(0, str(HERE.parents[1] / "scripts"))
        from closed_compare import build_arms as _ba, resolve_inputs as _ri
        class _NS: pass
        _ns = _NS(); _ns.collection = a.collection; _ns.order = "AUTO"; _ns.benign = "AUTO"; _ns.harm = "AUTO"; _ri(_ns)
        arms = [x[0] for x in _ba(json.loads(Path(_ns.order).read_text())["order"])]
    else:
        arms = list(ver) if ver else None
        if arms is None:
            raise SystemExit("replay needs the stored matrix")
    prior, axinfo = S.saturation_aware_prior(bj, arms)
    drop = ("willingness" if axinfo["comp_informative"] and not axinfo["will_informative"]
            else "comprehension" if axinfo["will_informative"] and not axinfo["comp_informative"] else None)
    X_c = np.array([S.feats(x)[0] for x in arms]); X_w = np.array([S.feats(x)[1] for x in arms])
    print(json.dumps({"stage": "decision", "tag": a.tag, "axis_info": axinfo, "dropped": drop or "none"}), flush=True)

    if a.mode == "replay":
        yf = lambda arm: ver[arm]
    else:
        import os
        from closed_compare import build_arms, resolve_inputs, _rows, make_target, section
        from online_live import Judges
        class NS: pass
        ns = NS(); ns.collection = a.collection; ns.order = "AUTO"; ns.benign = "AUTO"; ns.harm = "AUTO"
        resolve_inputs(ns)
        harm = _rows(ns.harm, a.n_items)
        tgt = make_target_live(a)
        judges = Judges(a.judge_device, safety=True)
        armmap = {x[0]: x for x in build_arms(json.loads(Path(ns.order).read_text())["order"])}
        def yf(arm):
            _, frame, gated, hp, build = armmap[arm]
            outs = tgt.generate([build(r) for r in harm]); O = [r["original"] for r in harm]
            U = [bool(x) for x in judges.answer_unsafe(O, [section(o, "ANSWER") or o for o in outs])]
            if gated:
                R = [bool(x) for x in judges.recon_equivalent(O, [section(o, "RECONSTRUCTED", "ANSWER") for o in outs])]
            else:
                R = [True] * len(outs)
            return float(np.mean([1.0 if (r and u) else 0.0 for r, u in zip(R, U)]))

    res_drop = online_run(arms, X_c, X_w, yf, prior, drop, a.budget, log=True)
    res_nodrop = online_run(arms, X_c, X_w, yf, bj.get("prior", prior), None, a.budget)
    out = dict(tag=a.tag, mode=a.mode, dropped_axis=drop or "none", axis_info=axinfo,
               online_drop=res_drop, online_nodrop=res_nodrop,
               effect_best=round(res_drop["best"] - res_nodrop["best"], 3), ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    p = Path(a.root) / f"online_selector_{a.tag}_{a.mode}.json"; p.write_text(json.dumps(out, indent=2))
    print(json.dumps({"tag": a.tag, "mode": a.mode, "dropped": drop or "none",
                      "best_drop": res_drop["best"], "best_nodrop": res_nodrop["best"],
                      "effect": out["effect_best"]}, indent=2))
    print("wrote", p)
    return 0


def make_target_live(a):
    from closed_compare import make_target
    class NS: pass
    ns = NS(); ns.backend = "vllm"; ns.model = a.model; ns.util = a.util; ns.max_model_len = 4096
    ns.no_thinking = False; ns.trust_remote_code = False; ns.tokenizer_mode = "auto"
    ns.concurrency = 8; ns.max_tokens = 320
    return make_target(ns)


if __name__ == "__main__":
    raise SystemExit(main())
