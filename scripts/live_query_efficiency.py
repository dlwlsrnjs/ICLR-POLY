#!/usr/bin/env python3
"""LIVE query-efficiency comparison for the GP-BAI arm selector (reviewer score-changer).

Removes the two caveats of the offline query-efficiency table (tab_sel_queryeff), which
replays search strategies over a *stored* joint matrix with a *synthetic* probe-noise model.
Here every probe is a REAL online query against a live target: generate the chosen config on
a calibration batch, gate by the reconstruction judge, then score answer-safety with the guard
judge (online_live.joint_on_config). All strategies share ONE resident target + ONE resident
pair of judges + the SAME calibration pool, so the only difference is how queries are spent.

Because generation is temperature 0 (deterministic), a probed arm's live gated J is fixed for
a given pool, so each unique arm is generated+judged ONCE and cached; strategies that revisit an
arm reuse the live cell. Net cost is at most |ARMS| real config-generations + the fingerprint.
This measures the arm matrix LIVE for a real target (no stored aggregates, no synthetic noise)
and then runs the identical strategy comparison as the paper.

Strategies (all over the 32-arm frag x order x n menu):
  ours (benign warm start)  : GP mean = domain-structured J_hat from the BENIGN fingerprint
                              (no harmful data); warm = argmax; then prior-warm GP-UCB.
  cross-target prior         : GP mean = leave-this-target-out pooled harmful arm prior.
  uninformed GP              : GP mean = flat; pure GP-UCB exploration over the kernel.
  random search              : random unobserved arm each step (seeded).
  independent-arm UCB        : forced exploration (seeded fixed order) then per-arm UCB, no kernel.

Queries-to-threshold uses the paper's definition: the threshold is the running-max gated J that
"ours" reaches within its first three queries; each strategy's score is the smallest query index
at which its running-max gated J meets that threshold (dash if not within the budget).

Issues real harmful queries to the target -> authorized red-team evaluation only. Outputs 0600.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_live import LiveTarget, Judges, joint_on_config, fingerprint  # noqa: E402
from online_adapt import gp_recommend  # noqa: E402
from structured_policy import predict_best  # noqa: E402
from build_context_selector_data_full import ARMS  # noqa: E402

SEED = 20260909


def run_strategy(name, next_arm, budget, probe, warm=None):
    """Generic online episode. next_arm(prior_ctx, observed)->arm index. probe(arm)->gated J (cached).
    Returns trajectory list of dicts and the running-max gated per query."""
    observed = {}
    traj = []
    for step in range(budget):
        arm = warm if (step == 0 and warm is not None) else next_arm(observed)
        if arm in observed:  # never re-probe (deterministic); pick any unseen fallback
            unseen = [a for a in range(len(ARMS)) if a not in observed]
            if not unseen:
                break
            arm = unseen[0]
        j = probe(arm)
        observed[arm] = j
        traj.append(dict(query=step + 1, arm=ARMS[arm], gated=round(float(j), 4)))
    run_max, best = [], -1.0
    for t in traj:
        best = max(best, t["gated"]); run_max.append(best)
    return traj, run_max


def q_to_threshold(run_max, thr):
    for i, v in enumerate(run_max):
        if v >= thr - 1e-9:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prior-data", default="results/context_selector_full_train_20260904")
    ap.add_argument("--structured-prior", default="results/structured_policy_full_20260904/structured_prior.json")
    ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--benign", default="private_artifacts/panel_v2/benign_probe.jsonl")
    ap.add_argument("--exclude-target-tag", default="",
                    help="manifest tag of this target to leave OUT of the pooled cross-target prior (LOTO)")
    ap.add_argument("--fp-harm", type=int, default=20)
    ap.add_argument("--fp-benign", type=int, default=15)
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--judge-device", default="cuda:0")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", default="results/live_query_efficiency_20260909")
    a = ap.parse_args()

    harm = [json.loads(l) for l in open(a.harm)]
    for r in harm:
        r.setdefault("original", r["questions"]["English"])
    benign = [json.loads(l) for l in open(a.benign)]
    for r in benign:
        r.setdefault("original", r["questions"]["English"])
    rng = np.random.default_rng(SEED)
    fp_harm = [harm[i] for i in rng.choice(len(harm), a.fp_harm, replace=False)]
    fp_benign = benign[:a.fp_benign]
    pool = [harm[i] for i in rng.choice(len(harm), a.calib, replace=False)]

    # offline warm-start material
    blob = np.load(Path(a.prior_data) / "context_values.npz")
    man = json.loads((Path(a.prior_data) / "manifest.json").read_text())
    off_J = blob["J"]; off_splits = blob["splits"]; arm_feats = np.asarray(blob["features"], np.float32)
    tags = man["targets"]
    keep = np.ones(off_J.shape[0], bool)
    if a.exclude_target_tag and a.exclude_target_tag in tags:
        keep[tags.index(a.exclude_target_tag)] = False   # leave-this-target-out pooled prior
    cross_prior = np.nanmean(off_J[keep][:, off_splits == "train", :], (0, 1))
    flat = np.full(len(ARMS), float(np.nanmean(cross_prior)))

    print(json.dumps({"stage": "loading", "target": a.tag,
                      "loto_excluded": a.exclude_target_tag or None}), flush=True)
    target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
    judges = Judges(a.judge_device)

    t0 = time.time()
    fp = fingerprint(target, judges, fp_harm, fp_benign)
    pj = json.loads(Path(a.structured_prior).read_text())
    _, benign_prior = predict_best(pj["theta"], list(ARMS), A=float(fp[pj["a_index"]]),
                                   C_raw=float(fp[pj["c_index"]]), c_scale=tuple(pj["c_scale"]))
    benign_prior = np.asarray(benign_prior, float)
    print(json.dumps({"stage": "fingerprint", "context": [round(float(x), 3) for x in fp],
                      "benign_warm_arm": ARMS[int(benign_prior.argmax())],
                      "cross_warm_arm": ARMS[int(cross_prior.argmax())]}), flush=True)

    cache = {}
    def probe(arm):
        if arm not in cache:
            j, rc, u, _ = joint_on_config(target, judges, pool, ARMS[arm])
            cache[arm] = float(j)
            print(json.dumps({"stage": "probe", "arm": ARMS[arm], "gated": round(j, 3),
                              "recon": round(rc, 3), "raw_asr": round(u, 3),
                              "unique_arms": len(cache)}), flush=True)
        return cache[arm]

    rng_r = np.random.default_rng(SEED)          # random search
    rng_i = np.random.default_rng(12345)         # independent-arm UCB forced-exploration order
    forced = list(rng_i.permutation(len(ARMS)))

    def mk_gp(prior):
        return lambda observed: gp_recommend(prior, observed, arm_feats)
    def rand_next(observed):
        unseen = [x for x in range(len(ARMS)) if x not in observed]
        return int(rng_r.choice(unseen))
    def indep_next(observed):
        for x in forced:
            if x not in observed:
                return int(x)                     # budget < |ARMS|: pure forced exploration
        means = {k: v for k, v in observed.items()}
        return max(means, key=means.get)

    strategies = {
        "ours (benign warm start)": (mk_gp(benign_prior), int(benign_prior.argmax())),
        "cross-target prior":       (mk_gp(cross_prior), int(cross_prior.argmax())),
        "uninformed GP search":     (mk_gp(flat), int(flat.argmax())),
        "random search":            (rand_next, None),
        "independent-arm UCB":      (indep_next, None),
    }
    results = {}
    for name, (nx, warm) in strategies.items():
        traj, run_max = run_strategy(name, nx, a.budget, probe, warm=warm)
        results[name] = dict(trajectory=traj, run_max=[round(x, 4) for x in run_max])

    thr = max(results["ours (benign warm start)"]["run_max"][:3])   # ours@3, per paper definition
    table = {name: q_to_threshold(r["run_max"], thr) for name, r in results.items()}

    summary = dict(target=a.tag, model=a.target, loto_excluded=a.exclude_target_tag or None,
                   fingerprint=[round(float(x), 3) for x in fp], budget=a.budget, calib=a.calib,
                   n_arms=len(ARMS), unique_arms_probed=len(cache), seconds=round(time.time() - t0, 1),
                   threshold_ours_at_3=round(float(thr), 4),
                   queries_to_threshold=table, per_strategy=results)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(summary, h, indent=2)
    print(json.dumps(dict(target=a.tag, threshold_ours_at_3=summary["threshold_ours_at_3"],
                          unique_arms_probed=len(cache), seconds=summary["seconds"],
                          queries_to_threshold=table), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
