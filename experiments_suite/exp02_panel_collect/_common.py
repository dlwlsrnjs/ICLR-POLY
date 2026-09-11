#!/usr/bin/env python3
"""Panel full-matrix collector shared body. Dataset drivers (collect_mj.py / collect_lg.py) set the
collection + tags and call run(). For each panel model: phase-1 benign probe (harmless) then a
FULL-MATRIX harmful eval of every arm (so per-model oracle / best-arm / heterogeneity are computable).
The full-matrix pass is the heavy GPU job; probe alone is harmless."""
import sys, argparse, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import engine  # noqa: E402


def run(collection, root, tag_suffix, default_n_items):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["probe", "matrix", "both", "list"])
    ap.add_argument("--set", choices=["held_in", "held_out", "all"], default="all")
    ap.add_argument("--models", default="", help="comma HF ids to override the panel")
    ap.add_argument("--judge-device", default="cuda:0")
    ap.add_argument("--util", type=float, default=0.45)
    ap.add_argument("--tensor-parallel", type=int, default=1,
                    help="vLLM tensor_parallel_size for the target (>1 to shard a big model across GPUs)")
    ap.add_argument("--fp-benign", type=int, default=24)
    ap.add_argument("--n-items", type=int, default=default_n_items)
    ap.add_argument("--judge-batch-size", type=int, default=16,
                    help="resident HF judge micro-batch; lower this if judge GPU memory is tight")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--arm-space", default="C", choices=["full", "C"],
                    help="C = 160 medium-willingness space (+baselines); full = 288 grid arms")
    ap.add_argument("--root", default="", help="override results root (e.g. the L40S box writes to its "
                    "own bigmodel_l40s/results and merges back). Default: this driver's results/ dir.")
    a = ap.parse_args()
    if a.root:
        root = os.path.abspath(a.root)
        os.makedirs(root, exist_ok=True)
    if a.judge_batch_size < 1:
        raise SystemExit("--judge-batch-size must be >= 1")
    os.environ["POLY_JUDGE_BATCH_SIZE"] = str(a.judge_batch_size)
    if a.models:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
    else:
        models = {"held_in": engine.PANEL_HELD_IN, "held_out": engine.PANEL_HELD_OUT,
                  "all": engine.PANEL_HELD_IN + engine.PANEL_HELD_OUT}[a.set]
    if a.phase == "list":
        for m in models:
            print(f"{engine.PANEL_TAGS.get(m, m)}\t{m}")
        return 0
    for m in models:
        tag = engine.PANEL_TAGS.get(m, m.split('/')[-1]) + tag_suffix
        print(f"=== {tag} ({m}) ===", flush=True)
        if a.phase in ("probe", "both"):
            engine.run_probe(collection, root, model=m, backend="vllm", tag=tag,
                             fp_benign=a.fp_benign, judge_device=a.judge_device, util=a.util,
                             tensor_parallel=a.tensor_parallel, force=a.force)
        if a.phase in ("matrix", "both"):
            engine.run_full_matrix(collection, root, tag, model=m, backend="vllm",
                                   n_items=a.n_items, judge_device=a.judge_device, util=a.util,
                                   tensor_parallel=a.tensor_parallel, force=a.force, arm_space=a.arm_space)
    print("PANEL_COLLECT_DONE", time.strftime("%H:%M:%S"))
    return 0
