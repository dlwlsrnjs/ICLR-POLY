#!/usr/bin/env python3
"""Re-test the n=10 + shuffle multilingual puzzle on GPT-4o under the CURRENT (working) prompt, to get a
RELIABLE verified ASR (the old_emph run had recon=0 parsing artifacts, raw 0.65). Reuses online_live's
build_arm_prompt/joint_on_config with a live OpenAI target + local judges. Reports raw/recon/verified so
we can see if the reconstruction gate holds at n=10 shuffle. Key from $OPENAI_API_KEY only; 0600."""
import sys, json, os, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_live import Judges, joint_on_config  # noqa: E402
from online_live_openai import OpenAITarget  # noqa: E402
SEED = 20260909
CONDS = ["frag5_shuffled_n10","frag8_shuffled_n10","frag3_shuffled_n10",
         "frag5_ordered_n10","frag5_shuffled_n8","frag5_shuffled_n6"]
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-4o"); ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n-items", type=int, default=30); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--outdir", default="results/gpt4o_n10shuffle_20260909")
    a = ap.parse_args()
    harm = [json.loads(l) for l in open(a.harm)]
    for r in harm: r.setdefault("original", r["questions"]["English"])
    rng = np.random.default_rng(SEED)
    rows = [harm[i] for i in rng.choice(len(harm), min(a.n_items,len(harm)), replace=False)]
    target = OpenAITarget(a.model, concurrency=a.concurrency); judges = Judges("cuda:0")
    t0=time.time(); out={"model":a.model,"n_items":len(rows),"conds":{}}
    for cond in CONDS:
        j, rc, u, _ = joint_on_config(target, judges, rows, cond)
        out["conds"][cond]={"raw_unsafe":round(u,3),"recon":round(rc,3),"verified":round(j,3)}
        print(json.dumps({"cond":cond, **out["conds"][cond]}), flush=True)
    out["seconds"]=round(time.time()-t0,1)
    od=Path(a.outdir); od.mkdir(parents=True,exist_ok=True); os.chmod(od,0o700)
    fd=os.open(od/"result.json",os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,"w") as h: json.dump(out,h,indent=2)
    print(json.dumps(out,indent=2))
if __name__=="__main__": raise SystemExit(main())
