#!/usr/bin/env python3
"""Decompose the puzzle's marginal value on a CLOSED model: score every arm on the SAME pool so we can
compare pure-comprehension (puzzle) vs pure-willingness (AIM/DeepInception/PAP) vs their COMBINATIONS
(puzzle+persona, puzzle+fiction, role-split+persona). Answers: does the puzzle still help on aligned
closed models? Reuses two_axis_selector arms/scoring. OpenAI target + local judges; key from env; 0600."""
import sys, json, os, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_axis_selector import make_arms, score_arm  # noqa: E402
from online_live import Judges  # noqa: E402
SEED=20260909
def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend",default="openai",choices=["openai","gemini","anthropic","vllm"])
    ap.add_argument("--model",default="gpt-4o"); ap.add_argument("--tag",default="gpt4o")
    ap.add_argument("--order",default="results/lang_rank_20260905/resource_order.json")
    ap.add_argument("--harm",default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n-items",type=int,default=30); ap.add_argument("--concurrency",type=int,default=8)
    ap.add_argument("--util",type=float,default=0.25); ap.add_argument("--max-model-len",type=int,default=4096)
    ap.add_argument("--no-thinking",action="store_true"); ap.add_argument("--trust-remote-code",action="store_true")
    ap.add_argument("--tokenizer-mode",default="auto"); ap.add_argument("--outdir",default="results/axis_decomp_20260909")
    a=ap.parse_args()
    order=json.loads(Path(a.order).read_text())["order"]; arms=make_arms(order)
    harm=[json.loads(l) for l in open(a.harm)]
    for r in harm: r.setdefault("original",r["questions"]["English"])
    rng=np.random.default_rng(SEED); pool=[harm[i] for i in rng.choice(len(harm),min(a.n_items,len(harm)),replace=False)]
    if a.backend in ("openai","gemini","anthropic"):
        from closed_targets import make_target; target=make_target(a.backend,a.model,concurrency=a.concurrency)
    else:
        from online_live import LiveTarget; target=LiveTarget(a.model,a.util,a.max_model_len,a.no_thinking,a.trust_remote_code,a.tokenizer_mode)
    judges=Judges("cuda:0"); t0=time.time(); out={"model":a.model,"n_items":len(pool),"arms":{}}
    for name,ax,gated,build,fe in arms:
        j,rc,u=score_arm(target,judges,pool,build,gated)
        out["arms"][name]={"axis":ax,"gated_rule":("recon-gated" if gated else "ungated"),
                           "verified":round(j,3),"recon":round(rc,3),"raw":round(u,3)}
        print(json.dumps({"arm":name,**out["arms"][name]}),flush=True)
    out["seconds"]=round(time.time()-t0,1)
    od=Path(a.outdir); od.mkdir(parents=True,exist_ok=True); os.chmod(od,0o700)
    fd=os.open(od/f"{a.tag}.json",os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,"w") as h: json.dump(out,h,indent=2)
    print("SAVED", a.tag)
if __name__=="__main__": raise SystemExit(main())
