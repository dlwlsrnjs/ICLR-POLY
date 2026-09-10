#!/usr/bin/env python3
"""After collection+chain finish and a GPU frees, run the LIVE agentic pipeline on a few
representative targets (strong/aligned, weak, mid) to demonstrate end-to-end, and compare
the live-found config's true gated ASR to the offline oracle for that target."""
import json, os, re, subprocess, time, glob
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
DL = json.loads((ROOT/"private_artifacts/selector_gpu_20260903/downloads.json").read_text())
DLMAP = {re.sub(r"[^a-z0-9]+","_",t["model_id"].split("/")[-1].lower()).strip("_"): t for t in DL["targets"]}
ORIG = {"qwen25_7b":"Qwen/Qwen2.5-7B-Instruct","qwen25_1_5b":"Qwen/Qwen2.5-1.5B-Instruct",
        "mistral7b":"mistralai/Mistral-7B-Instruct-v0.3","zephyr7b":"HuggingFaceH4/zephyr-7b-beta"}
# demo targets: aligned high-cap, aligned mid, weak-align
DEMO = ["qwen25_7b", "phi35", "qwen2_5_0_5b_instruct"]

def flags(tag, meta):
    e=[]
    if meta.get("model_type")=="qwen3" or "qwen3" in tag: e+=["--no-thinking"]
    if meta.get("max_position_embeddings",8192)<=4096: e+=["--max-model-len","4096"]
    return " ".join(e)

def ref(tag):
    if tag in DLMAP: return DLMAP[tag]["path"], DLMAP[tag]
    if tag in ORIG: return ORIG[tag], {}
    return None, None

def idle_gpu(thr=40000):
    o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    for l in o.strip().splitlines():
        i,f=[x.strip() for x in l.split(",")]
        if int(f)>thr: return int(i)
    return -1

def prior_path():
    for p in ["results/structured_policy_full_20260904/structured_prior.json",
              "results/structured_policy_20260904/structured_prior.json"]:
        if (ROOT/p).exists(): return p
    return None

def main():
    # wait for collection queue + chain to finish
    def alive(pid):
        try: os.kill(pid,0); return True
        except: return False
    # read pids best-effort
    for _ in range(600):
        q=subprocess.run(["pgrep","-f","run_full_factorial_queue"],capture_output=True,text=True).stdout.strip()
        if not q: break
        time.sleep(60)
    pr=prior_path()
    done={os.path.basename(os.path.dirname(p)) for p in glob.glob("private_artifacts/frag_factorial_20260903/*/harmful_summary.json")}
    ran=[]
    for tag in DEMO:
        if tag not in done: continue
        model, meta = ref(tag)
        if model is None: continue
        # wait for an idle GPU (need ~45GB for target + resident judges)
        gpu=-1
        for _ in range(120):
            gpu=idle_gpu()
            if gpu>=0: break
            time.sleep(60)
        if gpu<0: 
            print(json.dumps({"skip":tag,"reason":"no idle GPU"})); continue
        subprocess.run(["bash","scripts/run_agentic_pipeline.sh",tag,model,str(gpu),pr,flags(tag,meta)],cwd=ROOT)
        ran.append(tag)
    # compare live-found config to offline oracle
    report=[]
    for tag in ran:
        rp=ROOT/f"results/agentic_pipeline_20260904/{tag}.json"
        if not rp.exists(): continue
        res=json.loads(rp.read_text())
        report.append({"target":tag,"agentic_A_C":res.get("fingerprint"),"probe_queries":res.get("probe_queries"),
                       "best_config":res.get("best_config"),"best_gated":res.get("best_gated"),
                       "online_queries":res.get("queries"),"seconds":res.get("seconds")})
    (ROOT/"results/agentic_pipeline_20260904/report.json").write_text(json.dumps({"runs":report},indent=2)+"\n")
    (ROOT/"results/agentic_pipeline_20260904/DEMO_DONE.marker").write_text("done\n")
    print(json.dumps({"agentic_pipeline_runs":report},indent=2))

if __name__=="__main__":
    main()
