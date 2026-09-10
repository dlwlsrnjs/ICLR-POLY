#!/usr/bin/env python3
"""After extra families download and a GPU frees, collect them (panel_v2 benign+anchors
for the capability fingerprint, then the 32-arm fragment factorial for the joint arms),
then rebuild the extended dataset and refit the structured policies on the expanded set.
Continue-on-failure; opportunistic on idle GPUs (shared machine)."""
import json, os, re, subprocess, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
EXTRA = ROOT/"private_artifacts/extra_models_20260904"
def tagof(mid): return re.sub(r"[^a-z0-9]+","_",mid.split("/")[-1].lower()).strip("_")
FLAGS = {  # per-family loader flags (all need trust-remote-code except solar/llama)
    "solar":"", "minicpm":"--trust-remote-code", "baichuan":"--trust-remote-code",
    "exaone":"--trust-remote-code", "glm":"--trust-remote-code"}
def famof(tag):
    for k in ("solar","minicpm","baichuan","exaone","glm"):
        if k in tag: return k
    return "other"
def idle(thr=42000):
    o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    for l in o.strip().splitlines():
        i,f=[x.strip() for x in l.split(",")]
        if int(f)>thr: return int(i)
    return -1

def run(env, tag, model, gpu, extra):
    e=dict(os.environ, HF_HOME="/home/ubuntu/342/jinkwon/hf_cache", HF_HUB_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", GPU_MEM_UTIL="0.45")
    # panel_v2 (benign_recon + anchors)
    if not (ROOT/f"private_artifacts/panel_v2/{tag}/benign_recon.json").exists():
        subprocess.run(["bash","scripts/run_panel_v2.sh",tag,model,str(gpu),extra],cwd=ROOT,env=e,
                       stdout=open(ROOT/f"private_artifacts/panel_v2/new_{tag}.log","w"),stderr=subprocess.STDOUT)
    # 32-arm fragment factorial
    if not (ROOT/f"private_artifacts/frag_factorial_20260903/{tag}/harmful_summary.json").exists():
        subprocess.run(["bash","scripts/run_fragment_factorial.sh",tag,model,str(gpu),extra],cwd=ROOT,env=e,
                       stdout=open(ROOT/f"private_artifacts/frag_factorial_20260903/new_{tag}.log","w"),stderr=subprocess.STDOUT)

def main():
    # wait for downloads
    for _ in range(600):
        st=EXTRA/"download_status.json"
        if st.exists():
            d=json.loads(st.read_text())
            if len(d)>=5: break
        time.sleep(60)
    d=json.loads((EXTRA/"download_status.json").read_text())
    ok=[(mid,v) for mid,v in d.items() if v.get("ok")]
    done=[]
    for mid,v in ok:
        tag=tagof(mid); model=v["path"]; fam=famof(tag)
        gpu=-1
        for _ in range(240):
            gpu=idle()
            if gpu>=0: break
            time.sleep(60)
        if gpu<0: continue
        run({}, tag, model, gpu, FLAGS.get(fam,"--trust-remote-code"))
        if (ROOT/f"private_artifacts/frag_factorial_20260903/{tag}/harmful_summary.json").exists():
            done.append(tag)
    # rebuild + refit on expanded set
    VPY="/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python"
    subprocess.run([VPY,"scripts/build_context_selector_data_full.py","--outdir","results/context_selector_full_train_20260904"],cwd=ROOT)
    subprocess.run([VPY,"scripts/structured_policy_ext.py","--data","results/context_selector_full_train_20260904","--outdir","results/structured_policy_ext_20260904"],cwd=ROOT)
    subprocess.run([VPY,"scripts/structured_policy.py","--data","results/context_selector_full_train_20260904","--outdir","results/structured_policy_full_20260904"],cwd=ROOT)
    (ROOT/"results/NEW_MODELS_DONE.marker").write_text("done: "+",".join(done)+"\n")
    print(json.dumps({"collected_new": done}))

if __name__=="__main__":
    main()
