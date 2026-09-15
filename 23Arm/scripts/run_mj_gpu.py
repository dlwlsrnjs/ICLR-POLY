#!/usr/bin/env python3
"""Run the MultiJail-DATASET experiments (method baselines + MultiJail-combination + our combo)
on a given GPU, using directory-lock claims so two instances (GPU0 + GPU1) can share the model
list without duplication. GPU0 instance waits until GPU0 has >=NEED free (other users' servers).
Usage: run_mj_gpu.py --gpu {0,1}"""
import argparse, glob, subprocess, os, time, json, re
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
HARM = "private_artifacts/multijail_v1/harm_grid.jsonl"
ORDER = "private_artifacts/multijail_v1/resource_order.json"
CLAIM = ROOT / "results/mj_claims"; CLAIM.mkdir(parents=True, exist_ok=True)
NITEMS = "64"
HUB = "/home/ubuntu/342/jinkwon/hf_cache/hub"
DL = {re.sub(r'[^a-z0-9]+','_',t['model_id'].split('/')[-1].lower()).strip('_'): t
      for t in json.load(open(ROOT/"private_artifacts/selector_gpu_20260903/downloads.json"))["targets"]}
PANEL = [
    ("qwen25_7b","Qwen/Qwen2.5-7B-Instruct","models--Qwen--Qwen2.5-7B-Instruct","","0.35",None),
    ("qwen25_14b","Qwen/Qwen2.5-14B-Instruct","models--Qwen--Qwen2.5-14B-Instruct","","0.45",None),
    ("llama31_8b_it","meta-llama/Llama-3.1-8B-Instruct","models--meta-llama--Llama-3.1-8B-Instruct","","0.35",None),
    ("gemma2_9b_it","google/gemma-2-9b-it","models--google--gemma-2-9b-it","","0.40","TRITON_ATTN"),
    ("qwen25_3b","Qwen/Qwen2.5-3B-Instruct","models--Qwen--Qwen2.5-3B-Instruct","","0.35",None),
    ("llama32_3b_it","meta-llama/Llama-3.2-3B-Instruct","models--meta-llama--Llama-3.2-3B-Instruct","","0.35",None),
    ("gemma2_2b_it","google/gemma-2-2b-it","models--google--gemma-2-2b-it","","0.35","TRITON_ATTN"),
]
EVALS = [
    ("scripts/method_baselines_eval.py", "results/mj_method_20260906", ["--outdir","results/mj_method_20260906"]),
    ("scripts/multijail_eval.py",        "results/mj_multijail_20260906", ["--outdir","results/mj_multijail_20260906"]),
    ("scripts/combo_eval.py",            "results/mj_combo_20260906", ["--outdir","results/mj_combo_20260906","--order",ORDER,"--n","4"]),
    ("scripts/sequential_add.py",        "results/mj_sequential_20260906", ["--outdir","results/mj_sequential_20260906","--order",ORDER]),
    ("scripts/disorder_sweep.py",        "results/mj_disorder_20260906", ["--outdir","results/mj_disorder_20260906","--order",ORDER,"--n","4"]),
]
def gpu_free(i):
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits","-i",str(i)],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])
def claim(tag):
    try: os.mkdir(CLAIM/tag); return True
    except FileExistsError: return False
def done(tag):
    return all((ROOT/e[1]/f"{tag}.json").exists() for e in EVALS)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gpu",type=int,required=True); a=ap.parse_args()
    NEED=58000
    dep=ROOT/"results/multijail_20260906/MULTIJAIL_DONE.marker"
    for _ in range(400):
        if dep.exists(): break
        time.sleep(60)
    print(f"[{time.strftime('%H:%M:%S')}] lingua chain done; GPU{a.gpu} starting MJ dataset",flush=True)
    for tag,ref,repo,extra,util,be in PANEL:
        if done(tag): print(f"already done {tag}",flush=True); continue
        if len(glob.glob(f"{HUB}/{repo}/snapshots/*/*.safetensors"))==0: continue
        if not claim(tag): print(f"claimed by other: {tag}",flush=True); continue
        # wait for this GPU to be free enough
        for _ in range(240):
            if gpu_free(a.gpu)>=NEED: break
            time.sleep(30)
        else:
            os.rmdir(CLAIM/tag); print(f"GPU{a.gpu} never free for {tag}, releasing",flush=True); continue
        env=dict(os.environ,HF_HOME="/home/ubuntu/342/jinkwon/hf_cache",HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1",CUDA_VISIBLE_DEVICES=str(a.gpu),
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        if be: env["VLLM_ATTENTION_BACKEND"]=be
        else: env["VLLM_USE_FLASHINFER_SAMPLER"]="0"
        for script,outdir,eargs in EVALS:
            if (ROOT/outdir/f"{tag}.json").exists(): continue
            (ROOT/outdir).mkdir(parents=True,exist_ok=True)
            cmd=["/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python",script,"--target",ref,"--tag",tag,
                 "--harm",HARM,"--n-items",NITEMS,"--util",util]+eargs+(extra.split() if extra else [])
            print(f"[{time.strftime('%H:%M:%S')}] GPU{a.gpu} {tag} :: {os.path.basename(script)}",flush=True)
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=open(ROOT/outdir/f"log_{tag}.txt","w"),stderr=subprocess.STDOUT)
        print(f"  -> {tag} {'OK' if done(tag) else 'PARTIAL'}",flush=True)
    print(f"GPU{a.gpu} runner finished",flush=True)
if __name__=="__main__": main()
