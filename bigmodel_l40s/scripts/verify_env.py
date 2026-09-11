#!/usr/bin/env python3
"""Preflight for the L40S box. Checks GPUs, HF cache, judges, target weights, dataset inputs,
and that the tensor-parallel knob is wired. Run BEFORE run_l40s.sh:  python bigmodel_l40s/verify_env.py
Exit 0 = ready. It downloads nothing and touches no GPU heavily (just cuda counts)."""
import os, sys, json, importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # file is at bigmodel_l40s/scripts/verify_env.py
os.chdir(REPO)
ok = True
def check(name, cond, hint=""):
    global ok
    print(f"  [{'OK ' if cond else 'XX '}] {name}" + ("" if cond else f"   -> {hint}"))
    ok = ok and cond

print("== GPUs ==")
try:
    import torch
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"  visible cards: {n}  {names}")
    check("at least 2 GPUs (target + resident judge)", n >= 2, "single GPU can't hold target+judges; add a card")
    check("a card looks like L40S (informational)", any("L40S" in x for x in names) or True)
except Exception as e:
    check("torch + CUDA import", False, f"{e}")

print("== env ==")
hf = os.environ.get("HF_HOME")
check("HF_HOME set", bool(hf), "export HF_HOME=/your/hf_cache")
check("HF_HUB_OFFLINE=1", os.environ.get("HF_HUB_OFFLINE") == "1", "export HF_HUB_OFFLINE=1 (judges load offline)")

print("== judges cached (recon Qwen2.5-7B + safety Qwen3Guard-8B) ==")
hub = Path(hf or "") / "hub"
for repo in ("models--Qwen--Qwen2.5-7B-Instruct", "models--Qwen--Qwen3Guard-Gen-8B"):
    check(f"judge cached: {repo}", (hub / repo).exists(),
          f"python -c \"from huggingface_hub import snapshot_download as d; d('{repo[8:].replace('--','/')}')\"")

print("== target weights cached (from models.txt) ==")
mt = (REPO / "bigmodel_l40s" / "models.txt").read_text().splitlines()
targets = [l.split()[0] for l in mt if l.strip() and not l.strip().startswith("#")]
for hfid in targets:
    slug = "models--" + hfid.replace("/", "--")
    hit = (hub / slug).exists()
    check(f"target cached: {hfid}", hit,
          f"huggingface-cli download {hfid}  (or let vLLM pull it with HF_HUB_OFFLINE=0 once)")

print("== dataset inputs ==")
for f in ("private_artifacts/multijail_v1/harm_grid.jsonl",
          "private_artifacts/multijail_v1/benign_probe.jsonl",
          "private_artifacts/multijail_v1/resource_order.json",
          "private_artifacts/panel_v2/harm_grid.jsonl",
          "private_artifacts/panel_v2/benign_probe.jsonl",
          "results/lang_rank_20260905/resource_order.json"):
    check(f"input: {f}", (REPO / f).is_file() and (REPO / f).stat().st_size > 0,
          "run bash bigmodel_l40s/fetch_data.sh")

print("== code wiring ==")
src = (REPO / "scripts" / "online_live.py").read_text()
check("tensor_parallel_size wired in online_live.py", "tensor_parallel_size=int(tp)" in src)
cc = (REPO / "scripts" / "closed_compare.py").read_text()
check("--tensor-parallel CLI present", "--tensor-parallel" in cc)

print("\n" + ("ALL GOOD -> run: bash bigmodel_l40s/scripts/run_l40s.sh" if ok else "FIX the XX items above, then re-run."))
sys.exit(0 if ok else 1)
