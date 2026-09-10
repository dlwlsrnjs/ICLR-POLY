#!/usr/bin/env python3
"""Download additional distinct-lineage instruction models to broaden family coverage."""
import json, time
from pathlib import Path
from huggingface_hub import snapshot_download
OUT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw/private_artifacts/extra_models_20260904")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [
    ("upstage/SOLAR-10.7B-Instruct-v1.0", "solar"),
    ("openbmb/MiniCPM3-4B", "minicpm"),
    ("baichuan-inc/Baichuan2-7B-Chat", "baichuan"),
    ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "exaone"),
    ("THUDM/glm-4-9b-chat-hf", "glm"),
]
status = {}
for mid, fam in MODELS:
    try:
        t0 = time.time()
        p = snapshot_download(mid, local_dir=str(OUT / mid.split("/")[-1]),
                              allow_patterns=["*.json","*.safetensors","*.model","*.txt","tokenizer*","*.py"])
        status[mid] = {"family": fam, "path": p, "ok": True, "seconds": round(time.time()-t0,1)}
    except Exception as e:
        status[mid] = {"family": fam, "ok": False, "error": str(e)[:200]}
    (OUT / "download_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({mid: status[mid].get("ok")}), flush=True)
print(json.dumps({"done": sum(1 for v in status.values() if v.get("ok")), "of": len(MODELS)}))
