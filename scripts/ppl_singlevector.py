#!/usr/bin/env python3
"""Prompt perplexity of the single-vector published attacks (AIM, DeepInception, PAP) under the same
reference LM (Qwen2.5-7B) used by ppl_evasion.py, so the main comparison table can carry a PPL column.
Offline scoring, no generation."""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import method_baselines_eval as MB

SRC = "private_artifacts/panel_v2/granite_3_3_8b_instruct/harm_gen/restricted_target_outputs.jsonl"
REF = "Qwen/Qwen2.5-7B-Instruct"
N = 120


@torch.no_grad()
def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
    if ids.shape[1] < 2:
        return float("nan")
    return float(math.exp(min(model(ids, labels=ids).loss.item(), 20)))


def main():
    origs = []
    for l in open(SRC):
        d = json.loads(l)
        if d["condition"] == "english_direct":
            origs.append(d["original"])
        if len(origs) >= N:
            break
    rows = [{"questions": {"English": o}} for o in origs]
    tok = AutoTokenizer.from_pretrained(REF, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(REF, dtype=torch.bfloat16, device_map="cuda:0",
                                                 local_files_only=True).eval()
    res = {}
    for method, label in (("aim", "AIM"), ("deepinception", "DeepInception"), ("pap", "PAP")):
        prompts = MB.build_prompts(rows, method, "Finnish")
        vals = [ppl(model, tok, p) for p in prompts]
        vals = [v for v in vals if v == v]
        res[label] = round(float(np.median(vals)), 1)
        print(f"{label:16s} median prompt PPL {res[label]:8.1f}  (n={len(vals)})")
    Path("results/ppl_singlevector_20260908.json").write_text(json.dumps(res, indent=2))
    print("PPLSV_DONE")


if __name__ == "__main__":
    raise SystemExit(main())
