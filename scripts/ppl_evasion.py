#!/usr/bin/env python3
"""Perplexity of the attack PROMPT under a multilingual reference LM (Qwen2.5-7B), to test whether a
perplexity filter (Alon & Kamfonas 2023) would flag each method. Not a per-query quantity: our method
selects among natural-language configurations, so this scores the stealth of the prompt text itself.
Methods share the same underlying requests. Reports the median and 90th-percentile per-token PPL.

Offline scoring (forward pass only, no generation). Reads stored prompts from the single-target study
and regenerates the DrAttack / FlipAttack prompts with the same functions used for those baselines.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sota_baselines import drattack_prompt, flipattack_prompt

SRC = "private_artifacts/panel_v2/granite_3_3_8b_instruct/harm_gen/restricted_target_outputs.jsonl"
REF = "Qwen/Qwen2.5-7B-Instruct"
N = 120


def load_prompts():
    by = {}
    for l in open(SRC):
        d = json.loads(l)
        by.setdefault(d["item_id"], {})[d["condition"]] = d
    items = list(by.items())[:N]
    methods = {"plain (English)": [], "translation (Finnish)": [], "interleave $n{=}4$ (ours)": [],
               "DrAttack": [], "FlipAttack (FCS)": []}
    for iid, cond in items:
        if "english_direct" not in cond or "interleave_ordered_n4" not in cond:
            continue
        req = cond["english_direct"]["original"]
        methods["plain (English)"].append(cond["english_direct"]["prompt"])
        if "translated_direct_Finnish" in cond:
            methods["translation (Finnish)"].append(cond["translated_direct_Finnish"]["prompt"])
        methods["interleave $n{=}4$ (ours)"].append(cond["interleave_ordered_n4"]["prompt"])
        methods["DrAttack"].append(drattack_prompt(req))
        methods["FlipAttack (FCS)"].append(flipattack_prompt(req, "FCS"))
    return methods


@torch.no_grad()
def ppl(model, tok, text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
    if ids.shape[1] < 2:
        return float("nan")
    out = model(ids, labels=ids)
    return float(math.exp(min(out.loss.item(), 20)))


def main():
    methods = load_prompts()
    tok = AutoTokenizer.from_pretrained(REF, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(REF, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                 local_files_only=True).eval()
    res = {}
    for name, prompts in methods.items():
        vals = [ppl(model, tok, p) for p in prompts if p]
        vals = [v for v in vals if v == v]
        res[name] = {"median": round(float(np.median(vals)), 1), "p90": round(float(np.percentile(vals, 90)), 1),
                     "n": len(vals)}
        print(f"{name:26s} median PPL {res[name]['median']:8.1f}  p90 {res[name]['p90']:8.1f}  (n={res[name]['n']})")
    Path("results/ppl_evasion_20260908.json").write_text(json.dumps(res, indent=2))
    # favorable = ours well below the obfuscation attacks and not the highest
    ours = res["interleave $n{=}4$ (ours)"]["median"]
    dr = res["DrAttack"]["median"]; flip = res["FlipAttack (FCS)"]["median"]; plain = res["plain (English)"]["median"]
    print(f"\nours median={ours}  plain={plain}  DrAttack={dr}  FlipAttack={flip}")
    print(f"FAVORABLE={ours < flip and ours <= 3*plain}  (ours below FlipAttack and within ~3x plain)")
    print("PPL_DONE")


if __name__ == "__main__":
    raise SystemExit(main())
