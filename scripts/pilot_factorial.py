#!/usr/bin/env python3
"""Small paired factorial check for the PolyJigsaw pilot."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

from polyjig_pilot import LANG_NAMES, LANGS, SAFE_SAMPLES, reconstruction_score


def prompt_for(sample, condition, assignment, order):
    lines = [
        "You are solving a harmless ordering puzzle.",
        "Each line is one fragment of a single English instruction.",
        "Translate when needed, infer the coherent order, and output ONLY the reconstructed English sentence.",
        "Do not explain your reasoning.",
        "",
    ]
    for shown_idx, frag_idx in enumerate(order, 1):
        frag = sample.fragments[frag_idx]
        if condition.startswith("english"):
            label, text = "English", frag.en
        else:
            lang = assignment[frag_idx]
            label, text = LANG_NAMES[lang], frag.translations[lang]
        lines.append(f"{shown_idx}. [{label}] {text}")
    return "\n".join(lines)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    outdir = Path("polyjig_outputs/qwen7b_factorial")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260826)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="cpu", low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()

    rows = []
    conditions = (
        "english_ordered", "english_shuffled", "spanish_shuffled",
        "multilingual_ordered", "multilingual_shuffled",
    )
    prompts = []
    metadata = []
    for sample in SAFE_SAMPLES:
        n = len(sample.fragments)
        shuffled = list(range(n))
        rng.shuffle(shuffled)
        multilingual = LANGS.copy()
        rng.shuffle(multilingual)
        for condition in conditions:
            order = list(range(n)) if condition.endswith("ordered") else shuffled
            assignment = ["es"] * n if condition == "spanish_shuffled" else multilingual
            prompts.append(prompt_for(sample, condition, assignment, order))
            metadata.append((sample, condition, assignment, order))

    rendered = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    ) for p in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True)
    started = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=48, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - started
    outputs = tokenizer.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    for (sample, condition, assignment, order), prompt, output in zip(metadata, prompts, outputs):
        rows.append({
            "sample_id": sample.sample_id,
            "condition": condition,
            "assignment": ",".join(assignment),
            "order": ",".join(map(str, order)),
            "reconstruction_score": round(reconstruction_score(output.strip(), sample.original), 4),
            "output": output.strip(),
            "gold": sample.original,
        })

    with (outdir / "factorial_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": model_id,
        "seed": 20260826,
        "n_samples": len(SAFE_SAMPLES),
        "n_generations": len(rows),
        "batch_elapsed_seconds": round(elapsed, 2),
        "condition_means": {
            c: round(sum(r["reconstruction_score"] for r in rows if r["condition"] == c) / len(SAFE_SAMPLES), 4)
            for c in conditions
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        print(row["sample_id"], row["condition"], row["reconstruction_score"], repr(row["output"]))


if __name__ == "__main__":
    main()
