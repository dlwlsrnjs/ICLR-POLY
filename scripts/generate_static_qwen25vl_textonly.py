#!/usr/bin/env python3
"""Generate C0-C3 with Qwen2.5-VL in a strictly text-only setting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from polyjig_pilot import reconstruction_score


CONDITIONS = (
    "english_direct",
    "english_shuffled",
    "multilingual_ordered",
    "multilingual_shuffled",
)


def section(text: str, start: str, end: str | None = None) -> str:
    if end:
        pattern = rf"\[{re.escape(start)}\]\s*(.*?)(?:\[{re.escape(end)}\]|$)"
    else:
        pattern = rf"\[{re.escape(start)}\]\s*(.*)$"
    match = re.search(pattern, text, re.I | re.S)
    return match.group(1).strip() if match else ""


def secure_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        help="Prompt keys to generate; defaults to the fixed C0-C3 conditions",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    records = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    metadata = []
    prompts = []
    for record in records:
        for condition in args.conditions:
            metadata.append((record, condition))
            prompts.append(record["prompts"][condition])

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.target,
        device_map=args.device,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()

    started = time.time()
    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch
        ]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        suffix = generated[:, inputs["input_ids"].shape[1] :]
        outputs.extend(
            tokenizer.batch_decode(
                suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        )
        print(json.dumps({"stage": "target", "completed": min(start + len(batch), len(prompts)), "total": len(prompts)}), flush=True)

    rows = []
    for (record, condition), prompt, output in zip(metadata, prompts, outputs):
        reconstructed = record["original"] if condition == "english_direct" else section(output, "RECONSTRUCTED", "ANSWER")
        answer = output if condition == "english_direct" else section(output, "ANSWER")
        score = 1.0 if condition == "english_direct" else reconstruction_score(reconstructed, record["original"])
        rows.append({
            "item_id": record["item_id"],
            "scenario": record["scenario"],
            "risk_type": record["risk_type"],
            "condition": condition,
            "language_assignment": ",".join(record["language_assignment"]),
            "original": record["original"],
            "prompt": prompt,
            "raw_output": output,
            "reconstructed": reconstructed,
            "answer_section": answer,
            "reconstruction_score": round(score, 4),
            "reconstruction_pass_080": int(score >= 0.8),
            "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "target_model": args.target,
            "images_supplied": False,
            "decoding": {"temperature": 0, "do_sample": False, "max_new_tokens": args.max_new_tokens},
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o700)
    output_path = outdir / "restricted_target_outputs.jsonl"
    secure_write(output_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary = {
        "design": "paired Qwen2.5-VL text-only generation",
        "n_items": len(records),
        "n_generations": len(rows),
        "conditions": args.conditions,
        "target": args.target,
        "images_supplied": False,
        "elapsed_seconds": round(time.time() - started, 2),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    secure_write(outdir / "target_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
