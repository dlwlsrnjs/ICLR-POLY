#!/usr/bin/env python3
"""Generate paired Qwen2.5-VL completions with each original benchmark image."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from polyjig_pilot import reconstruction_score


def section(text: str, start: str, end: str | None = None) -> str:
    pattern = (
        rf"\[{re.escape(start)}\]\s*(.*?)(?:\[{re.escape(end)}\]|$)"
        if end else rf"\[{re.escape(start)}\]\s*(.*)$"
    )
    match = re.search(pattern, text, re.I | re.S)
    return match.group(1).strip() if match else ""


def secure_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="Optional smoke-test item limit")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    records = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    if args.limit:
        records = records[:args.limit]
    jobs = [
        (record, condition)
        for record in records
        for condition in args.conditions
        if condition in record.get("prompts", {})
    ]
    if not jobs:
        raise ValueError("No requested condition is present in the input records")
    processor = AutoProcessor.from_pretrained(
        args.target, local_files_only=True, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.target, device_map=args.device, dtype=torch.bfloat16, local_files_only=True
    ).eval()

    outputs = []
    started = time.time()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        images = []
        rendered = []
        for record, condition in batch:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": record["prompts"][condition]},
            ]}]
            rendered.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
            with Image.open(record["image_file"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(text=rendered, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        suffix = generated[:, inputs["input_ids"].shape[1]:]
        outputs.extend(processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))
        print(json.dumps({"stage": "target_multimodal", "completed": min(start + len(batch), len(jobs)), "total": len(jobs)}), flush=True)

    rows = []
    for (record, condition), output in zip(jobs, outputs):
        direct = condition == "english_direct"
        reconstructed = record["original"] if direct else section(output, "RECONSTRUCTED", "ANSWER")
        answer = output if direct else section(output, "ANSWER")
        lexical = 1.0 if direct else reconstruction_score(reconstructed, record["original"])
        rows.append({
            "item_id": record["item_id"],
            "scenario": record["scenario"],
            "risk_type": record["risk_type"],
            "condition": condition,
            "original": record["original"],
            "prompt": record["prompts"][condition],
            "raw_output": output,
            "reconstructed": reconstructed,
            "answer_section": answer,
            "lexical_reconstruction_score": round(lexical, 4),
            "image_file": record["image_file"],
            "image_sha256": record["image_sha256"],
            "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "target_model": args.target,
            "images_supplied": True,
            "decoding": {"temperature": 0, "do_sample": False, "max_new_tokens": args.max_new_tokens},
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o700)
    audit = outdir / "restricted_target_outputs.jsonl"
    secure_write(audit, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary = {
        "design": "paired Qwen2.5-VL original-image generation",
        "n_items": len({record["item_id"] for record, _ in jobs}),
        "n_generations": len(rows),
        "conditions": args.conditions,
        "images_supplied": True,
        "unique_image_hashes": len({row["image_sha256"] for row in rows}),
        "elapsed_seconds": round(time.time() - started, 2),
        "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    }
    secure_write(outdir / "target_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
