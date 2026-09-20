#!/usr/bin/env python3
"""Backtranslate GPT-repaired translations with pinned MiLMMT for independent QA."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


MODEL_ID = "xiaomi-research/MiLMMT-46-12B-v1.0"
REVISION = "a27dbbb37142ff076990820a1c9f0827beb5d6ea"
PROTOCOL = "gpt56sol_repair_milmmt_backtranslation_v1"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--expected-count", type=int, default=430)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.model_path.resolve().name != REVISION:
        raise ValueError("MiLMMT revision mismatch")
    rows = load_jsonl(args.source)
    if len(rows) != args.expected_count or len({row["key"] for row in rows}) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} unique GPT translations")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "pairs": len(rows),
        "backtranslation_model": MODEL_ID,
        "backtranslation_revision": REVISION,
        "temperature": 0,
        "max_tokens": 1024,
        "packages": {x: importlib.metadata.version(x) for x in ["torch", "transformers", "tokenizers"]},
    }
    meta_path = args.out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Changed backtranslation protocol; use a new output directory")
    atomic_json(meta_path, metadata)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    dest = args.out / "translations.jsonl"
    prior = load_jsonl(dest) if dest.exists() else []
    done = {row["key"] for row in prior}
    if len(done) != len(prior):
        raise ValueError("Duplicate saved backtranslations")
    todo = [row for row in rows if row["key"] not in done]
    lock = (args.out / "backtranslate.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    with dest.open("a") as handle:
        for start in range(0, len(todo), 8):
            batch = todo[start : start + 8]
            prompts = []
            for row in batch:
                language = "Chinese (Simplified)" if row["language"] == "Chinese" else row["language"]
                prompts.append(f"Translate this from {language} to English:\n{language}: {row['translated']}\nEnglish:")
            encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
            if encoded["input_ids"].shape[1] + 1024 > 4096:
                raise ValueError("Backtranslation context overflow")
            with torch.inference_mode():
                sequences = model.generate(**encoded, max_new_tokens=1024, do_sample=False, use_cache=True)
            width = encoded["input_ids"].shape[1]
            for row, prompt, sequence in zip(batch, prompts, sequences):
                tokens = sequence[width:].tolist()
                clean = []
                finish_reason = "length"
                for token in tokens:
                    if token == tokenizer.eos_token_id:
                        finish_reason = "stop"
                        break
                    if token != tokenizer.pad_token_id:
                        clean.append(token)
                result = dict(row)
                result.update({
                    "backtranslation": tokenizer.decode(clean, skip_special_tokens=True).strip(),
                    "back_prompt": prompt,
                    "back_finish_reason": finish_reason,
                    "back_terminated": finish_reason == "stop" and bool(clean),
                    "back_input_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                    "back_output_tokens": len(clean),
                    "backtranslation_model": MODEL_ID,
                    "backtranslation_revision": REVISION,
                    "backtranslation_protocol": PROTOCOL,
                })
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print("backtranslated", min(start + 8, len(todo)), "/", len(todo), flush=True)

    final = load_jsonl(dest)
    if len(final) != len(rows) or len({row["key"] for row in final}) != len(rows):
        raise ValueError("Incomplete backtranslations")
    atomic_json(args.out / "complete.json", {
        "pairs": len(final),
        "translations_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "all_back_terminated": all(row["back_terminated"] for row in final),
    })


if __name__ == "__main__":
    main()
