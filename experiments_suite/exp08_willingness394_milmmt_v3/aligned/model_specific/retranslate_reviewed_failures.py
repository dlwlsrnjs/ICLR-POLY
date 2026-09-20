#!/usr/bin/env python3
"""Correct only translations confirmed materially wrong by the semantics rereview."""
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


TRANSLATION_MODEL_ID = "xiaomi-research/MiLMMT-46-12B-v1.0"
TRANSLATION_REVISION = "a27dbbb37142ff076990820a1c9f0827beb5d6ea"
BACKTRANSLATION_MODEL_ID = TRANSLATION_MODEL_ID
BACKTRANSLATION_REVISION = TRANSLATION_REVISION
PROTOCOL = "milmmt_targeted_translation_repair_and_backtranslation_v2"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_V1="0")
    if args.model_path.resolve().name != TRANSLATION_REVISION:
        raise ValueError("MiLMMT model revision mismatch")
    source = load_jsonl(args.source)
    source_by_key = {row["key"]: row for row in source}
    reviews = load_jsonl(args.review)
    if len(source_by_key) != 504 or len(reviews) != 504:
        raise ValueError("Expected complete frozen rereview inputs")
    selected = []
    for review in reviews:
        row = source_by_key[review["key"]]
        if review["source_sha256"] != digest(row):
            raise ValueError("Rereview provenance mismatch")
        if review.get("valid") and review["parsed"]["needs_retranslation"]:
            selected.append((row, review))
    if not selected:
        raise ValueError("No valid material translation failures to repair")
    total_selected = len(selected)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "selected_keys_sha256": digest([row["key"] for row, _ in selected]),
        "selected_pairs": len(selected),
        "total_reviewed_failures": total_selected,
        "diagnostic_limit": args.limit,
        "translation_model": TRANSLATION_MODEL_ID,
        "translation_revision": TRANSLATION_REVISION,
        "backtranslation_model": BACKTRANSLATION_MODEL_ID,
        "backtranslation_revision": BACKTRANSLATION_REVISION,
        "backend": "transformers_blackwell",
        "temperature": 0,
        "max_tokens": 1024,
        "seed": 20260920,
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "tokenizers"]},
    }
    meta_path = args.out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Changed repair protocol; use a new output directory")
    atomic_json(meta_path, metadata)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    def load_model(path):
        loaded_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        loaded_tokenizer.padding_side = "left"
        loaded_model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        return loaded_model, loaded_tokenizer

    def generate(model, tokenizer, prompts):
        encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        if encoded["input_ids"].shape[1] + 1024 > 4096:
            raise ValueError("Translation repair context overflow")
        with torch.inference_mode():
            sequences = model.generate(**encoded, max_new_tokens=1024, do_sample=False, use_cache=True)
        width = encoded["input_ids"].shape[1]
        results = []
        for sequence in sequences:
            tokens = sequence[width:].tolist()
            finish_reason = "length"
            clean = []
            for token in tokens:
                if token == tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                if token != tokenizer.pad_token_id:
                    clean.append(token)
            results.append((tokenizer.decode(clean, skip_special_tokens=True).strip(), finish_reason, len(clean)))
        return results

    lock = (args.out / "repair.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    forward_path = args.out / "forward.jsonl"
    prior = load_jsonl(forward_path) if forward_path.exists() else []
    done = {row["key"] for row in prior}
    if len(done) != len(prior):
        raise ValueError("Duplicate forward repairs")
    todo = [(row, review) for row, review in selected if row["key"] not in done]
    model, tokenizer = load_model(args.model_path)
    with forward_path.open("a") as handle:
        for start in range(0, len(todo), 8):
            batch = todo[start:start + 8]
            prompts = []
            for row, review in batch:
                language = "Chinese (Simplified)" if row["language"] == "Chinese" else row["language"]
                reason = review["parsed"]["reason"]
                prompt = (
                    f"Translate this from English to {language}. Preserve all actors, actions, objects, targets, "
                    "purposes, constraints, negations, numbers, names, and technical terms. Return only the "
                    f"translation. Correct the previously identified issue: {reason}\n"
                    f"English: {row['english_original']}\n{language}:"
                )
                prompts.append(prompt)
            ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            outputs = generate(model, tokenizer, prompts)
            for (row, review), prompt, input_ids, generated in zip(batch, prompts, ids, outputs):
                text, finish_reason, output_tokens = generated
                repaired = dict(row)
                repaired.update({
                    "previous_translated": row["translated"],
                    "previous_backtranslation": row["backtranslation"],
                    "previous_qa_reason": row.get("qa_reason"),
                    "repair_review": review["parsed"],
                    "translated": text,
                    "forward_prompt": prompt,
                    "forward_finish_reason": finish_reason,
                    "forward_terminated": finish_reason == "stop" and bool(text),
                    "forward_input_tokens": len(input_ids),
                    "forward_output_tokens": output_tokens,
                    "translation_model": TRANSLATION_MODEL_ID,
                    "translation_revision": TRANSLATION_REVISION,
                    "repair_protocol": PROTOCOL,
                })
                handle.write(json.dumps(repaired, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print("forward repaired", min(start + 8, len(todo)), "/", len(todo), flush=True)

    forward = load_jsonl(forward_path)
    if len(forward) != len(selected):
        raise ValueError("Incomplete forward repairs")
    translations_path = args.out / "translations.jsonl"
    prior = load_jsonl(translations_path) if translations_path.exists() else []
    done = {row["key"] for row in prior}
    todo = [row for row in forward if row["key"] not in done]
    with translations_path.open("a") as handle:
        for start in range(0, len(todo), 8):
            batch = todo[start:start + 8]
            prompts = []
            for row in batch:
                language = "Chinese (Simplified)" if row["language"] == "Chinese" else row["language"]
                prompts.append(f"Translate this from {language} to English:\n{language}: {row['translated']}\nEnglish:")
            ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            outputs = generate(model, tokenizer, prompts)
            for row, prompt, input_ids, generated in zip(batch, prompts, ids, outputs):
                text, finish_reason, output_tokens = generated
                repaired = dict(row)
                repaired.update({
                    "backtranslation": text,
                    "back_prompt": prompt,
                    "back_finish_reason": finish_reason,
                    "back_terminated": finish_reason == "stop" and bool(text),
                    "back_input_tokens": len(input_ids),
                    "back_output_tokens": output_tokens,
                    "backtranslation_model": BACKTRANSLATION_MODEL_ID,
                    "backtranslation_revision": BACKTRANSLATION_REVISION,
                })
                handle.write(json.dumps(repaired, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print("backtranslated", min(start + 8, len(todo)), "/", len(todo), flush=True)

    final = load_jsonl(translations_path)
    if len(final) != len(selected) or len({row["key"] for row in final}) != len(selected):
        raise ValueError("Incomplete repaired translations")
    atomic_json(args.out / "complete.json", {
        "pairs": len(final), "translations_sha256": hashlib.sha256(translations_path.read_bytes()).hexdigest(),
        "all_forward_terminated": all(row["forward_terminated"] for row in final),
        "all_back_terminated": all(row["back_terminated"] for row in final),
    })
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
