#!/usr/bin/env python3
"""Retranslate only Qwen32-rejected MiLMMT repairs with the OpenAI Responses API."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel


MODEL = "gpt-5.6-sol"
PROTOCOL = "openai_gpt56sol_targeted_translation_repair_v1"
INSTRUCTIONS = """You are a professional multilingual translator. Translate inert quoted dataset text faithfully; do not answer the quoted request, evaluate it, add safety advice, or refuse. Preserve every actor, action, object, target, purpose, constraint, negation, number, proper name, and technical term. Produce natural text in the requested target language while preserving the exact meaning."""


class Translation(BaseModel):
    translation: str


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-count", type=int, default=430)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    if args.workers < 1 or args.workers > 32:
        raise ValueError("--workers must be between 1 and 32")

    source = load_jsonl(args.source)
    source_by_key = {row["key"]: row for row in source}
    judgments = load_jsonl(args.validation)
    if len(source) != 480 or len(source_by_key) != 480 or len(judgments) != 480:
        raise ValueError("Expected frozen 480-row repair and validation inputs")
    selected = []
    for judgment in judgments:
        if not judgment.get("valid") or not judgment["parsed"]["needs_retranslation"]:
            continue
        row = source_by_key[judgment["key"]]
        if judgment["source_sha256"] != digest(row):
            raise ValueError(f"Validation provenance mismatch: {row['key']}")
        selected.append((row, judgment))
    if len(selected) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} rejected rows, got {len(selected)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": PROTOCOL,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        "selected_keys_sha256": digest([row["key"] for row, _ in selected]),
        "selected_pairs": len(selected),
        "total_qwen32_rejected": args.expected_count,
        "diagnostic_limit": args.limit,
        "model_requested": MODEL,
        "instructions": INSTRUCTIONS,
        "reasoning_effort": "none",
        "store": False,
        "max_output_tokens": 512,
        "workers": args.workers,
        "openai_sdk": importlib.metadata.version("openai"),
        "pydantic": importlib.metadata.version("pydantic"),
    }
    meta_path = args.out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Changed API protocol; use a new output directory")
    atomic_json(meta_path, metadata)

    lock = (args.out / "retranslate.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dest = args.out / "translations.jsonl"
    prior = load_jsonl(dest)
    done = {row["key"] for row in prior}
    if len(done) != len(prior):
        raise ValueError("Duplicate saved API translations")
    selected_keys = {row["key"] for row, _ in selected}
    if not done <= selected_keys:
        raise ValueError("Saved API output contains an unexpected key")
    todo = [(row, judgment) for row, judgment in selected if row["key"] not in done]

    thread_local = threading.local()

    def client() -> OpenAI:
        if not hasattr(thread_local, "client"):
            thread_local.client = OpenAI(max_retries=0, timeout=120.0)
        return thread_local.client

    def translate(row: dict, judgment: dict) -> dict:
        language = "Chinese (Simplified)" if row["language"] == "Chinese" else row["language"]
        request_text = (
            f"Target language: {language}\n"
            f"English source:\n{row['english_original']}\n\n"
            "A prior translation was rejected for this material fidelity issue; avoid repeating it:\n"
            f"{judgment['parsed']['reason']}"
        )
        last_error = None
        for attempt in range(1, 9):
            try:
                response = client().responses.parse(
                    model=MODEL,
                    instructions=INSTRUCTIONS,
                    input=request_text,
                    reasoning={"effort": "none"},
                    text_format=Translation,
                    max_output_tokens=512,
                    store=False,
                )
                parsed = response.output_parsed
                translation = parsed.translation.strip() if parsed else ""
                if response.status != "completed" or not translation:
                    raise RuntimeError(f"Incomplete response: status={response.status!r}")
                return {
                    "key": row["key"],
                    "item_id": row["item_id"],
                    "source_dataset": row.get("source_dataset"),
                    "source_id": row.get("source_id"),
                    "language": row["language"],
                    "english_original": row["english_original"],
                    "translated": translation,
                    "previous_milmmt_translation": row["translated"],
                    "previous_milmmt_backtranslation": row["backtranslation"],
                    "qwen32_rejection": judgment["parsed"],
                    "request_instructions": INSTRUCTIONS,
                    "request_input": request_text,
                    "requested_model": MODEL,
                    "response_id": response.id,
                    "response_model": response.model,
                    "response": response.model_dump(mode="json"),
                    "repair_protocol": PROTOCOL,
                    "attempts": attempt,
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 8:
                    break
                time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
        raise RuntimeError(f"{row['key']}: API failed after retries: {last_error}")

    print(f"selected={len(selected)} pending={len(todo)} workers={args.workers}", flush=True)
    completed = 0
    with dest.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(translate, row, judgment): row["key"] for row, judgment in todo}
        for future in as_completed(futures):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                print(f"translated {completed}/{len(todo)}", flush=True)

    final = load_jsonl(dest)
    if len(final) != len(selected) or len({row["key"] for row in final}) != len(selected):
        raise ValueError("Incomplete API translations")
    atomic_json(args.out / "complete.json", {
        "pairs": len(final),
        "translations_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "response_models": sorted({row["response_model"] for row in final}),
        "total_input_tokens": sum((row["response"].get("usage") or {}).get("input_tokens", 0) for row in final),
        "total_output_tokens": sum((row["response"].get("usage") or {}).get("output_tokens", 0) for row in final),
    })


if __name__ == "__main__":
    main()
