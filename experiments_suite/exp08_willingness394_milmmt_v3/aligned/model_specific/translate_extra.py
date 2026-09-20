#!/usr/bin/env python3
"""Translate all 331 selected prompts with pinned MiLMMT and backtranslate them."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import threading
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
TARGET_ROOT = RUN.parents[1]
MODEL_ID = "xiaomi-research/MiLMMT-46-12B-v1.0"
REVISION = "a27dbbb37142ff076990820a1c9f0827beb5d6ea"
LANGUAGES = ("Norwegian", "Finnish", "Arabic")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_generated(llm, tokenizer, rows: list[dict], path: Path, forward: bool) -> None:
    from vllm import SamplingParams

    existing = read_jsonl(path)
    known = {row["key"]: row for row in existing}
    if len(known) != len(existing):
        raise RuntimeError(f"Duplicate keys in {path}")
    todo = [row for row in rows if row["key"] not in known]
    print(f"{path.name} remaining={len(todo)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), 64):
            batch = todo[start : start + 64]
            prompts = []
            for row in batch:
                if forward:
                    source, target, text = "English", row["language"], row["english_original"]
                else:
                    source, target, text = row["language"], "English", row["translated"]
                source_name = "Chinese (Simplified)" if source == "Chinese" else source
                target_name = "Chinese (Simplified)" if target == "Chinese" else target
                prompts.append(f"Translate this from {source_name} to {target_name}:\n{source_name}: {text}\n{target_name}:")
            token_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            if max(map(len, token_ids)) + 1024 > 4096:
                raise RuntimeError("Translation context overflow")
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in token_ids],
                SamplingParams(temperature=0, top_k=1, max_tokens=1024, seed=20260918),
                use_tqdm=False,
            )
            for row, prompt, ids, output in zip(batch, prompts, token_ids, outputs):
                generated = output.outputs[0]
                field = "translated" if forward else "backtranslation"
                prefix = "forward" if forward else "back"
                result = dict(row)
                result.update(
                    {
                        field: generated.text.strip(),
                        f"{prefix}_finish_reason": generated.finish_reason,
                        f"{prefix}_terminated": generated.finish_reason == "stop" and bool(generated.text.strip()),
                        f"{prefix}_prompt": prompt,
                        f"{prefix}_input_tokens": len(ids),
                        f"{prefix}_output_tokens": len(generated.token_ids),
                        "translation_model": MODEL_ID,
                        "translation_revision": REVISION,
                    }
                )
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{path.name} {min(start + 64, len(todo))}/{len(todo)}", flush=True)


def config_compat(config):
    text_config = config.text_config
    pattern = text_config._sliding_window_pattern
    if pattern != 6:
        raise RuntimeError(f"Unexpected sliding-window pattern: {pattern}")
    expected = [
        "full_attention" if (index + 1) % pattern == 0 else "sliding_attention"
        for index in range(text_config.num_hidden_layers)
    ]
    if text_config.layer_types != expected:
        raise RuntimeError("MiLMMT layer pattern changed")
    text_config.sliding_window_pattern = pattern
    return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--items',type=Path,required=True);parser.add_argument('--model',type=Path,required=True);parser.add_argument('--languages',nargs='+',required=True);parser.add_argument('--gpu',default='1');a=parser.parse_args()
    if metadata.version('vllm')!='0.8.5':raise RuntimeError('Requires pinned vLLM 0.8.5')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu:raise RuntimeError('GPU mismatch')
    if a.model.resolve().name!=REVISION:raise RuntimeError('Translator revision mismatch')
    items=json.loads(a.items.read_text());assert len(items)==331
    jobs=[dict(key=item['id']+'|'+lang,item_id=item['id'],language=lang,english_original=item['prompt']) for item in items for lang in a.languages]
    a.run.mkdir(parents=True,exist_ok=True)
    meta=dict(model=MODEL_ID,revision=REVISION,languages=a.languages,items_sha256=hashlib.sha256(a.items.read_bytes()).hexdigest(),max_tokens=1024,seed=20260918,temperature=0,top_k=1,gpu=a.gpu,packages={x:metadata.version(x) for x in ['vllm','torch','transformers','tokenizers']})
    mp=a.run/'metadata.json'
    if mp.exists() and json.loads(mp.read_text())!=meta:raise RuntimeError('Changed translation inputs')
    mp.write_text(json.dumps(meta,indent=2)+'\n')
    from transformers import AutoTokenizer
    from vllm import LLM
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    llm=LLM(model=str(a.model),dtype='bfloat16',max_model_len=4096,gpu_memory_utilization=.90,tensor_parallel_size=1,enforce_eager=True,max_num_seqs=16,max_num_batched_tokens=2048,enable_chunked_prefill=True,disable_log_stats=True,limit_mm_per_prompt={'image':0},hf_overrides=config_compat)
    append_generated(llm,tok,jobs,a.run/'forward.jsonl',True)
    rows=read_jsonl(a.run/'forward.jsonl');assert len(rows)==len(jobs)
    append_generated(llm,tok,rows,a.run/'translations.jsonl',False)
    assert len(read_jsonl(a.run/'translations.jsonl'))==len(jobs)
    (a.run/'complete.json').write_text(json.dumps(dict(pairs=len(jobs),sha256=hashlib.sha256((a.run/'translations.jsonl').read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
