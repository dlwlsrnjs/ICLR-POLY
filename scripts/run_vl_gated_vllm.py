#!/usr/bin/env python3
"""vLLM-backed multimodal gated-ASR generation (fast path for the VL configs).

Same conditions and output schema as run_vl_gated.py, but uses vLLM's continuous
batching instead of HF .generate() — much higher throughput on the VL bottleneck.
Runs inside the isolated .vllm_env. Output rows stay compatible with the HF-based
reconstruction and Qwen3Guard judges.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polyjig_pilot import reconstruction_score  # noqa: E402
from run_vl_gated import build_conditions, IMAGE_ROOT, secure_write  # noqa: E402
from run_polyjig_gated import section  # noqa: E402


def data_uri(path: str, max_side: int = 1024) -> str:
    """Encode an image as a data URI, downscaled so the base64 payload stays small.

    The model further caps resolution via mm_processor_kwargs(max_pixels); this only
    avoids holding thousands of full-resolution PNGs in memory at once.
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--interleave-ns", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--translated-langs", nargs="*", default=["Finnish"])
    ap.add_argument("--blank-image", default=None)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--shard", default="0/1", help="i/n item sharding for multi-GPU")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    records = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    records = [r for idx, r in enumerate(records) if idx % shard_n == shard_i]
    jobs = []
    for record in records:
        for cond in build_conditions(record, args.seed, args.interleave_ns, args.translated_langs):
            jobs.append((record, cond))

    # One corrupt/0-byte PNG used to abort an entire shard mid-run (and silently
    # halve a config's coverage). Validate up front instead: drop every condition
    # of an item whose image cannot be decoded, and report which ones were dropped.
    skipped_items: list[dict[str, str]] = []
    if not args.blank_image:
        from PIL import Image, UnidentifiedImageError
        bad_refs: dict[str, str] = {}
        for _, cond in jobs:
            ref = str(IMAGE_ROOT / cond["image_path"])
            if ref in bad_refs:
                continue
            try:
                Image.open(ref).verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                bad_refs[ref] = type(exc).__name__
        if bad_refs:
            dropped, seen = set(), set()
            for record, cond in jobs:
                ref = str(IMAGE_ROOT / cond["image_path"])
                if ref in bad_refs:
                    dropped.add(record["item_id"])
                    if record["item_id"] not in seen:
                        seen.add(record["item_id"])
                        skipped_items.append({"item_id": record["item_id"],
                                              "image_path": cond["image_path"],
                                              "error": bad_refs[ref]})
            jobs = [(r, c) for r, c in jobs if r["item_id"] not in dropped]
            print(json.dumps({"stage": "image_validation", "unreadable_images": len(bad_refs),
                              "dropped_items": len(dropped)}), flush=True)

    # Cache data URIs per image path (many conditions share the item's image).
    uri_cache: dict[str, str] = {}
    def uri_for(cond) -> str:
        ref = args.blank_image if args.blank_image else str(IMAGE_ROOT / cond["image_path"])
        if ref not in uri_cache:
            uri_cache[ref] = data_uri(ref)
        return uri_cache[ref]

    conversations = []
    for record, cond in jobs:
        conversations.append([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri_for(cond)}},
            {"type": "text", "text": cond["prompt"]},
        ]}])

    llm = LLM(
        model=args.target, dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8192, enforce_eager=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    started = time.time()
    results = llm.chat(conversations, sampling, use_tqdm=True)
    outputs = [r.outputs[0].text for r in results]

    rows = []
    for (record, cond), output in zip(jobs, outputs):
        if cond["is_direct"]:
            reconstructed, answer, rscore = record["original"], output, 1.0
        else:
            reconstructed = section(output, "RECONSTRUCTED", "ANSWER")
            answer = section(output, "ANSWER")
            rscore = reconstruction_score(reconstructed, record["original"])
        rows.append({
            "item_id": record["item_id"], "scenario": record["scenario"],
            "modality": record.get("modality", "text_dominant"),
            "risk_type": record.get("risk_type", "text_dominant"),
            "condition": cond["condition"], "is_direct": int(cond["is_direct"]),
            "image_path": ("<blank>" if args.blank_image else cond["image_path"]),
            "original": record["original"], "prompt": cond["prompt"], "raw_output": output,
            "reconstructed": reconstructed, "answer_section": answer,
            "reconstruction_score": round(rscore, 4), "reconstruction_pass_080": int(rscore >= 0.8),
            "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "target_model": args.target, "images_supplied": True, "engine": "vllm",
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o700)
    output_path = outdir / "restricted_target_outputs.jsonl"
    secure_write(output_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {
        "design": "vLLM multimodal gated-ASR", "engine": "vllm",
        "n_items": len(records), "n_generations": len(rows),
        "conditions": sorted({r["condition"] for r in rows}),
        "target": args.target, "blank_image": bool(args.blank_image),
        "skipped_unreadable_image_items": skipped_items,
        "elapsed_seconds": round(time.time() - started, 2),
        "throughput_gen_per_s": round(len(rows) / max(1e-9, time.time() - started), 2),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (outdir / "target_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
