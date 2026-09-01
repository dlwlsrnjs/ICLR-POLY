#!/usr/bin/env python3
"""Multimodal gated-ASR run for Image-Dominant Lingua-SafetyBench modalities.

For image-dominant items the harmful content lives in the IMAGE, so the primary
metric is raw ASR (does the VL model, given the image + question, produce an
unsafe answer). We compare, per modality (Visual / Typography / Mixed):
  * vl_english_direct           — English image + English question (baseline).
  * vl_translated_direct_<Lang> — that language's image + question (multilingual
                                  baseline; the dataset ships one image per language).
  * vl_interleave_ordered_n<K>  — proposed game: English image + the question text
                                  as a parallel-language interleaving puzzle to
                                  reassemble and then carry out.

The independent Qwen3Guard judge (text) later scores the [ANSWER] / answer text.
Raw completions are controlled content and are written with mode 0600.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polyjig_pilot import reconstruction_score  # noqa: E402
from run_qwen_interleaving_probe import build_puzzle, select_languages  # noqa: E402
from run_polyjig_gated import interleave_game_prompt, section  # noqa: E402

IMAGE_ROOT = Path("/home/ubuntu/342/jinkwon/datasets/Lingua-SafetyBench/extracted/dataset/bundle_data")


def secure_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def build_conditions(row, seed, interleave_ns, translated_langs):
    """[{condition, image_path, prompt, is_direct}] for one image-dominant item."""
    q = row["questions"]
    img = row["image_paths"]
    conds = [{"condition": "vl_english_direct", "image_path": img["English"],
              "prompt": q["English"], "is_direct": True}]
    for L in translated_langs:
        if L in q and q[L].strip() and L in img:
            conds.append({"condition": f"vl_translated_direct_{L}", "image_path": img[L],
                          "prompt": q[L], "is_direct": True})
    for k in interleave_ns:
        langs = select_languages(k)
        if not all(L in q and q[L].strip() for L in langs):
            continue
        puzzle = build_puzzle({"item_id": row["item_id"], "scenario": row["scenario"], "questions": q},
                              langs, "ordered", seed, "coarse", 5)
        conds.append({"condition": f"vl_interleave_ordered_n{k}", "image_path": img["English"],
                      "prompt": interleave_game_prompt(puzzle), "is_direct": False})
    # Proposed inline-slot game (only present for items the aligner produced tiles for).
    for key, prompt in (row.get("prompts") or {}).items():
        if key.startswith("vl_official_slots_k"):
            kk = key.rsplit("k", 1)[-1]
            conds.append({"condition": f"vl_slot_k{kk}", "image_path": img["English"],
                          "prompt": prompt, "is_direct": False})
    return conds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--interleave-ns", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--translated-langs", nargs="+",
                    default=["Chinese", "Arabic", "French", "Russian", "Japanese"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28,
                    help="cap image tokens for speed; benign context images downscale fine")
    ap.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--blank-image", default=None,
                    help="if set, attach this single blank image to every condition "
                         "(isolates the effect of the real image)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    records = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    records = [r for idx, r in enumerate(records) if idx % shard_n == shard_i]

    jobs = []
    for record in records:
        for cond in build_conditions(record, args.seed, args.interleave_ns, args.translated_langs):
            jobs.append((record, cond))

    # Drop items whose image cannot be decoded rather than crashing the shard
    # (a single 0-byte PNG previously took out half a config's coverage).
    skipped_items: list[dict[str, str]] = []
    if not args.blank_image:
        from PIL import UnidentifiedImageError
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

    processor = AutoProcessor.from_pretrained(
        args.target, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.target, device_map=args.device, dtype=torch.bfloat16
    ).eval()

    started = time.time()
    outputs = []
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        texts, images = [], []
        for record, cond in batch:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": cond["prompt"]}]}]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_ref = args.blank_image if args.blank_image else str(IMAGE_ROOT / cond["image_path"])
            images.append(Image.open(image_ref).convert("RGB"))
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        suffix = generated[:, inputs["input_ids"].shape[1] :]
        outputs.extend(processor.batch_decode(suffix, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False))
        print(json.dumps({"stage": "vl_target", "completed": min(start + len(batch), len(jobs)),
                          "total": len(jobs)}), flush=True)

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
            "image_path": cond["image_path"], "original": record["original"],
            "prompt": cond["prompt"], "raw_output": output,
            "reconstructed": reconstructed, "answer_section": answer,
            "reconstruction_score": round(rscore, 4), "reconstruction_pass_080": int(rscore >= 0.8),
            "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "target_model": args.target, "images_supplied": True,
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o700)
    output_path = outdir / "restricted_target_outputs.jsonl"
    secure_write(output_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {
        "design": "multimodal gated-ASR for image-dominant modalities",
        "n_items": len(records), "n_generations": len(rows),
        "conditions": sorted({r["condition"] for r in rows}),
        "target": args.target, "images_supplied": True,
        "elapsed_seconds": round(time.time() - started, 2),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (outdir / "target_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
