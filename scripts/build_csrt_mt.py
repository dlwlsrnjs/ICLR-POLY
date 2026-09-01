#!/usr/bin/env python3
"""Full-coverage CSRT baseline via NLLB segment translation (Yoo et al. 2024).

Our official-span CSRT (csrt_k) is faithful but only covers items that have
verified content-span alignments. To give CSRT full coverage AND reproduce its
original construction exactly, this builds code-switched sentences by splitting
the English request into k contiguous segments and MACHINE-TRANSLATING each
segment into a different language with NLLB-200, then concatenating them in
reading order — a genuine intra-sentential code-switch over the whole sentence.

This uses machine translation (as the original CSRT does, and as AttaQ's own
multilingual translations were produced). It is therefore a BASELINE-ONLY tool;
our proposed method never introduces new translation. The resulting prompts are
written into the data rows under prompts["csrt_mt_n{k}"] so the generator emits
them as direct-submission conditions. Controlled text -> 0600.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

# CSRT segment languages (subset of our set that NLLB supports well).
NLLB_CODE = {
    "French": "fra_Latn", "Spanish": "spa_Latn", "German": "deu_Latn",
    "Russian": "rus_Cyrl", "Chinese": "zho_Hans", "Arabic": "arb_Arab",
}
CSRT_LANG_ORDER = ["French", "Spanish", "German", "Russian", "Chinese", "Arabic"]


def secure_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as h:
        h.write(text)
    os.chmod(path, 0o600)


def split_into_segments(text: str, k: int) -> list[str]:
    """Split into k contiguous word-runs of roughly equal length."""
    words = text.split()
    if len(words) < k:
        return [text]
    size = len(words) / k
    segs, start = [], 0
    for i in range(1, k + 1):
        end = len(words) if i == k else round(i * size)
        segs.append(" ".join(words[start:end]))
        start = end
    return [s for s in segs if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = [json.loads(l) for l in Path(args.data).open(encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device).eval()

    # Collect every (segment, target_lang) translation job across all items/ks.
    jobs = []  # (row_idx, k, seg_idx, lang, text)
    plans = {}  # (row_idx,k) -> list of (seg_idx, lang, english_seg)
    for ri, r in enumerate(rows):
        for k in args.ks:
            segs = split_into_segments(r["original"], k)
            langs = [CSRT_LANG_ORDER[i % len(CSRT_LANG_ORDER)] for i in range(len(segs))]
            # Keep the first segment in English (natural code-switch onset), switch the rest.
            plan = []
            for si, (seg, lang) in enumerate(zip(segs, langs)):
                if si == 0:
                    plan.append((si, "English", seg))
                else:
                    plan.append((si, lang, seg))
                    jobs.append((ri, k, si, lang, seg))
            plans[(ri, k)] = plan

    # Batched NLLB translation grouped by target language.
    translated: dict[tuple, str] = {}
    started = time.time()
    by_lang: dict[str, list] = {}
    for job in jobs:
        by_lang.setdefault(job[3], []).append(job)
    for lang, group in by_lang.items():
        tgt = NLLB_CODE[lang]
        tok.src_lang = "eng_Latn"
        bos = tok.convert_tokens_to_ids(tgt)
        for start in range(0, len(group), args.batch_size):
            batch = group[start:start + args.batch_size]
            enc = tok([g[4] for g in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(args.device)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=256)
            outs = tok.batch_decode(gen, skip_special_tokens=True)
            for g, o in zip(batch, outs):
                translated[(g[0], g[1], g[2])] = o.strip()
        print(json.dumps({"stage": "nllb", "lang": lang, "done": len(group)}), flush=True)

    # Assemble code-switched prompts.
    for (ri, k), plan in plans.items():
        parts = []
        for si, lang, seg in plan:
            parts.append(seg if lang == "English" else translated.get((ri, k, si), seg))
        rows[ri].setdefault("prompts", {})[f"csrt_mt_n{k}"] = " ".join(parts)

    secure_write(Path(args.output),
                 "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"n_items": len(rows), "ks": args.ks, "n_translations": len(jobs),
                      "elapsed_seconds": round(time.time() - started, 2),
                      "output": args.output,
                      "note": "MACHINE-TRANSLATED baseline (CSRT); not used by our method"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
