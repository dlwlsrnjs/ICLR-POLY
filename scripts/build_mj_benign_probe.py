#!/usr/bin/env python3
"""Build the harmless probe file for the MultiJail configuration space.

The existing harmless probe (private_artifacts/panel_v2/benign_probe.jsonl) carries official FLORES
translations for the Lingua language set only, so it cannot build MultiJail's constructions, whose
languages are Bengali, Swahili, Javanese, Korean, Thai, Italian and Vietnamese. The FLORES
repositories that hold those languages are access-restricted for this account, so we translate the
same harmless FLORES English sentences into the missing languages with the locally cached
NLLB-200-distilled-1.3B, and keep the official Arabic and Chinese columns that the file already has.

The probe measures how well a target reassembles a construction, and the prior only needs the
ranking across constructions, so machine-translated harmless text is adequate. It is a difference in
translation provenance from the harmful items, and the paper says so.

No harmful content is read or written: input and output are FLORES news sentences.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

MISSING = {"Italian": "ita_Latn", "Vietnamese": "vie_Latn", "Korean": "kor_Hang",
           "Thai": "tha_Thai", "Bengali": "ben_Beng", "Swahili": "swh_Latn", "Javanese": "jav_Latn"}
KEEP = ("English", "Arabic", "Chinese")     # already official FLORES in the source file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="private_artifacts/panel_v2/benign_probe.jsonl")
    ap.add_argument("--out", default="private_artifacts/multijail_v1/benign_probe.jsonl")
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.src, encoding="utf-8")][:a.n_items]
    eng = [r["questions"]["English"] for r in rows]
    print(json.dumps({"stage": "build-mj-benign", "items": len(rows), "translate_into": list(MISSING)}), flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "facebook/nllb-200-distilled-1.3B"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16, local_files_only=True).cuda().eval()

    out_cols = {}
    for lang, code in MISSING.items():
        bos = tok.convert_tokens_to_ids(code)
        got = []
        with torch.no_grad():
            for i in range(0, len(eng), a.batch):
                enc = tok(eng[i:i + a.batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to("cuda")
                gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=256, num_beams=4)
                got += tok.batch_decode(gen, skip_special_tokens=True)
        out_cols[lang] = got
        print(json.dumps({"translated": lang, "n": len(got), "sample": got[0][:80]}), flush=True)

    new = []
    for i, r in enumerate(rows):
        q = {k: r["questions"][k] for k in KEEP if k in r["questions"]}
        for lang in MISSING:
            q[lang] = out_cols[lang][i]
        new.append({"item_id": r["item_id"], "scenario": "Benign Control", "risk_type": "benign",
                    "original": q["English"], "questions": q,
                    "translation_source": {**{k: "FLORES official" for k in KEEP},
                                           **{k: "NLLB-200-distilled-1.3B from English" for k in MISSING}},
                    "image_paths": {}, "language_assignment": []})
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as h:
        for r in new:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"saved": str(p), "items": len(new), "languages": sorted(new[0]["questions"])}))


if __name__ == "__main__":
    raise SystemExit(main())
