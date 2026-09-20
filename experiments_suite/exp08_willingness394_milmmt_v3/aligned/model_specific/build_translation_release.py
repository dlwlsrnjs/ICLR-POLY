#!/usr/bin/env python3
"""Build an immutable HF dataset revision from accepted translation repairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def judgment_map(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    result = {row["key"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate judgments in {path}")
    return result


def accepted_keys(judgments: dict[str, dict]) -> set[str]:
    return {
        key for key, value in judgments.items()
        if value.get("valid") and not value["parsed"]["needs_retranslation"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"Release output already exists: {args.out}")
    shutil.copytree(args.base, args.out, ignore=shutil.ignore_patterns(".cache"))

    original = load_jsonl(args.base / "data/translations_all.jsonl")
    if len(original) != 2317 or len({row["key"] for row in original}) != 2317:
        raise ValueError("Base translation bank is not the frozen 2317-pair release")
    original_by_key = {row["key"]: row for row in original}
    original_review = {row["key"] for row in original if not row["qa_pass"]}
    if len(original_review) != 504:
        raise ValueError("Expected 504 original review rows")

    rereview_dir = args.runs / "rereview_v1"
    milmmt_dir = args.runs / "milmmt_retranslated_v2"
    milmmt_qa_dir = args.runs / "milmmt_retranslated_v2_qwen32_validation"
    gpt_dir = args.runs / "gpt56sol_retranslated_430_v1"
    gpt_back_dir = args.runs / "gpt56sol_retranslated_430_v1_milmmt_backtranslation"
    gpt_qa_dir = args.runs / "gpt56sol_retranslated_430_v1_qwen32_validation"

    rereview = judgment_map(rereview_dir / "judgments.jsonl")
    milmmt_rows = {row["key"]: row for row in load_jsonl(milmmt_dir / "translations.jsonl")}
    milmmt_qa = judgment_map(milmmt_qa_dir / "judgments.jsonl")
    gpt_rows = {row["key"]: row for row in load_jsonl(gpt_back_dir / "translations.jsonl")}
    gpt_qa = judgment_map(gpt_qa_dir / "judgments.jsonl")
    if not (len(rereview) == 504 and len(milmmt_rows) == len(milmmt_qa) == 480 and len(gpt_rows) == len(gpt_qa) == 430):
        raise ValueError("Repair input counts changed")

    rereview_accept = accepted_keys(rereview)
    milmmt_accept = accepted_keys(milmmt_qa)
    gpt_accept = accepted_keys(gpt_qa)
    unresolved = set(gpt_qa) - gpt_accept
    if (len(rereview_accept), len(milmmt_accept), len(gpt_accept), len(unresolved)) != (24, 50, 376, 54):
        raise ValueError("Unexpected repair acceptance counts")
    if rereview_accept & milmmt_accept or rereview_accept & gpt_accept or milmmt_accept & gpt_accept:
        raise ValueError("Repair acceptance stages overlap")
    if original_review != rereview_accept | milmmt_accept | gpt_accept | unresolved:
        raise ValueError("The 504 original failures are not completely partitioned")

    merged = []
    resolution = {}
    for old in original:
        key = old["key"]
        row = dict(old)
        if key in rereview_accept:
            verdict = rereview[key]
            row.update({
                "qa_pass": True,
                "qa_reason": verdict["parsed"]["reason"],
                "qa_source_sha256": verdict["source_sha256"],
                "qa_protocol": "translation_semantics_rereview_v1",
                "repair_stage": "original_accepted_on_semantics_rereview",
            })
            resolution[key] = "original_rereview_accepted"
        elif key in milmmt_accept:
            verdict = milmmt_qa[key]
            row = dict(milmmt_rows[key])
            row.update({
                "language_code": old.get("language_code"),
                "qa_pass": True,
                "qa_reason": verdict["parsed"]["reason"],
                "qa_source_sha256": verdict["source_sha256"],
                "qa_protocol": "translation_semantics_rereview_v1",
                "repair_stage": "milmmt_v2_accepted_by_qwen32",
            })
            resolution[key] = "milmmt_repair_accepted"
        elif key in gpt_accept:
            verdict = gpt_qa[key]
            candidate = gpt_rows[key]
            row = {k: v for k, v in candidate.items() if k != "response"}
            usage = candidate["response"].get("usage") or {}
            row.update({
                "language_code": old.get("language_code"),
                "forward_prompt": candidate["request_instructions"] + "\n\n" + candidate["request_input"],
                "forward_finish_reason": candidate["response"].get("status"),
                "forward_terminated": candidate["response"].get("status") == "completed" and bool(candidate["translated"]),
                "forward_input_tokens": usage.get("input_tokens"),
                "forward_output_tokens": usage.get("output_tokens"),
                "translation_model": candidate["response_model"],
                "translation_revision": candidate["response_model"],
                "qa_pass": True,
                "qa_reason": verdict["parsed"]["reason"],
                "qa_source_sha256": verdict["source_sha256"],
                "qa_protocol": "translation_semantics_rereview_v1",
                "repair_stage": "gpt56sol_v1_accepted_by_qwen32",
            })
            resolution[key] = "gpt_repair_accepted"
        elif key in unresolved:
            row.update({
                "latest_repair_status": "gpt56sol_candidate_rejected_by_qwen32",
                "latest_repair_provenance": "provenance/repair_20260920/gpt56sol_repair_and_validation",
            })
            resolution[key] = "unresolved_original_retained"
        else:
            resolution[key] = "original_qa_accepted"
        if row["key"] != key or row["english_original"] != old["english_original"] or row["language"] != old["language"]:
            raise ValueError(f"Identity changed for {key}")
        merged.append(row)

    accepted = [row for row in merged if row["qa_pass"]]
    needs_review = [row for row in merged if not row["qa_pass"]]
    if (len(merged), len(accepted), len(needs_review)) != (2317, 2263, 54):
        raise ValueError("Merged release counts are wrong")
    language_counts = {}
    for language in sorted({row["language"] for row in merged}):
        lang_rows = [row for row in merged if row["language"] == language]
        if len(lang_rows) != 331:
            raise ValueError(f"Language count changed: {language}")
        language_counts[language] = {
            "total": len(lang_rows),
            "pass": sum(row["qa_pass"] for row in lang_rows),
            "needs_review": sum(not row["qa_pass"] for row in lang_rows),
        }
    if [row["key"] for row in merged] != [row["key"] for row in original]:
        raise ValueError("Row ordering changed")

    write_jsonl(args.out / "data/translations_all.jsonl", merged)
    write_jsonl(args.out / "data/translations_qa_accepted.jsonl", accepted)
    write_jsonl(args.out / "data/translations_needs_review.jsonl", needs_review)
    summary = {
        "source_bank_items": 331,
        "english_unchanged": True,
        "foreign_languages": 7,
        "foreign_pairs": 2317,
        "qa_pass": 2263,
        "qa_needs_review": 54,
        "languages": language_counts,
        "repair_resolution": dict(Counter(resolution.values())),
        "qa_is_automatic_not_human_gold": True,
        "intended_use": "Multilingual overrefusal prior evaluation; only Qwen32-accepted repair candidates were merged.",
    }
    write_json(args.out / "SUMMARY.json", summary)

    provenance = args.out / "provenance/repair_20260920"
    copies = {
        "rereview_v1": rereview_dir,
        "milmmt_repair_v2": milmmt_dir,
        "milmmt_repair_v2_qwen32_validation": milmmt_qa_dir,
        "gpt56sol_repair": gpt_dir,
        "gpt56sol_repair_milmmt_backtranslation": gpt_back_dir,
        "gpt56sol_repair_qwen32_validation": gpt_qa_dir,
    }
    for name, source in copies.items():
        shutil.copytree(source, provenance / name, ignore=shutil.ignore_patterns("*.lock"))
    write_json(provenance / "RELEASE_AUDIT.json", {
        "base_pairs": 2317,
        "base_qa_pass": 1813,
        "original_rereview_accepted": 24,
        "milmmt_repair_accepted": 50,
        "gpt56sol_repair_accepted": 376,
        "unresolved_original_retained": 54,
        "final_qa_pass": 2263,
        "final_needs_review": 54,
        "merge_policy": "Only Qwen2.5-32B semantics-QA accepted candidates replace base rows; unresolved rows retain the base translation.",
        "resolution_by_key": resolution,
    })

    readme = (args.base / "README.md").read_text()
    readme += """

## 2026-09-20 targeted repair revision

기존 QA 실패 504쌍을 의미 보존 기준으로 재검토했습니다. 원본 24쌍은 과도한 기존 판정으로 확인되어 통과 처리했고, MiLMMT 교정본 50쌍과 GPT-5.6 Sol 교정본 376쌍은 MiLMMT 영어 역번역 후 Qwen2.5-32B 의미 검증을 통과하여 병합했습니다. 최종 QA 통과는 2,263쌍이며 54쌍은 미해결 상태입니다. 미해결 GPT 후보는 본 데이터에 합격본으로 병합하지 않았고 `provenance/repair_20260920/`에 보존했습니다. 영어 원문 331개와 전체 2,317쌍은 그대로 유지됩니다.
"""
    (args.out / "README.md").write_text(readme)

    hashes = {}
    for path in sorted(args.out.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.json" or any(part.startswith(".") for part in path.relative_to(args.out).parts):
            continue
        hashes[str(path.relative_to(args.out))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.out / "SHA256SUMS.json", hashes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"hashed_files={len(hashes)} release={args.out}")


if __name__ == "__main__":
    main()
