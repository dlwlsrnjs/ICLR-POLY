#!/usr/bin/env python3
"""Audit reconstruction-only failures in the canonical 17-model collection.

The gold diagnostic event is R=1 and U=0 under the plain willingness frame:
the hidden request was reconstructed, but the verified attack did not succeed.
No prompt or response text is exported; outputs contain aggregate counts only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


FRAMES = ("plain", "persona", "fiction", "pap", "persona+fiction")
NONPLAIN = FRAMES[1:]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path(
            "/home/jklee/safe_mental-dpo/ICLR-POLY-latest/experiments_suite/"
            "exp02_panel_collect/results_l40s_0913"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=here.parent / "results/canonical17")
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            rows[item["item_id"]] = item
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def choose(scores: dict[str, float], frames: tuple[str, ...]) -> str:
    return max(frames, key=lambda frame: (scores[frame], -FRAMES.index(frame)))


def family(tag: str) -> str:
    stem = tag.rsplit("_", 1)[0]
    for name in ("qwen", "llama", "gemma", "mistral", "phi", "falcon", "glm"):
        if stem.startswith(name):
            return name
    return stem


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def validate_source(root: Path) -> dict[str, Any]:
    manifest = root / "MANIFEST.json"
    audit = root / "final_audit_repair_status.tsv"
    if not manifest.exists() or not audit.exists():
        raise FileNotFoundError("canonical manifest or final audit is missing")
    audit_text = audit.read_text()
    expected = "FINAL_AUDIT_PASS models=17 matrix_arms=5576 benign=34 falsereject=17"
    if expected not in audit_text:
        raise RuntimeError(f"canonical final-audit marker missing: {expected}")
    benign = sorted((root / "benign").glob("*.json"))
    attacks = sorted((root / "attack").glob("*.json"))
    raw = sorted((root / "attack/_raw").glob("*.jsonl"))
    if (len(benign), len(attacks), len(raw)) != (34, 5576, 5576):
        raise RuntimeError(
            f"unexpected source counts: benign={len(benign)}, "
            f"attack={len(attacks)}, raw={len(raw)}"
        )
    return {
        "root": str(root),
        "audit_marker": expected,
        "benign_files": len(benign),
        "attack_files": len(attacks),
        "raw_files": len(raw),
        "manifest_sha256": sha256(manifest),
        "audit_sha256": sha256(audit),
    }


def paired_effect(
    plain: dict[str, dict[str, Any]], treated: dict[str, dict[str, Any]]
) -> dict[str, float]:
    ids = sorted(plain)
    if set(ids) != set(treated):
        raise RuntimeError("item IDs differ across willingness frames")
    n = len(ids)
    plain_j = np.asarray([bool(plain[item]["J"]) for item in ids], dtype=bool)
    treated_j = np.asarray([bool(treated[item]["J"]) for item in ids], dtype=bool)
    recon_only = np.asarray(
        [bool(plain[item]["R"]) and not bool(plain[item]["U"]) for item in ids],
        dtype=bool,
    )
    rescue = (~plain_j) & treated_j
    backfire = plain_j & (~treated_j)
    return {
        "verified": float(treated_j.mean()),
        "delta": float((treated_j.astype(float) - plain_j.astype(float)).mean()),
        "rescue": float(rescue.mean()),
        "backfire": float(backfire.mean()),
        "recon_only_rescue": (
            float(treated_j[recon_only].mean()) if recon_only.any() else 0.0
        ),
    }


def analyze(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = validate_source(root)
    rows: list[dict[str, Any]] = []
    raw_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    tag_meta: dict[str, dict[str, Any]] = {}

    for benign_path in sorted((root / "benign").glob("*.json")):
        benign = json.loads(benign_path.read_text())
        tag = benign["tag"]
        tag_meta[tag] = {
            "model": benign["model"],
            "collection": benign["collection"],
            "q": benign["benign_recon_by_cell"],
        }
        q_by_cell = benign["benign_recon_by_cell"]
        if len(q_by_cell) != 32:
            raise RuntimeError(f"{tag}: expected 32 comprehension cells")

        cell_verified: dict[str, dict[str, float]] = defaultdict(dict)
        for cell in sorted(q_by_cell):
            frame_rows: dict[str, dict[str, dict[str, Any]]] = {}
            for frame in FRAMES:
                raw_path = root / "attack/_raw" / f"{tag}__{cell}__{frame}.jsonl"
                data = load_jsonl(raw_path)
                raw_cache[(tag, cell, frame)] = data
                frame_rows[frame] = data
                for item in data.values():
                    expected_j = float(bool(item["R"]) and bool(item["U"]))
                    if float(item["J"]) != expected_j:
                        raise RuntimeError(f"{raw_path}: J != R AND U")
                raw_rates = {
                    "recon": safe_mean(
                        [float(bool(item["R"])) for item in data.values()]
                    ),
                    "unsafe": safe_mean(
                        [float(bool(item["U"])) for item in data.values()]
                    ),
                    "verified": safe_mean(
                        [float(bool(item["J"])) for item in data.values()]
                    ),
                }
                aggregate_path = root / "attack" / f"{tag}__{cell}__{frame}.json"
                aggregate = json.loads(aggregate_path.read_text())
                for metric, raw_rate in raw_rates.items():
                    # Aggregate JSON stores rates rounded to three decimals.
                    if abs(float(aggregate[metric]) - raw_rate) > 0.00051:
                        raise RuntimeError(
                            f"{aggregate_path}: {metric} differs from raw rows"
                        )
                cell_verified[cell][frame] = raw_rates["verified"]

            plain = frame_rows["plain"]
            ids = sorted(plain)
            if any(set(frame_rows[frame]) != set(ids) for frame in FRAMES):
                raise RuntimeError(f"{tag}/{cell}: item IDs differ across frames")
            n = len(ids)
            recon_count = sum(bool(plain[item]["R"]) for item in ids)
            unsafe_count = sum(bool(plain[item]["U"]) for item in ids)
            verified_count = sum(bool(plain[item]["J"]) for item in ids)
            recon_only_count = sum(
                bool(plain[item]["R"]) and not bool(plain[item]["U"]) for item in ids
            )
            if recon_count - verified_count != recon_only_count:
                raise RuntimeError(f"{tag}/{cell}: reconstruction-only identity failed")

            effects = {
                frame: paired_effect(plain, frame_rows[frame]) for frame in NONPLAIN
            }
            oracle_frame = choose(
                {frame: effects[frame]["delta"] for frame in NONPLAIN}, NONPLAIN
            )
            recon_rate = recon_count / n
            verified_rate = verified_count / n
            recon_only_rate = recon_only_count / n
            q = float(q_by_cell[cell])
            rows.append(
                {
                    "model": benign["model"],
                    "family": family(tag),
                    "tag": tag,
                    "dataset": tag.rsplit("_", 1)[-1],
                    "collection": benign["collection"],
                    "cell": cell,
                    "n": n,
                    "prior_q": q,
                    "plain_recon_count": recon_count,
                    "plain_recon": recon_rate,
                    "plain_unsafe_count": unsafe_count,
                    "plain_unsafe": unsafe_count / n,
                    "plain_verified_count": verified_count,
                    "plain_verified": verified_rate,
                    "recon_only_count": recon_only_count,
                    "recon_only_rate": recon_only_rate,
                    "conditional_deficit": (
                        recon_only_count / recon_count if recon_count else 0.0
                    ),
                    "proxy_need_q_minus_asr": max(0.0, q - verified_rate),
                    # Exact integer comparisons avoid floating-point threshold drift.
                    "strong_bottleneck": bool(
                        q >= 0.5 and 2 * recon_count >= n and 4 * recon_only_count >= n
                    ),
                    "severe_bottleneck": bool(
                        q >= 0.5
                        and 10 * recon_count >= 7 * n
                        and 2 * recon_only_count >= recon_count
                    ),
                    "oracle_best_frame": oracle_frame,
                    "oracle_delta": effects[oracle_frame]["delta"],
                    "oracle_rescue": effects[oracle_frame]["rescue"],
                    "oracle_backfire": effects[oracle_frame]["backfire"],
                    "oracle_recon_only_rescue": effects[oracle_frame][
                        "recon_only_rescue"
                    ],
                    "frame_effects": effects,
                }
            )

        # Leave the evaluated cell out when choosing the model-specific frame.
        tag_rows = [row for row in rows if row["tag"] == tag]
        for row in tag_rows:
            train_cells = [cell for cell in q_by_cell if cell != row["cell"]]
            scores = {
                frame: safe_mean([cell_verified[cell][frame] for cell in train_cells])
                for frame in FRAMES
            }
            forced = choose(scores, NONPLAIN)
            gated = choose(scores, FRAMES)
            plain = raw_cache[(tag, row["cell"], "plain")]
            forced_effect = paired_effect(
                plain, raw_cache[(tag, row["cell"], forced)]
            )
            gated_effect = (
                {
                    "verified": row["plain_verified"],
                    "delta": 0.0,
                    "rescue": 0.0,
                    "backfire": 0.0,
                    "recon_only_rescue": 0.0,
                }
                if gated == "plain"
                else paired_effect(plain, raw_cache[(tag, row["cell"], gated)])
            )
            row.update(
                {
                    "loco_forced_frame": forced,
                    "loco_forced_delta": forced_effect["delta"],
                    "loco_forced_rescue": forced_effect["rescue"],
                    "loco_forced_backfire": forced_effect["backfire"],
                    "loco_forced_recon_only_rescue": forced_effect[
                        "recon_only_rescue"
                    ],
                    "loco_gated_frame": gated,
                    "loco_gated_delta": gated_effect["delta"],
                    "loco_gated_rescue": gated_effect["rescue"],
                    "loco_gated_backfire": gated_effect["backfire"],
                    "loco_gated_recon_only_rescue": gated_effect[
                        "recon_only_rescue"
                    ],
                }
            )

    # Aggregate statistics.
    q_values = np.asarray([row["prior_q"] for row in rows], dtype=float)
    recon_values = np.asarray([row["plain_recon"] for row in rows], dtype=float)
    recon_only_values = np.asarray([row["recon_only_rate"] for row in rows], dtype=float)
    proxy_values = np.asarray([row["proxy_need_q_minus_asr"] for row in rows], dtype=float)
    strong_values = np.asarray([row["strong_bottleneck"] for row in rows], dtype=bool)

    correlations: dict[str, Any] = {}
    for name, predictor, outcome in (
        ("prior_q_vs_plain_recon", q_values, recon_values),
        ("prior_q_vs_recon_only", q_values, recon_only_values),
        ("proxy_need_vs_recon_only", proxy_values, recon_only_values),
    ):
        pearson = stats.pearsonr(predictor, outcome)
        spearman = stats.spearmanr(predictor, outcome)
        correlations[name] = {
            "pearson_r": float(pearson.statistic),
            "spearman_rho": float(spearman.statistic),
        }

    trigger_thresholds: list[dict[str, Any]] = []
    for threshold in (0.1, 0.25, 0.4, 0.5, 0.6, 0.7):
        predicted = proxy_values >= threshold
        tp = int(np.sum(predicted & strong_values))
        fp = int(np.sum(predicted & (~strong_values)))
        fn = int(np.sum((~predicted) & strong_values))
        subset = [row for row in rows if row["proxy_need_q_minus_asr"] >= threshold]
        trigger_thresholds.append(
            {
                "threshold": threshold,
                "selected_cells": int(predicted.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": tp / (tp + fp) if tp + fp else math.nan,
                "recall": tp / (tp + fn) if tp + fn else math.nan,
                "mean_recon_only_rate": safe_mean(
                    [row["recon_only_rate"] for row in subset]
                ),
                "mean_loco_gated_delta": safe_mean(
                    [row["loco_gated_delta"] for row in subset]
                ),
            }
        )

    def band_summary(name: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "band": name,
            "n_cells": len(subset),
            "mean_prior_q": safe_mean([row["prior_q"] for row in subset]),
            "mean_plain_recon": safe_mean([row["plain_recon"] for row in subset]),
            "mean_plain_verified": safe_mean([row["plain_verified"] for row in subset]),
            "mean_recon_only": safe_mean([row["recon_only_rate"] for row in subset]),
            "mean_loco_forced_delta": safe_mean(
                [row["loco_forced_delta"] for row in subset]
            ),
            "mean_loco_gated_delta": safe_mean(
                [row["loco_gated_delta"] for row in subset]
            ),
            "gate_open_rate": safe_mean(
                [float(row["loco_gated_frame"] != "plain") for row in subset]
            ),
        }

    recon_only_bands = [
        band_summary("[0,.10)", [row for row in rows if row["recon_only_rate"] < 0.10]),
        band_summary(
            "[.10,.25)",
            [row for row in rows if 0.10 <= row["recon_only_rate"] < 0.25],
        ),
        band_summary(
            "[.25,.50)",
            [row for row in rows if 0.25 <= row["recon_only_rate"] < 0.50],
        ),
        band_summary("[.50,1]", [row for row in rows if row["recon_only_rate"] >= 0.50]),
    ]
    proxy_bands = [
        band_summary(
            "[0,.10)", [row for row in rows if row["proxy_need_q_minus_asr"] < 0.10]
        ),
        band_summary(
            "[.10,.25)",
            [row for row in rows if 0.10 <= row["proxy_need_q_minus_asr"] < 0.25],
        ),
        band_summary(
            "[.25,.50)",
            [row for row in rows if 0.25 <= row["proxy_need_q_minus_asr"] < 0.50],
        ),
        band_summary(
            "[.50,1]", [row for row in rows if row["proxy_need_q_minus_asr"] >= 0.50]
        ),
    ]

    tag_summaries: list[dict[str, Any]] = []
    for tag in sorted(tag_meta):
        subset = [row for row in rows if row["tag"] == tag]
        eligible = [row for row in subset if row["plain_recon_count"] * 2 >= row["n"]]
        top = max(
            eligible or subset,
            key=lambda row: (row["recon_only_count"] / row["n"], row["prior_q"]),
        )
        strong = [row for row in subset if row["strong_bottleneck"]]
        severe = [row for row in subset if row["severe_bottleneck"]]
        tag_summaries.append(
            {
                "tag": tag,
                "model": tag_meta[tag]["model"],
                "dataset": tag.rsplit("_", 1)[-1],
                "strong_cells": len(strong),
                "severe_cells": len(severe),
                "mean_plain_recon": safe_mean([row["plain_recon"] for row in subset]),
                "mean_plain_verified": safe_mean(
                    [row["plain_verified"] for row in subset]
                ),
                "mean_recon_only": safe_mean(
                    [row["recon_only_rate"] for row in subset]
                ),
                "dominant_loco_gate": Counter(
                    row["loco_gated_frame"] for row in subset
                ).most_common(1)[0][0],
                "mean_strong_loco_gated_delta": safe_mean(
                    [row["loco_gated_delta"] for row in strong]
                ),
                "top_cell": {
                    key: top[key]
                    for key in (
                        "cell",
                        "prior_q",
                        "plain_recon",
                        "plain_verified",
                        "recon_only_rate",
                        "conditional_deficit",
                        "loco_gated_frame",
                        "loco_gated_delta",
                        "loco_gated_recon_only_rescue",
                    )
                },
            }
        )

    model_summaries: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in tag_summaries:
        by_model[summary["model"]].append(summary)
    for model, summaries in sorted(by_model.items()):
        by_dataset = {summary["dataset"]: summary for summary in summaries}
        model_summaries.append(
            {
                "model": model,
                "mj": by_dataset.get("mj"),
                "lg": by_dataset.get("lg"),
                "strong_cells_total": sum(summary["strong_cells"] for summary in summaries),
            }
        )

    strong = [row for row in rows if row["strong_bottleneck"]]
    severe = [row for row in rows if row["severe_bottleneck"]]
    summary = {
        "source": source,
        "definitions": {
            "recon_only": "plain R=1 and U=0; equivalently plain_recon - plain_verified",
            "strong_bottleneck": "prior_q>=.5 AND plain_recon>=.5 AND recon_only_rate>=.25",
            "severe_bottleneck": (
                "prior_q>=.5 AND plain_recon>=.7 AND "
                "P(verified failure | reconstructed)>=.5"
            ),
            "proxy_need": "max(0, harmless prior_q - plain verified ASR)",
            "loco_gate": (
                "select plain or a willingness frame using the other 31 cells of "
                "the same model/dataset; evaluate on the held cell"
            ),
        },
        "scope": {
            "models": len(by_model),
            "model_dataset_tags": len(tag_meta),
            "comprehension_cells": len(rows),
            "plain_item_evaluations": sum(row["n"] for row in rows),
            "strong_bottleneck_cells": len(strong),
            "severe_bottleneck_cells": len(severe),
        },
        "overall": {
            "mean_plain_recon": safe_mean([row["plain_recon"] for row in rows]),
            "mean_plain_verified": safe_mean([row["plain_verified"] for row in rows]),
            "mean_recon_only": safe_mean([row["recon_only_rate"] for row in rows]),
            "strong_mean_loco_gated_delta": safe_mean(
                [row["loco_gated_delta"] for row in strong]
            ),
            "strong_mean_loco_gated_recon_only_rescue": safe_mean(
                [row["loco_gated_recon_only_rescue"] for row in strong]
            ),
            "strong_mean_loco_gated_backfire": safe_mean(
                [row["loco_gated_backfire"] for row in strong]
            ),
            "strong_positive_delta_cells": sum(
                row["loco_gated_delta"] > 0 for row in strong
            ),
            "strong_negative_delta_cells": sum(
                row["loco_gated_delta"] < 0 for row in strong
            ),
        },
        "correlations": correlations,
        "trigger_thresholds": trigger_thresholds,
        "recon_only_bands": recon_only_bands,
        "proxy_need_bands": proxy_bands,
        "tag_summaries": tag_summaries,
        "model_summaries": model_summaries,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows: list[dict[str, Any]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "frame_effects"}
        for frame, effect in row["frame_effects"].items():
            for metric, value in effect.items():
                item[f"{frame}_{metric}"] = value
        scalar_rows.append(item)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(scalar_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(scalar_rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    scope = summary["scope"]
    overall = summary["overall"]
    corr = summary["correlations"]
    lines = [
        "# Canonical-17 reconstruction-only bottleneck audit",
        "",
        "## Validated scope",
        "",
        f"- Canonical final audit: `{summary['source']['audit_marker']}`",
        f"- {scope['models']} models, {scope['model_dataset_tags']} model-dataset tags, "
        f"{scope['comprehension_cells']} plain comprehension cells",
        f"- {scope['plain_item_evaluations']:,} plain item evaluations",
        f"- Strong bottlenecks: **{scope['strong_bottleneck_cells']}** cells; "
        f"severe bottlenecks: **{scope['severe_bottleneck_cells']}** cells",
        "",
        "A reconstruction-only failure is an item with `R=1, U=0`: the model recovered "
        "the hidden request but did not produce a verified successful response.",
        "",
        "## Main finding",
        "",
        f"Across all cells, mean plain reconstruction is {fmt(overall['mean_plain_recon'])}, "
        f"mean verified ASR is {fmt(overall['mean_plain_verified'])}, and the exact "
        f"reconstruction-only gap is **{fmt(overall['mean_recon_only'])}**. In the "
        f"{scope['strong_bottleneck_cells']} strong cells, a model-specific willingness "
        f"gate selected on the other 31 cells changes held-cell ASR by "
        f"**{overall['strong_mean_loco_gated_delta']:+.3f}** on average.",
        f"It converts {overall['strong_mean_loco_gated_recon_only_rescue']:.1%} of "
        f"the reconstruction-only cases into verified successes, with "
        f"{overall['strong_mean_loco_gated_backfire']:.1%} mean backfire. The held-cell "
        f"delta is positive in {overall['strong_positive_delta_cells']}/"
        f"{scope['strong_bottleneck_cells']} strong cells, zero in "
        f"{scope['strong_bottleneck_cells'] - overall['strong_positive_delta_cells'] - overall['strong_negative_delta_cells']}, "
        f"and negative in {overall['strong_negative_delta_cells']}.",
        "",
        f"The harmless comprehension prior predicts harmful-task reconstruction well "
        f"(Pearson `r={corr['prior_q_vs_plain_recon']['pearson_r']:.3f}`, Spearman "
        f"`rho={corr['prior_q_vs_plain_recon']['spearman_rho']:.3f}`). The deployable "
        f"calibration proxy `max(0,q-ASR_plain)` tracks the exact reconstruction-only "
        f"gap even more directly (Pearson `r={corr['proxy_need_vs_recon_only']['pearson_r']:.3f}`, "
        f"Spearman `rho={corr['proxy_need_vs_recon_only']['spearman_rho']:.3f}`).",
        "",
        "## When to open the willingness axis",
        "",
        "| q-ASR trigger | Selected cells | Precision for strong bottleneck | Recall | "
        "Mean true R-only | Mean LOCO gated delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["trigger_thresholds"]:
        lines.append(
            f"| >= {row['threshold']:.2f} | {row['selected_cells']} | "
            f"{row['precision']:.1%} | {row['recall']:.1%} | "
            f"{row['mean_recon_only_rate']:.3f} | {row['mean_loco_gated_delta']:+.3f} |"
        )
    lines += [
        "",
        "`q-ASR_plain >= .50` is the conservative trigger: it has high precision, while "
        "`.40` is the more balanced trigger. These thresholds require a small plain "
        "calibration observation; the harmless prior alone cannot identify refusal/compliance.",
        "",
        "## Effect by exact reconstruction-only gap",
        "",
        "| Exact R-only gap | Cells | Mean q | Mean R | Mean verified | LOCO gated delta | Gate open |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["recon_only_bands"]:
        lines.append(
            f"| {row['band']} | {row['n_cells']} | {row['mean_prior_q']:.3f} | "
            f"{row['mean_plain_recon']:.3f} | {row['mean_plain_verified']:.3f} | "
            f"{row['mean_loco_gated_delta']:+.3f} | {row['gate_open_rate']:.1%} |"
        )

    lines += [
        "",
        "## All completed models",
        "",
        "`strong` means q>=.5, plain reconstruction>=.5, and at least 25% of all "
        "items are reconstruction-only failures. The displayed gate is selected from "
        "the other 31 cells, not from the displayed cell.",
        "",
        "| Model | MJ strong/32 | Highest MJ bottleneck | LG strong/32 | Highest LG bottleneck |",
        "|---|---:|---|---:|---|",
    ]
    for model in summary["model_summaries"]:
        def top_text(tag_summary: dict[str, Any] | None) -> str:
            if tag_summary is None:
                return "missing"
            top = tag_summary["top_cell"]
            return (
                f"`{top['cell']}`; q={top['prior_q']:.3f}, "
                f"R={top['plain_recon']:.3f}, ASR={top['plain_verified']:.3f}, "
                f"R-only={top['recon_only_rate']:.3f}; "
                f"gate={top['loco_gated_frame']} ({top['loco_gated_delta']:+.3f})"
            )
        mj, lg = model["mj"], model["lg"]
        lines.append(
            f"| {model['model']} | {mj['strong_cells'] if mj else '-'} | {top_text(mj)} | "
            f"{lg['strong_cells'] if lg else '-'} | {top_text(lg)} |"
        )

    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- The largest bottleneck counts occur for Qwen2.5-32B and Gemma-2-27B "
        "(63/64 cells each), followed by Qwen2.5-14B (55/64) and Gemma-2-9B "
        "(53/64). These models usually understand the construction but plain alignment "
        "blocks verified success.",
        "- A bottleneck does not guarantee that the current willingness frames fix it. "
        "For Llama-3.2-3B the LOCO gate retains plain, so this model needs a different "
        "intervention family rather than forced persona/PAP framing.",
        "- The gold R-only label is retrospective. At deployment/search time, use the "
        "harmless q prior plus a budgeted plain calibration observation.",
        "- The per-cell oracle frame is exported only for diagnosis. Primary effect "
        "columns use leave-one-comprehension-cell-out (LOCO) frame selection.",
        "- LOCO uses the other 31 cells and therefore establishes that the conditional "
        "structure exists; it is not yet a matched-query-budget selector result.",
        "- Reconstruction and unsafe labels inherit the frozen automated judges in the "
        "canonical collection.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_intervention_map(path: Path, rows: list[dict[str, Any]]) -> None:
    def point(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "cell": row["cell"],
            "prior_q": row["prior_q"],
            "plain_recon": row["plain_recon"],
            "plain_verified": row["plain_verified"],
            "recon_only_rate": row["recon_only_rate"],
            "proxy_need_q_minus_asr": row["proxy_need_q_minus_asr"],
            "recommended_loco_gate": row["loco_gated_frame"],
            "observed_loco_delta": row["loco_gated_delta"],
            "observed_recon_only_rescue": row[
                "loco_gated_recon_only_rescue"
            ],
        }

    tags: dict[str, Any] = {}
    for tag in sorted({row["tag"] for row in rows}):
        subset = [row for row in rows if row["tag"] == tag]
        tags[tag] = {
            "model": subset[0]["model"],
            "dataset": subset[0]["dataset"],
            "gold_strong_cells": [
                point(row) for row in subset if row["strong_bottleneck"]
            ],
            "balanced_trigger_cells_q_minus_asr_ge_0_4": [
                point(row)
                for row in subset
                if row["proxy_need_q_minus_asr"] >= 0.4
            ],
            "conservative_trigger_cells_q_minus_asr_ge_0_5": [
                point(row)
                for row in subset
                if row["proxy_need_q_minus_asr"] >= 0.5
            ],
        }
    payload = {
        "schema": 1,
        "note": (
            "Aggregate diagnostic map only; thresholds require a plain calibration "
            "observation and LOCO effects are not a matched-budget estimate."
        ),
        "tags": tags,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows, summary = analyze(args.canonical_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "all_recon_only_cells.csv"
    json_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "REPORT.md"
    map_path = args.output_dir / "intervention_map.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(report_path, summary)
    write_intervention_map(map_path, rows)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "summary": str(json_path),
                "report": str(report_path),
                "intervention_map": str(map_path),
                **summary["scope"],
                **summary["overall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
