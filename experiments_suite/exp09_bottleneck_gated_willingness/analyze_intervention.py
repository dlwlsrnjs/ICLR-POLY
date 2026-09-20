#!/usr/bin/env python3
"""Measure when a model-specific willingness-frame intervention helps.

The 160-arm space is a 32-cell comprehension design crossed with five
willingness frames.  For every model/dataset tag and comprehension cell we:

1. Estimate the model-specific intervention frame using the other 31 cells.
2. Define the observed willingness bottleneck as recon(plain)-verified(plain).
3. Measure verified-ASR uplift, paired rescue, and paired backfire on the held
   cell.

The leave-one-cell-out frame choice prevents using the evaluated cell to choose
its own treatment.  This is a retrospective diagnostic, not a zero-query
deployment rule: the observed bottleneck contains attack-evaluation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


FRAMES = ("plain", "persona", "fiction", "pap", "persona+fiction")
INTERVENTION_FRAMES = tuple(frame for frame in FRAMES if frame != "plain")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo = here.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack-dir",
        type=Path,
        default=repo / "experiments_suite/exp02_panel_collect/results/attack",
    )
    parser.add_argument(
        "--benign-dir",
        type=Path,
        default=repo / "experiments_suite/exp02_panel_collect/results/benign",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=repo / "experiments_suite/exp07_panel17_selector_replay/results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here.parent / "results/bottleneck_willingness_intervention.json",
    )
    parser.add_argument(
        "--include-rows",
        action="store_true",
        help="Include all cell/fold rows in the JSON instead of summary tables only.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def json_ready(value: Any) -> Any:
    """Convert non-finite floats to JSON null for standards-compliant output."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def correlation(rows: list[dict[str, Any]], x: str, y: str) -> dict[str, float]:
    pairs = np.asarray(
        [(row[x], row[y]) for row in rows if math.isfinite(row[x]) and math.isfinite(row[y])],
        dtype=float,
    )
    pearson = stats.pearsonr(pairs[:, 0], pairs[:, 1])
    spearman = stats.spearmanr(pairs[:, 0], pairs[:, 1])
    return {
        "n": int(len(pairs)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = finite([row[key] for row in rows])
    return float(values.mean()) if len(values) else math.nan


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n_cells": len(rows),
        "plain_verified": mean(rows, "plain_verified"),
        "plain_recon": mean(rows, "plain_recon"),
        "selected_verified": mean(rows, "selected_verified"),
        "delta_verified": mean(rows, "delta_verified"),
        "need_gap": mean(rows, "need_gap"),
        "conditional_deficit": mean(rows, "conditional_deficit"),
        "paired_rescue": mean(rows, "paired_rescue"),
        "paired_backfire": mean(rows, "paired_backfire"),
        "paired_net": mean(rows, "paired_net"),
    }


def load_attack(attack_dir: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], list[Path]]:
    records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    paths = sorted(attack_dir.glob("*.json"))
    for path in paths:
        payload = json.loads(path.read_text())
        method = payload.get("method", "")
        if "__" not in method:
            continue
        cell, frame = method.rsplit("__", 1)
        if frame not in FRAMES:
            continue
        records[payload["tag"]][method] = payload
    return records, paths


def load_matrices(matrix_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    matrices: dict[str, Any] = {}
    paths: list[Path] = []
    for dataset in ("mj", "lg"):
        path = matrix_dir / f"{dataset}_item_matrix.json"
        matrices[dataset] = json.loads(path.read_text())
        paths.append(path)
    return matrices, paths


def choose_frame_loo(methods: dict[str, dict[str, Any]], held_cell: str, cells: list[str]) -> str:
    scores: dict[str, float] = {}
    train_cells = [cell for cell in cells if cell != held_cell]
    for frame in INTERVENTION_FRAMES:
        scores[frame] = float(
            np.mean([methods[f"{cell}__{frame}"]["verified"] for cell in train_cells])
        )
    # Stable tie breaking follows FRAMES order.
    return max(INTERVENTION_FRAMES, key=lambda frame: (scores[frame], -FRAMES.index(frame)))


def paired_rates(
    matrices: dict[str, Any], tag: str, plain_method: str, selected_method: str
) -> tuple[float, float, float]:
    dataset = tag.rsplit("_", 1)[-1]
    matrix = matrices[dataset]["matrix"][tag]
    plain = np.asarray(matrix[plain_method], dtype=float)
    selected = np.asarray(matrix[selected_method], dtype=float)
    valid = np.isfinite(plain) & np.isfinite(selected)
    plain, selected = plain[valid], selected[valid]
    rescue = float(np.mean((plain == 0) & (selected == 1)))
    backfire = float(np.mean((plain == 1) & (selected == 0)))
    return rescue, backfire, rescue - backfire


def summarize_crossfit(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n_fold_cells": len(rows),
        "benign_comprehension_q": mean(rows, "benign_comprehension_q"),
        "estimated_need": mean(rows, "estimated_need"),
        "plain_verified_calibration": mean(rows, "plain_verified_calibration"),
        "heldout_plain_verified": mean(rows, "heldout_plain_verified"),
        "heldout_selected_verified": mean(rows, "heldout_selected_verified"),
        "heldout_delta_verified": mean(rows, "heldout_delta_verified"),
        "heldout_rescue": mean(rows, "heldout_rescue"),
        "heldout_backfire": mean(rows, "heldout_backfire"),
        "heldout_net": mean(rows, "heldout_net"),
        "gate_open_rate": mean(rows, "gate_open"),
        "heldout_gated_selected_verified": mean(
            rows, "heldout_gated_selected_verified"
        ),
        "heldout_gated_delta_verified": mean(rows, "heldout_gated_delta_verified"),
        "heldout_gated_rescue": mean(rows, "heldout_gated_rescue"),
        "heldout_gated_backfire": mean(rows, "heldout_gated_backfire"),
        "heldout_gated_net": mean(rows, "heldout_gated_net"),
    }


def crossfit_validation(
    attacks: dict[str, dict[str, dict[str, Any]]],
    matrices: dict[str, Any],
    benign_dir: Path,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Predict intervention value on items disjoint from gate estimation."""
    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for tag, methods in sorted(attacks.items()):
        benign_path = benign_dir / f"{tag}.json"
        if not benign_path.exists():
            skipped[tag] = "missing benign comprehension prior"
            continue
        dataset = tag.rsplit("_", 1)[-1]
        if tag not in matrices[dataset]["matrix"]:
            skipped[tag] = "missing item matrix"
            continue
        matrix = matrices[dataset]["matrix"][tag]
        q_by_cell = json.loads(benign_path.read_text())["benign_recon_by_cell"]
        cells = sorted(q_by_cell)
        if len(cells) != 32 or any(
            f"{cell}__{frame}" not in methods or f"{cell}__{frame}" not in matrix
            for cell in cells
            for frame in FRAMES
        ):
            skipped[tag] = "incomplete 32x5 factorial"
            continue

        n_items = len(matrices[dataset]["items"])
        for evaluation_parity in (0, 1):
            evaluation_indices = np.asarray(
                [index for index in range(n_items) if index % 2 == evaluation_parity]
            )
            calibration_indices = np.asarray(
                [index for index in range(n_items) if index % 2 != evaluation_parity]
            )
            for held_cell in cells:
                train_cells = [cell for cell in cells if cell != held_cell]
                frame_scores: dict[str, float] = {}
                for frame in FRAMES:
                    values = np.concatenate(
                        [
                            np.asarray(matrix[f"{cell}__{frame}"], dtype=float)[
                                calibration_indices
                            ]
                            for cell in train_cells
                        ]
                    )
                    frame_scores[frame] = float(np.nanmean(values))
                selected_frame = max(
                    INTERVENTION_FRAMES,
                    key=lambda frame: (frame_scores[frame], -FRAMES.index(frame)),
                )
                gated_frame = max(
                    FRAMES,
                    key=lambda frame: (frame_scores[frame], -FRAMES.index(frame)),
                )

                plain_method = f"{held_cell}__plain"
                selected_method = f"{held_cell}__{selected_frame}"
                gated_method = f"{held_cell}__{gated_frame}"
                calibration_plain = float(
                    np.nanmean(
                        np.asarray(matrix[plain_method], dtype=float)[calibration_indices]
                    )
                )
                q = float(q_by_cell[held_cell])
                estimated_need = max(0.0, q - calibration_plain)

                heldout_plain = np.asarray(matrix[plain_method], dtype=float)[
                    evaluation_indices
                ]
                heldout_selected = np.asarray(matrix[selected_method], dtype=float)[
                    evaluation_indices
                ]
                heldout_gated = np.asarray(matrix[gated_method], dtype=float)[
                    evaluation_indices
                ]
                valid = (
                    np.isfinite(heldout_plain)
                    & np.isfinite(heldout_selected)
                    & np.isfinite(heldout_gated)
                )
                heldout_plain = heldout_plain[valid]
                heldout_selected = heldout_selected[valid]
                heldout_gated = heldout_gated[valid]
                rescue = float(
                    np.mean((heldout_plain == 0) & (heldout_selected == 1))
                )
                backfire = float(
                    np.mean((heldout_plain == 1) & (heldout_selected == 0))
                )
                gated_rescue = float(
                    np.mean((heldout_plain == 0) & (heldout_gated == 1))
                )
                gated_backfire = float(
                    np.mean((heldout_plain == 1) & (heldout_gated == 0))
                )
                rows.append(
                    {
                        "tag": tag,
                        "dataset": dataset,
                        "cell": held_cell,
                        "evaluation_parity": evaluation_parity,
                        "selected_frame": selected_frame,
                        "gated_frame": gated_frame,
                        "gate_open": float(gated_frame != "plain"),
                        "benign_comprehension_q": q,
                        "plain_verified_calibration": calibration_plain,
                        "estimated_need": estimated_need,
                        "heldout_plain_verified": float(np.mean(heldout_plain)),
                        "heldout_selected_verified": float(np.mean(heldout_selected)),
                        "heldout_delta_verified": float(
                            np.mean(heldout_selected - heldout_plain)
                        ),
                        "heldout_rescue": rescue,
                        "heldout_backfire": backfire,
                        "heldout_net": rescue - backfire,
                        "heldout_gated_selected_verified": float(
                            np.mean(heldout_gated)
                        ),
                        "heldout_gated_delta_verified": float(
                            np.mean(heldout_gated - heldout_plain)
                        ),
                        "heldout_gated_rescue": gated_rescue,
                        "heldout_gated_backfire": gated_backfire,
                        "heldout_gated_net": gated_rescue - gated_backfire,
                    }
                )

    need_edges = np.quantile(
        [row["estimated_need"] for row in rows], [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    quartiles: list[dict[str, Any]] = []
    for index in range(4):
        low, high = float(need_edges[index]), float(need_edges[index + 1])
        if index < 3:
            subset = [row for row in rows if low <= row["estimated_need"] < high]
        else:
            subset = [row for row in rows if low <= row["estimated_need"] <= high]
        quartiles.append(
            {
                "quartile": index + 1,
                "low": low,
                "high": high,
                **summarize_crossfit(subset),
            }
        )

    q_median = float(np.median([row["benign_comprehension_q"] for row in rows]))
    need_median = float(np.median([row["estimated_need"] for row in rows]))
    zones: list[dict[str, Any]] = []
    for q_high in (False, True):
        for need_high in (False, True):
            subset = [
                row
                for row in rows
                if (row["benign_comprehension_q"] >= q_median) == q_high
                and (row["estimated_need"] >= need_median) == need_high
            ]
            zones.append(
                {
                    "zone": f"{'high' if q_high else 'low'}_benign_comprehension__"
                    f"{'high' if need_high else 'low'}_estimated_need",
                    **summarize_crossfit(subset),
                }
            )

    per_tag: list[dict[str, Any]] = []
    for tag in sorted({row["tag"] for row in rows}):
        subset = [row for row in rows if row["tag"] == tag]
        frame_counts = {
            frame: sum(row["selected_frame"] == frame for row in subset)
            for frame in INTERVENTION_FRAMES
        }
        gated_frame_counts = {
            frame: sum(row["gated_frame"] == frame for row in subset)
            for frame in FRAMES
        }
        per_tag.append(
            {
                "tag": tag,
                "selected_frame_counts": frame_counts,
                "gated_frame_counts": gated_frame_counts,
                **summarize_crossfit(subset),
                "estimated_need_vs_heldout_delta": correlation(
                    subset, "estimated_need", "heldout_delta_verified"
                ),
            }
        )

    result = {
        "design": (
            "two-fold item cross-fit; calibration items estimate need and choose "
            "a model-specific frame on the other 31 cells; disjoint items measure effect"
        ),
        "estimated_need_definition": (
            "max(0, benign_comprehension_q - plain_verified_on_calibration_items)"
        ),
        "gate_definition": (
            "choose the best of plain and four willingness frames on calibration "
            "items from the other 31 cells; evaluate on the held cell and disjoint items"
        ),
        "scope": {
            "n_model_dataset_tags": len({row["tag"] for row in rows}),
            "n_fold_cells": len(rows),
            "n_unique_cells": len({(row["tag"], row["cell"]) for row in rows}),
            "skipped": skipped,
        },
        "overall": summarize_crossfit(rows),
        "correlations": {
            "estimated_need_vs_heldout_delta": correlation(
                rows, "estimated_need", "heldout_delta_verified"
            ),
            "benign_comprehension_vs_heldout_delta": correlation(
                rows, "benign_comprehension_q", "heldout_delta_verified"
            ),
        },
        "need_quartile_edges": [float(value) for value in need_edges],
        "need_quartiles": quartiles,
        "zone_thresholds": {
            "benign_comprehension_q_median": q_median,
            "estimated_need_median": need_median,
        },
        "zones": zones,
        "per_tag": per_tag,
        "inference_note": (
            "Correlation p-values are unclustered diagnostics; paper-level inference "
            "must cluster/bootstrap by model and comprehension cell."
        ),
    }
    if include_rows:
        result["rows"] = rows
    return result


def main() -> None:
    args = parse_args()
    attacks, attack_paths = load_attack(args.attack_dir)
    matrices, matrix_paths = load_matrices(args.matrix_dir)

    rows: list[dict[str, Any]] = []
    benign_paths: list[Path] = []
    skipped: dict[str, str] = {}

    for tag, methods in sorted(attacks.items()):
        cells = sorted(
            method[: -len("__plain")]
            for method in methods
            if method.endswith("__plain") and method.count("__") == 1
        )
        if len(cells) != 32 or any(
            f"{cell}__{frame}" not in methods for cell in cells for frame in FRAMES
        ):
            skipped[tag] = "incomplete 32x5 factorial"
            continue
        dataset = tag.rsplit("_", 1)[-1]
        if tag not in matrices[dataset]["matrix"]:
            skipped[tag] = "missing item matrix"
            continue
        benign_path = args.benign_dir / f"{tag}.json"
        if benign_path.exists():
            benign = json.loads(benign_path.read_text())
            benign_paths.append(benign_path)
            q_by_cell = benign["benign_recon_by_cell"]
        else:
            q_by_cell = {}

        for cell in cells:
            selected_frame = choose_frame_loo(methods, cell, cells)
            plain_method = f"{cell}__plain"
            selected_method = f"{cell}__{selected_frame}"
            plain = methods[plain_method]
            selected = methods[selected_method]
            recon = float(plain["recon"])
            verified = float(plain["verified"])
            need_gap = max(0.0, recon - verified)
            conditional_deficit = (1.0 - verified / recon) if recon > 0 else math.nan
            rescue, backfire, net = paired_rates(
                matrices, tag, plain_method, selected_method
            )
            rows.append(
                {
                    "tag": tag,
                    "dataset": dataset,
                    "cell": cell,
                    "selected_frame_loo": selected_frame,
                    "benign_comprehension_q": float(q_by_cell.get(cell, math.nan)),
                    "plain_recon": recon,
                    "plain_verified": verified,
                    "need_gap": need_gap,
                    "conditional_deficit": conditional_deficit,
                    "selected_verified": float(selected["verified"]),
                    "delta_verified": float(selected["verified"] - verified),
                    "paired_rescue": rescue,
                    "paired_backfire": backfire,
                    "paired_net": net,
                }
            )

    # Quartile edges are reported explicitly so the grouping is reproducible.
    need_values = np.asarray([row["need_gap"] for row in rows], dtype=float)
    quartile_edges = np.quantile(need_values, [0.0, 0.25, 0.5, 0.75, 1.0])
    need_quartiles: list[dict[str, Any]] = []
    for index in range(4):
        low, high = float(quartile_edges[index]), float(quartile_edges[index + 1])
        if index < 3:
            subset = [row for row in rows if low <= row["need_gap"] < high]
        else:
            subset = [row for row in rows if low <= row["need_gap"] <= high]
        need_quartiles.append(
            {"quartile": index + 1, "low": low, "high": high, **summarize(subset)}
        )

    recon_median = float(np.median([row["plain_recon"] for row in rows]))
    deficit_median = float(
        np.nanmedian([row["conditional_deficit"] for row in rows])
    )
    zones: list[dict[str, Any]] = []
    for recon_high in (False, True):
        for deficit_high in (False, True):
            subset = [
                row
                for row in rows
                if (row["plain_recon"] >= recon_median) == recon_high
                and math.isfinite(row["conditional_deficit"])
                and (row["conditional_deficit"] >= deficit_median) == deficit_high
            ]
            zones.append(
                {
                    "zone": f"{'high' if recon_high else 'low'}_comprehension__"
                    f"{'high' if deficit_high else 'low'}_willingness_need",
                    **summarize(subset),
                }
            )

    per_tag: list[dict[str, Any]] = []
    for tag in sorted({row["tag"] for row in rows}):
        subset = [row for row in rows if row["tag"] == tag]
        tag_median = float(np.median([row["need_gap"] for row in subset]))
        high = [row for row in subset if row["need_gap"] >= tag_median]
        low = [row for row in subset if row["need_gap"] < tag_median]
        frame_counts = {
            frame: sum(row["selected_frame_loo"] == frame for row in subset)
            for frame in INTERVENTION_FRAMES
        }
        per_tag.append(
            {
                "tag": tag,
                "selected_frame_counts": frame_counts,
                **summarize(subset),
                "within_tag_need_delta": correlation(subset, "need_gap", "delta_verified"),
                "high_need_delta_verified": mean(high, "delta_verified"),
                "low_need_delta_verified": mean(low, "delta_verified"),
            }
        )

    used_paths = sorted(set(attack_paths + benign_paths + matrix_paths))
    payload = {
        "analysis": "leave-one-comprehension-cell-out model-specific willingness intervention",
        "definitions": {
            "need_gap": "max(0, plain_recon - plain_verified)",
            "conditional_deficit": "1 - plain_verified/plain_recon (plain_recon > 0)",
            "delta_verified": "selected_verified - plain_verified",
            "paired_rescue": "P(plain item=0 and selected-frame item=1)",
            "paired_backfire": "P(plain item=1 and selected-frame item=0)",
            "frame_selection": "highest mean verified ASR on the other 31 cells, among four non-plain frames",
        },
        "scope": {
            "n_model_dataset_tags": len({row["tag"] for row in rows}),
            "n_cells": len(rows),
            "frames": list(FRAMES),
            "skipped": skipped,
            "missing_benign_prior_tags": sorted(
                tag
                for tag in {row["tag"] for row in rows}
                if not (args.benign_dir / f"{tag}.json").exists()
            ),
        },
        "overall": summarize(rows),
        "correlations": {
            "need_gap_vs_delta": correlation(rows, "need_gap", "delta_verified"),
            "conditional_deficit_vs_delta": correlation(
                rows, "conditional_deficit", "delta_verified"
            ),
            "benign_comprehension_vs_delta": correlation(
                rows, "benign_comprehension_q", "delta_verified"
            ),
            "plain_recon_vs_delta": correlation(rows, "plain_recon", "delta_verified"),
        },
        "need_quartile_edges": [float(value) for value in quartile_edges],
        "need_quartiles": need_quartiles,
        "zone_thresholds": {
            "plain_recon_median": recon_median,
            "conditional_deficit_median": deficit_median,
        },
        "zones": zones,
        "per_tag": per_tag,
        "crossfit_predictive_validation": crossfit_validation(
            attacks, matrices, args.benign_dir, include_rows=args.include_rows
        ),
        "provenance": {
            "n_source_files": len(used_paths),
            "source_manifest_sha256": manifest_sha256(used_paths),
            "source_roots": [
                str(args.attack_dir),
                str(args.benign_dir),
                str(args.matrix_dir),
            ],
        },
        "limitations": [
            "Observed need_gap uses attack-evaluation outcomes and is diagnostic, not a zero-query deployment feature.",
            "need_gap and delta_verified share the plain term; use crossfit_predictive_validation for the less biased predictive check.",
            "Frame selection is leave-one-cell-out but still within the same model and dataset.",
            "Verified labels and reconstruction labels inherit the limitations of their automated judges.",
            "Causal claims require a preregistered prospective evaluation with fixed gates and budgets.",
        ],
    }
    if args.include_rows:
        payload["rows"] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"output": str(args.output), **payload["scope"], **payload["overall"]}, indent=2))


if __name__ == "__main__":
    main()
