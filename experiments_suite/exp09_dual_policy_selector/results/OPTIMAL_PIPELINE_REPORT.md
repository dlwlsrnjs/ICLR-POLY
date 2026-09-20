# Frozen separate-axis pipeline result

This is an offline replay over the already generated MJ/LG 160-setting grids. Candidate policies were ranked on selection, shortlisted on validation, frozen, and evaluated once on item-disjoint held-out test splits.

## Frozen policies

| Dataset | Frozen policy | Test verified ASR | Oracle recovery | Budget AUC | Mean harmful pulls | Mean harmless probes |
|---|---|---:|---:|---:|---:|---:|
| MJ | `P5_response_state_dynamic` | 81.99% | 83.52% | 0.6797 | 4.25 | 36.23 |
| LG | `P9_full_evidence_soft_knn` | 81.82% | 88.44% | 0.7244 | 3.72 | 38.70 |

## One-policy deployment choice

Equal-dataset selection/validation chose `P5_response_state_dynamic` without test feedback. Its per-dataset held-out results are stored in `selected_pipeline_global.json`.

| Dataset | Verified ASR | Oracle recovery | Budget AUC | Mean harmful pulls | Harmless probes |
|---|---:|---:|---:|---:|---:|
| MJ | 82.35% | 83.90% | 0.6777 | 4.28 | 19.01 |
| LG | 80.75% | 87.28% | 0.7132 | 3.84 | 19.25 |

| Dataset | Global policy vs comparator | Δ verified ASR | Δ budget AUC | Δ harmful pulls |
|---|---|---:|---:|---:|
| MJ | B1 joint | -2.21 pp | +0.0227 | -0.28 |
| MJ | B2 separate | -6.25 pp | +0.0158 | -0.22 |
| LG | B1 joint | -2.14 pp | +0.0272 | -0.34 |
| LG | B2 separate | -3.74 pp | +0.0412 | -0.51 |

## Dataset-specific-policy held-out comparisons

B1 is the conventional prior-free joint 160-arm harmful-only GP-UCB baseline. B2 is the stronger prior-free harmful-only baseline that keeps the two axes separate.

| Dataset | Comparator | Δ verified ASR | Δ budget AUC | Δ mean harmful pulls |
|---|---|---:|---:|---:|
| MJ | B1 | -2.57 pp | +0.0247 | -0.31 |
| MJ | B2 | -6.62 pp | +0.0178 | -0.25 |
| LG | B1 | -1.07 pp | +0.0383 | -0.47 |
| LG | B2 | -2.67 pp | +0.0524 | -0.64 |

## Interpretation

- The selected objective is harmful-query efficiency, not test-set maximum final ASR. That is why B2 can finish slightly higher at budget 12 while the frozen method has better early-budget AUC and fewer harmful pulls.
- MJ selected `P5_response_state_dynamic`.
- LG selected `P9_full_evidence_soft_knn`.
- `P9_full_evidence_soft_knn`, when selected, uses all valid successes and failures, no hard cluster and no explicit OOD gate; `P7/P8` are calibrated OOD/soft-neighbour ablations.
- Understanding and willingness embeddings, clusters, posteriors and acquisition functions remain separate. The renderer merely accepts one setting selected by each axis.

## Limits

- Exact WildGuard refusal state is available for 70.59% of MJ rows and 76.47% of LG rows. Rows without it retain the verified-success observation but cannot contribute an exact partial-versus-full-refusal transition.
- The held-out split is small (MJ 16 items, LG 11 items, crossed with 17 models). Report confidence intervals and repeat on a fresh live set before a paper-level superiority claim.
- Harmless probes are free only in the harmful-risk budget. They are still reported as real requests and must be included in token and wall-clock cost tables.

## Reproduction

```bash
/home/ljk98/POLY/.venv-gen/bin/python code/build_separate_axis_clusters.py
/home/ljk98/POLY/.venv-gen/bin/python code/search_separate_axis_pipeline.py
/home/ljk98/POLY/.venv-gen/bin/python code/run_separate_axis_ablation.py --seeds 20 --budget 12
/home/ljk98/POLY/.venv-gen/bin/python code/audit_separate_axis_pipeline.py
```
