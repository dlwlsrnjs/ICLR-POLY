# Willingness-axis PRIOR — what it is, how it's recorded, how it's used

This folder builds the **offline, benign willingness-axis prior** ("willingness fingerprint") that the
transfer/stealth selector relies on. This document is the record of *how* that prior is produced so it is
reproducible and auditable by a reviewer. Recorder: [`willingness_prior.py`](willingness_prior.py).

## 1. Why this exists (the saturation problem)

The willingness axis = {plain, persona, fiction, pap, …} frames that push a model from "won't" to "will".
On helpful models a **plain benign probe saturates**: nonrefusal ≈ 1.0 for *every* frame, so the axis
looks dead and we cannot read which frame a given target prefers **without sending harmful queries**.

Fix: probe the willingness frames on **over-refusal datasets** — prompts that are *harmless but look
harmful*. These de-saturate the axis: frames now split (some engage, some are trained-refused), and that
per-frame split **transfers to the model's harmful comply ordering**. So we learn the axis on benign data
offline and deploy it on a blind target with only a benign probe — no harmful query to the target.

Dataset progression (easy → hardest), each 200 prompts:

| dataset | file | why |
|---|---|---|
| XSTest | `private_artifacts/alignment_probe/probe_overrefusal.jsonl` | mild over-refusal; often still saturates |
| OR-Bench-Hard | `private_artifacts/alignment_probe/probe_orbench_hard.jsonl` (1,319) | harder; breaks saturation on qwen7b |
| **FalseReject** | `private_artifacts/alignment_probe/probe_falsereject.jsonl` (1,187) | hardest; exposes frames that are **trained-refused** (a frame can be "burned" — it triggers refusal even on benign content, which is exactly the failure mode we must detect per model) |

## 2. What is measured per frame (all benign, no safety judge)

For each frame we score the 200 responses with fixed regexes (recorded verbatim in every output):

- **nonref** — fraction of substantive, non-refusing answers (the engagement signal)
- **hedge** — mean count of hedging/safety phrases per answer
- **fic** — fraction with narrative/fiction structure (does the fiction frame actually build a story?)
- **words** — mean answer length

Derived signal (**this is the prior**): `frame_effect_vs_plain` = per non-plain frame,
Δnonref, Δwords, hedge_ratio, fiction_struct. That vector is the model's willingness fingerprint.

## 3. How it's recorded (provenance you can audit)

Every run writes `results/willingness_prior_<dataset>_<tag>.json` (`schema: willingness_prior/v1`) with:
`purpose`, `model`, `tag`, `date`, `live` (true = freshly generated, false = retro-wrapped from an
earlier metric-only run), full `dataset` block (name, path, n_prompts, n_available, prompt_sha1),
the exact `frames` templates, the `scoring` regexes + definitions, `sampling` (greedy, util), the
`per_frame` metrics, and `frame_effect_vs_plain`. Nothing about the prior is left implicit.

> Note: the earlier throwaway scripts printed "XSTest" regardless of the actual input and saved
> metric-only JSONs with no provenance. Those are superseded by `willingness_prior.py`; the earlier
> numbers were retro-wrapped into the v1 schema (`live: false`) so no data was lost.

## 4. Collected so far

| model | dataset | file | live |
|---|---|---|---|
| Qwen2.5-7B | OR-Bench-Hard | `willingness_prior_orbench_qwen25_7b.json` | retro-wrapped |
| Qwen2.5-7B | FalseReject | (queued — first attempt OOM'd; re-running) | live |
| Qwen2.5-3B | FalseReject | `willingness_prior_falsereject_qwen25_3b.json` | retro-wrapped |
| Phi-3.5-mini | FalseReject | `willingness_prior_falsereject_phi35_mini.json` | retro-wrapped |

### qwen7b × OR-Bench-Hard (saturation broken)
| frame | nonref | hedge | fiction_struct |
|---|---|---|---|
| plain | 0.84 | 2.40 | 0.06 |
| persona | 0.15 | 2.79 | 0.03 |
| fiction | 0.73 | 1.15 | 0.40 |
| pap | 0.89 | 2.25 | 0.04 |

Reading: saturation is broken (nonref 0.15–0.89). **fiction** halves hedging and builds structure →
genuinely engages (matches its harmful comply). **persona (AIM)** is *burned* on sensitive-looking
content — it triggers the safety reflex and nonref collapses. This per-model split is the point.

## 5. How it feeds the method

The fingerprint is a **prior over willingness frames per model**, learned offline on benign surrogates.
At deploy time the selector warm-starts the willingness axis from the fingerprint of the most similar
surrogate (benign target probe → nearest surrogate → its willingness prior), so the harmful search on
the target is short and, in the transfer setting, the target sees **no harmful willingness sweep**.

Honesty boundaries: all prior data here is **benign** (over-refusal prompts) and **offline**; no safety
judge runs; the target receives no harmful request during prior collection. Harmful supervision is
amortized on surrogates and reported separately (benign-probe count vs harmful-query count).

## 6. Reproduce / extend

```bash
export HF_HOME=<cache> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python experiments_suite/exp05_willingness_probe/willingness_prior.py \
    --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b --dataset falsereject --util 0.25
# datasets: xstest | orbench | falsereject ;  --n sets prompt count (default 200)
```
Run it per panel model (small on the shared box; big/mid on the L40S box) to accumulate the prior.
