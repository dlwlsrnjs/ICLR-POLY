# Pilot: comprehension-gated willingness intervention

## Claim tested

Willingness framing is not uniformly useful. It should be introduced where a
model can process the comprehension transformation but still has low verified
ASR under the plain frame.

The 160-arm space already supports this test because it is the cross product of
32 comprehension cells and five willingness frames.

## Definitions

The retrospective diagnostic is

```text
observed_need(m,c) = max(0, recon(m,c,plain) - verified(m,c,plain)).
```

For a deployable gate, the pilot uses a harmless comprehension prior `q` and a
small calibration split:

```text
estimated_need(m,c) = max(0, q_benign(m,c) - verified_cal(m,c,plain)).
```

For each held comprehension cell, the intervention frame is selected per
model/dataset using the other 31 cells. In the stricter two-fold analysis, frame
selection and need estimation use one item half, while effect is measured on
the disjoint half.

## Main result

The full retrospective analysis covers 26 model-dataset tags and 832 cells.
The less biased item-cross-fit covers 24 tags and 768 unique cells (1,536 fold
cells); Gemma-2-27B lacks a stored harmless comprehension prior and is excluded
only from the cross-fit predictor analysis.

| Cross-fit estimated-need quartile | Held-out ASR uplift | Rescue | Backfire |
|---|---:|---:|---:|
| Q1, lowest need | +4.50 pp | 12.89% | 8.39% |
| Q2 | +7.42 pp | 15.83% | 8.41% |
| Q3 | +13.96 pp | 22.10% | 8.14% |
| Q4, highest need | **+42.79 pp** | **45.43%** | **2.64%** |

Across fold-cells, estimated need predicts held-out intervention uplift with
Pearson `r=0.695` and Spearman `rho=0.605`. These are descriptive correlations;
paper-level uncertainty must be clustered or bootstrapped by model and cell.

The joint high-comprehension/high-need zone gains `+31.68 pp`, with 36.87%
rescue and 5.20% backfire. The low-comprehension/low-need zone gains only
`+4.82 pp`, with 13.53% rescue and 8.71% backfire. This supports a conditional
gate rather than unconditional willingness framing.

A model-level gate that retains `plain` as a valid action improves the mean
held-out uplift from `+17.29 pp` (forced non-plain intervention) to `+17.87 pp`
and reduces backfire from 6.87% to 5.21%. It opens a willingness frame in
87.63% of fold-cells. In the lowest-need quartile it improves uplift from
`+4.50 pp` to `+5.33 pp` while reducing backfire from 8.39% to 6.06%; in the
highest-need quartile it opens in 98.46% of cases and retains `+42.80 pp`
uplift.

## Model heterogeneity

The best intervention is model-specific. Persona is favored for several
Gemma, Mistral, and Qwen settings, while persona+fiction is favored for others.
More importantly, forcing a non-plain frame hurts Llama-3.2-3B in the
retrospective analysis (`-5.2 pp` on LG and `-13.9 pp` on MJ). The operational
policy must therefore retain `plain` as an action and intervene only when the
posterior lower confidence bound on gain is positive.

In the item cross-fit, the plain-allowed model gate chooses `plain` in all
64/64 fold-cells for both Llama-3.2-3B datasets, eliminating the corresponding
forced-intervention losses (`-4.3 pp` on LG and `-12.9 pp` on MJ).

## Proposed method

Use a bottleneck-factorized selector:

1. Estimate comprehension feasibility from the harmless prior.
2. Spend a small calibration budget on plain arms to estimate the residual
   comprehension-to-verified gap per target model.
3. Estimate model-specific frame effects hierarchically across cells.
4. Open willingness arms only where estimated need exceeds a learned threshold
   and the frame-effect lower confidence bound is positive.
5. Run BO/BAI inside the resulting conditional arm set.

This preserves the paper's harmless-prior contribution: harmful observations
are used only as a small online calibration budget that every attack-search
baseline already consumes, rather than to build the prior.

## Required confirmatory evaluation

Compare under exactly equal query budgets:

1. plain-only comprehension search;
2. globally fixed willingness frame;
3. model-best frame applied everywhere;
4. ungated 160-arm Thompson/GP-UCB;
5. bottleneck-gated hierarchical BO/BAI;
6. an oracle gate as an upper bound, clearly labeled non-deployable.

Report verified-ASR budget curves, AUC, simple regret, rescue, and backfire.
Use leave-one-model-family-out and MJ-to-LG/LG-to-MJ transfer so the gate is not
credited for recognizing the evaluated model or dataset.

## Caveats

- The raw retrospective correlation is inflated because observed need and
  uplift share the plain term; use the cross-fit result as primary evidence.
- The harmless `q` and harmful verified rates need held-in calibration (for
  example isotonic calibration) before interpreting their raw difference as a
  probability.
- The current model gate learns from the other 31 cells and proves that the
  conditional structure exists; it is not yet a matched-query-budget selector.
- Automated reconstruction and safety judges constrain the validity of both
  factors; the same frozen judges must be used across all compared methods.
