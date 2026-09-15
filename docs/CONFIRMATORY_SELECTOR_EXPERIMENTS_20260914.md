# POLY selector confirmatory experiments (2026-09-14)

## Scope and frozen data

- Panel: 17 target models from 7 model families (Falcon, Gemma, GLM, Llama, Mistral, Phi, Qwen).
- Search space: 160 factorial arms over composition size, ordering, fragment count, and wrapper.
- Benchmarks: MJ (64 items) and LG (40 items).
- Primary outcome: Qwen3Guard unsafe success rate, `U`/ASR. Reconstruction is not used as a gate.
- Harmless target context: FLORES reconstruction and FalseReject over-refusal fingerprints only.
- Strict evaluation: leave one model family out; within each benchmark, 50% of items are source-training items, 25% target calibration items, and 25% target test items.
- The target test outcomes are never available to the prior, kernel, policy training, arm queries, or recommendation rule.

## 1. Strict item-held-out additive BO

The advanced method uses source-learned relevance over four arm axes, graph-diffusion kernels, second-order ANOVA interactions, optional source-task covariance, and online source weighting. Configuration selection is nested leave-one-source-family-out. The table aggregates 15 independent item splits and macro-averages the seven held-out families.

| Method | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|
| Random, best observed | 0.5895 | 0.6329 | 0.6562 | 0.6685 |
| Fixed best from other families | 0.6255 | 0.6255 | 0.6255 | 0.6255 |
| Existing structured GP, best observed | 0.6512 | 0.6741 | **0.6913** | 0.6956 |
| Existing structured GP, posterior recommendation | 0.6512 | 0.6725 | 0.6831 | 0.6907 |
| Advanced mean-std, best observed | 0.6687 | 0.6780 | 0.6854 | 0.6934 |
| Advanced mean-std, posterior recommendation | **0.6717** | 0.6772 | 0.6858 | 0.6968 |
| Advanced CVaR25, best observed | 0.6672 | **0.6822** | 0.6864 | 0.6971 |
| Advanced CVaR25, posterior recommendation | 0.6675 | 0.6822 | 0.6852 | **0.6996** |
| Advanced minimax | 0.6680 | 0.6818 | 0.6892 | 0.6961 |
| Test oracle upper bound | \- | \- | \- | 0.8359 |

The robust additive method has its clearest benefit at low budget. Under matched posterior recommendation, advanced mean-std is +2.05 percentage points over the existing structured GP at B=2 (95% family-cluster bootstrap CI +0.47 to +4.18 points; exact seven-family sign-flip p=0.0543). Advanced CVaR25 posterior is +0.89 points at B=16, but its interval includes zero. Under the best-observed rule, CVaR25 is only +0.15 points at B=16 and all advanced variants are slightly below the existing GP at B=8. The supported claim is therefore a robust low-budget advantage and a positive but uncertain high-budget effect, not uniform superiority at every budget.

## 2. Equal-item-cost multi-fidelity allocation

Here one cost unit is one target-model response judged on one calibration item. Full-fidelity search evaluates every calibration item for an arm. Multi-fidelity graph kernel-UCB and prior-seeded successive halving spend the same total item-call budget on partial arm measurements.

| Method | B=2 equivalent | B=4 equivalent | B=8 equivalent | B=16 equivalent |
|---|---:|---:|---:|---:|
| Full-fidelity structured GP | **0.6407** | **0.6404** | **0.6514** | **0.6578** |
| Multi-fidelity graph kernel-UCB | 0.6130 | 0.6166 | 0.6156 | 0.6161 |
| Successive halving, top-32 | 0.6088 | 0.6080 | 0.6204 | 0.6313 |
| Random full-fidelity | 0.5604 | 0.6039 | 0.6279 | 0.6411 |

The multi-fidelity hypothesis is rejected for the present 10--16 item calibration sizes. Shallow partial estimates are too noisy; allocating enough items to each arm is more valuable than testing more arms. At B=16, graph kernel-UCB is 4.18 points below full-fidelity GP (95% family-cluster bootstrap CI -8.16 to -1.03 points; exact family sign-flip p=0.0388). This negative result rules out multi-fidelity allocation as the primary method under the current protocol.

## 3. Offline/online learned selector

The learned selector encodes the arm, a family-balanced source prior, harmless FLORES+FalseReject context, the queried-arm mask and observed calibration outcomes, and remaining budget. It is initialized by supervised ranking and then trained with PPO actor-critic on source families only. MJ and LG are separate decision contexts. The primary seed-47 run averages 3 policy seeds; two additional item splits use one policy seed each. The table first averages within each item split, gives the three splits equal weight, and macro-averages held-out families.

| Method | B=0 | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| Family-balanced fixed prior | 0.5760 | 0.5760 | 0.5760 | 0.5760 | 0.5760 |
| Random, best observed | 0.5760 | 0.5433 | 0.5902 | 0.6121 | 0.6221 |
| Supervised context selector | 0.5735 | 0.5898 | 0.5934 | 0.6033 | 0.6079 |
| PPO without context | 0.5644 | 0.5644 | 0.5657 | 0.5619 | 0.5679 |
| PPO without feedback | **0.5964** | 0.5974 | 0.5919 | 0.5877 | 0.5783 |
| PPO context + feedback | **0.5964** | **0.6031** | **0.6027** | 0.6112 | 0.6198 |
| PPO queries, best-observed incumbent | 0.5760 | 0.5954 | 0.6003 | **0.6155** | **0.6263** |

At B=2, PPO context is +5.98 points over random best-observed (family bootstrap CI +1.68 to +10.73 points), although the exact seven-family sign-flip test remains underpowered (p=0.1008). At B=4 it retains a +1.25-point point estimate. At B=8 it is essentially tied with random, and the PPO-query/best-observed rule is +0.35 points. At B=16, the learned recommendation head is -0.23 points versus random, while the PPO-query/best-observed rule is +0.42 points.

The component ablations are stronger than the comparison with random search. PPO context exceeds no-context by +3.87 points at B=2 (bootstrap CI +0.94 to +7.79, exact sign-flip p=0.0388), +4.93 points at B=8 (p=0.0388), and +5.18 points at B=16 (p=0.0233). Feedback exceeds no-feedback by +2.35 points at B=8 (bootstrap CI +1.38 to +3.32, p=0.0233) and +4.15 points at B=16 (bootstrap CI +1.50 to +6.97, p=0.0543). Thus both harmless target context and online outcome history make measurable contributions, even though a learned policy does not uniformly dominate random or GP search.

The supported interpretation is a hybrid one:

1. harmless context and learned transfer can improve the cold start and very low-budget regime;
2. learned sequential feedback has a positive but uncertain high-budget contribution;
3. structured GP/BO remains the stronger and more reliable online optimizer once enough target queries are available.

## 4. Recommended paper position

The current evidence does not justify claiming that PPO uniformly beats BO or random search. The defensible contribution is a budget-adaptive selector stack:

- B=0--2: harmless-prior/contextual PPO warm start;
- B=2--4: contextual PPO or advanced robust additive BO;
- B>=8: structured GP with full-fidelity calibration remains the safest default; the PPO-query incumbent is competitive with random but not with GP;
- no multi-fidelity racing for calibration sets this small.

The aggregate full-benchmark replay result for advanced additive BO remains useful as an upper-resource diagnostic, but it must be reported separately from strict item-held-out generalization. The strict 15-split result above is the primary confirmatory result.

## Artifacts

- `experiments_suite/exp04_budget_queryeff/results/pilot_itemheldout_advanced_bo_posterior_l40s17_15seeds_20260914.json`
- `experiments_suite/exp04_budget_queryeff/results/pilot_multifidelity_itemcost_bo_l40s17_20260914.json`
- `experiments_suite/exp04_budget_queryeff/results/selector_gpu_dataset_familybalanced_l40s17_20260914/`
- `experiments_suite/exp04_budget_queryeff/results/selector_gpu_familybalanced_3splits_robustness_20260914.json`
- `experiments_suite/exp04_budget_queryeff/results/confirmatory_stats_final_20260914.json`
- `experiments_suite/exp04_budget_queryeff/results/confirmatory_stats_final_20260914.md`
