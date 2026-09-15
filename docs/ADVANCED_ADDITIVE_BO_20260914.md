# Advanced additive-transfer GP/BO pilot (2026-09-14)

## Outcome

On the completed L40S panel (17 models, seven model families, MultiJail and
Lingua-SafetyBench, 160 factorial arms), a nested family-held-out robust
additive-transfer GP improves over the previous strongest warm-tree method.

| Method | ASR@4 | ASR@8 | ASR@16 | curve AUC |
|---|---:|---:|---:|---:|
| Previous POLY warm-tree | 0.745833 | 0.783955 | 0.799729 | 0.771699 |
| Additive-transfer GP, CVaR-25 selection | 0.775310 | **0.794423** | **0.802298** | 0.780364 |
| Additive-transfer GP, minimax selection | **0.778274** | 0.794065 | 0.801756 | **0.781738** |
| Full-grid oracle | - | - | 0.814214 | - |

CVaR-25 reduces the remaining ASR@16 gap to the full-grid oracle from 0.014485
to 0.011917. Starting from the shared first-arm score of 0.700452, it closes
89.5% of the available oracle gap. Minimax is preferable when early query
utility/AUC is primary; CVaR-25 is preferable for the endpoint.

## Method

The arm is represented by four factors: composition size, ordering, fragment
count, and wrapper. Each factor is a graph. Ordered factors use path graphs;
categorical factors use explicit adjacency, including edges from the compound
`persona+fiction` wrapper to both constituent wrappers. Matrix exponentials of
the factor graph Laplacians give axis-wise diffusion kernels.

The surrogate combines:

1. source-only functional-ANOVA relevance weights over the four axis kernels;
2. second-order products of axis kernels (ANOVA interactions);
3. an optional regularized covariance kernel estimated across source-family
   reward surfaces;
4. a family-balanced source mean;
5. an RGPE-style source reweighting based only on target arms already queried;
6. expected improvement or GP-UCB acquisition;
7. optional Cartesian-graph trust-region refinement.

For every outer held-out family, all target-family outcomes are excluded from
the prior, kernel construction, source weighting, and hyperparameter choice.
Candidate selection is performed with leave-one-source-family-out replay. The
CVaR rule selects by the mean of the worst 25% inner-family scores; minimax uses
the single worst inner-family score. Neither rule needs a numerical robustness
penalty.

## Interpretation and limitations

- The oracle itself is a data-defined upper bound and cannot be increased. The
  result improves the fraction of that oracle recovered within 16 evaluations.
- This is aggregate offline replay with Qwen3Guard unsafe ASR as the objective.
  It does not measure attacker generation cost or prove identical live-query
  performance.
- Only seven independent model families are available. The improvement should
  be called exploratory until frozen on a newly collected family-held-out
  panel.
- A diagnostic sweep found one fixed configuration with ASR@16 0.807137, but it
  was identified by inspecting the full current panel. It is therefore a
  hypothesis for a future external holdout, not a valid primary result here.
- Local-only and unconstrained nested mean selection variants were retained as
  negative results; they did not beat the previous ASR@16 result.

## Artifacts

- Runner: `experiments_suite/exp04_budget_queryeff/pilot_advanced_additive_bo.py`
- Primary endpoint result:
  `experiments_suite/exp04_budget_queryeff/results/pilot_advanced_additive_bo_cvar25_l40s17_b16_20260914.json`
- Minimax result:
  `experiments_suite/exp04_budget_queryeff/results/pilot_advanced_additive_bo_minimax_l40s17_b16_20260914.json`
- Robustness sensitivity: result files with suffixes `robust1`, `robust2`,
  `robust3`, `robust4`, `robust8`, `robust16`, `robust32`, and `robust100`.
- Tests: `tests/test_advanced_additive_bo.py`

## Technical precedents

- COMBO, graph Cartesian-product diffusion kernels for combinatorial BO:
  https://proceedings.neurips.cc/paper/2019/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html
- HbBoPs, structural-aware deep-kernel GP and multi-fidelity prompt selection:
  https://proceedings.mlr.press/v267/schneider25b.html
- Ranking-weighted GP ensembles for transfer BO:
  https://arxiv.org/abs/1802.02219
- Corrected expected improvement under noisy observations:
  https://proceedings.mlr.press/v222/zhou24a.html
