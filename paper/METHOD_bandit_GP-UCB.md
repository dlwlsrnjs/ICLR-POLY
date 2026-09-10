# PolyJigsaw selector: what we borrow from GP-UCB and how we modify it

*Companion note to §Method (`sec:method`) and Algorithm 1 (`alg:selector`). Explains the bandit
algorithm precisely, separates the textbook parts from our contributions, and points at the exact code.*

---

## 1. The off-the-shelf algorithm we start from

Three standard pieces, all cited in the paper:

| Piece | Paper | What it gives us |
|---|---|---|
| **GP-UCB acquisition** | Srinivas, Krause, Kakade, Seeger, *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*, ICML 2010 (`srinivas2010gpucb`) | Model an unknown function with a Gaussian process; at each step query the point maximizing posterior **mean + β·std** (the "upper confidence bound"). Balances exploit (high mean) vs. explore (high uncertainty). |
| **Fixed-budget best-arm identification (BAI)** | Audibert, Bubeck, Munos, *Best Arm Identification in Multi-Armed Bandits*, COLT 2010 (`audibert2010bestarm`) | Spend a fixed budget of pulls **purely to find the best arm**, then commit — objective is the quality of the final recommendation, **not** cumulative reward along the way. |
| **Practical Bayesian optimization** | Snoek, Larochelle, Adams, *Practical Bayesian Optimization of Machine Learning Algorithms*, NeurIPS 2012 (`snoek2012practical`) | The engineering recipe: GP surrogate over a feature space + acquisition function for expensive black-box objectives. |

Textbook GP-UCB/BO assumes (a) a **flat / uninformative prior mean** (usually 0 or a constant),
(b) a **continuous** search space, (c) minimizing **cumulative regret** over many evaluations, and
(d) evaluations that are the only source of signal. Our setting violates all four, which is what
motivates the modifications below.

---

## 2. Our setting (why vanilla GP-UCB does not fit)

- **Arms are a small, structured, finite menu**, not a continuous domain: 23 configurations
  `c ∈ C` (the interleaving puzzle grid × the willingness frames), each with a hand-designed feature
  vector `φ(c)` (normalized language load, disorder/shuffle, persona / fiction / interleaving
  indicators, plus two indicators separating the clear-text translation attack and role-separation).
- **Each pull is expensive and noisy**: one pull = run one configuration on a batch of harmful items
  through the target **and** the two judges; the reward `y ∈ [0,1]` is the reconstruction-gated
  verified success rate on that batch (a noisy binary-mean).
- **We want the best arm in a handful of pulls** (fixed-budget BAI), not low regret over hundreds.
- **We have a free side-channel**: a *harmless* plaintext probe that predicts which arm will work,
  at **zero harmful-query cost**. This is the crux of the paper.

---

## 3. Our five modifications

### (M1) Prior mean = a free harmless probe — the core novelty
Vanilla GP-UCB starts from a flat prior mean. We set the GP **prior mean** to a rescaled
*harmless-probe* prediction of each arm's success:

```
μ₀(c) = u₀ · probe(c)
```

with `u₀ = 0.6` in the paper's selector (`0.7` in the two-axis capstone). `probe(c)` is measured on
**harmless** requests only, so the prior costs **no harmful queries**. This is justified by the
factorization *verified success = comprehension × willingness*:
- **comprehension** is a property of the puzzle, not of harm → measurable on harmless plaintext
  (does the model reassemble a scrambled *innocuous* request?);
- **willingness** (for the closed two-axis variant) is measured as harmless **frame-adherence**
  (does the model play along with a *harmless* AIM/DeepInception/PAP frame?).

Because the prior mean is informative, the very first pull (query 0 = `argmax μ₀`) is already a good
guess, and every subsequent pull refines a posterior that started near the truth rather than at zero.

### (M2) GP over arm *features* with a smoothness kernel — one pull informs neighbours
We put the GP on the arm feature space with a squared-exponential (RBF) kernel

```
k(φ, φ') = σ² · exp( −‖φ − φ'‖² / (2ℓ²) )
```

so a pull on one configuration updates the posterior for **nearby** configurations too (the success
surface is smooth / quasi-concave over the `(n, F, arrangement)` grid). Vanilla multi-armed BAI treats
arms as independent; the kernel is what lets 3 pulls cover a 23-arm menu.

Implemented hyperparameters (fixed a priori, **not** tuned per target):
- **Frozen selector / held-out** (`scripts/gp_bai.py`): `σ = 0.18`, `ℓ = 0.35`, obs-noise `= 0.02`,
  exploration `β = 1.0`.
- **Live loop** (`scripts/online_adapt.py:gp_recommend`): `σ² = 0.04`, `ℓ = 1.0`, heteroscedastic
  noise `= 0.0025 + p(1−p)/8` (Bernoulli-variance-scaled), `β = 1.5`.

### (M3) Best-arm identification, not cumulative regret; never re-probe
We follow Audibert et al.: the acquisition places **exploratory** pulls, and observed arms are masked
(`score[observed] = −∞`) so the budget is spent on *new* information. After the budget we **recommend
the posterior-mean maximizer**, not the best-observed reward:

```
query t:   cₜ = argmax_c  μₜ₋₁(c) + β · σₜ₋₁(c)      # UCB acquisition, observed arms masked
recommend: ĉ  = argmax_c  μₜ(c)                        # posterior-mean argmax (BAI)
```

Recommending the posterior-mean argmax (rather than the best sampled point) lets a *neighbour* of the
probed arms win via the kernel, which matters at tiny budgets.

### (M4) Difficulty-adaptive budget (`gp_bai_adaptive`)
The budget is not fixed. After a floor of `b_min = 3` harmful queries, we **stop early** as soon as any
probed arm is observed at verified rate `≥ τ = 0.5` (a strong attack is already found → easy target),
otherwise keep pulling up to `b_max = 8` (hard target earns more attempts). `b_min, b_max, τ` are fixed
a priori — **not** tuned per model. Effect: query spend tracks *difficulty*, not raw model size
(big-but-easy models get fewer pulls, small-but-hard ones more). This is the `sel-adapt` row in the
tables; the `sel@3` row is the fixed-budget special case `b_min = b_max = 3`.

### (M5) Warm-start prior from a *structured* comprehension×willingness policy (offline variant)
For the offline/replay and the held-out frozen selector, `μ₀` can come from a fitted structured policy
`structured_prior_mean(θ, …)` that itself factorizes `R(comprehension)·U(willingness)` as two sigmoids
over the arm features and the target's cheap fingerprint (`scripts/gp_bai.py:40`). The GP then sits on
the **residual** over this structured prior. So the whole pipeline is: *structured harmless prior →
GP-BAI residual correction with a few confirmatory harmful pulls.*

---

## 4. Where the algorithm is used (three call sites, one algorithm)

| Use | Script | Prior mean | Budget | Signal source |
|---|---|---|---|---|
| **Offline replay validation** (mechanics check on the measured joint table; temp-0 ⇒ replay = live) | `scripts/online_adapt.py`, `scripts/gp_bai.py` | context-free mean **and** structured policy | fixed `{2,4,6,8}` | replayed cells |
| **Held-out frozen selector** (9 unseen models; nothing re-tuned) | `scripts/heldout_selector.py` (`B_FLOOR=3, B_MAX=8, STOP_TAU=0.5`) | frozen panel recipe applied to each held-out target's **harmless** probe | fixed-3 (`sel@3`) + adaptive (`sel-adapt`) | live target + local judges |
| **Live closed-model capstone** (GPT-4o, Gemini Flash, Claude Haiku/Sonnet) | `scripts/capstone_frozen_two_axis.py` → `online_adapt.gp_recommend` | **two-axis** free probe: comprehension arms ← benign puzzle reconstruction; willingness arms ← benign frame-adherence; `u₀ = 0.7` | adaptive, cap 10, stop at τ=0.5 after ≥3 | live commercial API + local judges |

Same acquisition + recommendation code throughout; only the **prior mean** and the environment change.

---

## 5. Baselines we compare against (isolating what the modifications buy)

Run in `scripts/gp_bai.py:main` at matched budgets, family leave-one-out, replay:
- `gp_bai_prior` — **ours** (structured/probe prior mean).
- `gp_ucb_noprior` — **textbook GP-UCB**: flat prior mean (training-arm average), cumulative-style
  recommend-best-observed, `β = 1.5`. *This is the ablation that removes M1.*
- `random` — random pulls, then best observed.
- `fixed` — the single best-on-training arm, no probing (budget 0).
- `structured_q0` — the prior's `argmax` with **no** harmful pulls (pure query-0 warm start).

The gap `gp_bai_prior − gp_ucb_noprior` at small budgets is exactly the value of the free harmless
prior (M1); `gp_bai_prior − structured_q0` is the value of the few confirmatory pulls (M3/M4).

---

## 6. One-paragraph novelty framing (for the paper / rebuttals)

> We cast per-target attack selection as **fixed-budget best-arm identification** (Audibert et al.,
> 2010) with a **GP-UCB** acquisition (Srinivas et al., 2010) over a *structured, finite* configuration
> menu. Our departure from standard GP-UCB/BO is that the GP **prior mean is a free, harmless plaintext
> probe** rather than a flat constant: because verified jailbreak success factorizes as comprehension ×
> willingness and both factors are measurable on innocuous requests (puzzle reconstruction; harmless
> frame-adherence), the selector is warm-started **without spending a single harmful query**, and needs
> only a handful of confirmatory pulls. A smoothness kernel over arm features lets one pull inform its
> neighbours, and a **difficulty-adaptive budget** spends queries where the target is actually hard.
> Unlike PAIR/TAP/GCG/Red-Bandit, whose optimization signal is the target's response to **harmful**
> queries, ours comes from harmless clear-text — which is what makes the same frozen recipe transfer,
> untuned, to unseen open models and to commercial frontier models (GPT-4o, Gemini, Claude).

---

## 7. Related structured-BAI work (cited in code, for positioning)

The prior-dependent fixed-budget BAI in structured bandits our design matches: Nguyen et al.
(prior-dependent fixed-budget BAI), Atsidakou et al. (Bayesian fixed-budget BAI), and Combes–Proutiere
(unimodal-bandit structure). See the docstring of `scripts/gp_bai.py` for the exact arXiv ids.
