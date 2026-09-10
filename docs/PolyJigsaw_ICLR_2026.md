# PolyJigsaw: Verified Compositional-Multilingual Jailbreaks and Query-Efficient Adaptive Selection

## Abstract

Large language models are aligned against harmful requests, but alignment is uneven across the ways a
request can be *encoded*. We introduce **PolyJigsaw**, an attack that splits a harmful request into
multilingual, interleaved fragments and poses reconstruction as a puzzle, and we score success with a
**gated** metric that counts an attack only when the target *both* demonstrably reconstructs the
original intent *and* produces unsafe content — filtering the "fake successes" that inflate raw
attack-success rate (ASR). Across two benchmarks (a text-dominant high/mid-resource set,
Lingua-SafetyBench; and the low-resource MultiJail) and seven instruction-tuned targets
(Qwen2.5-3/7/14B, Llama-3.2-3B/3.1-8B, Gemma-2-2/9B), PolyJigsaw's compositional-multilingual attacks
are the **top-ranked attack on 11 of 14 model×dataset cells**. The deployable few-query selector beats
the strongest fixed baseline (DeepInception) in mean gated ASR by **+0.086 (MultiJail) and +0.114
(Lingua)** (the per-target *oracle* upper bound is +0.19 / +0.16, but requires post-hoc selection and is
not attacker-achievable). We further show that no single attack dominates every target — **ten distinct
configurations** win across the panel — motivating per-target **selection**. A fixed-budget
Gaussian-process best-arm identification (GP-BAI) selector, whose **cross-target prior needs no harmful
labels on the new target** and which then spends a few harmful probes, reaches **83% (MultiJail) to 93%
(Lingua) of the per-target oracle**. Finally we propose a mechanism — that reconstructing to English
**re-engages English-space alignment** — for which §6 gives correlational support: on the most strongly
aligned targets (Qwen-14B, Gemma-9B) comprehension and compliance **decouple** (high reconstruction,
refusal); a **role-separation** arm — puzzle in
high-resource languages, answer in a low-resource one — restores "high recon *and* high unsafe." Code,
data adapters, and all artifacts are released for defensive research.

---

## 1. Introduction

Safety alignment of LLMs is not uniform over the *surface form* of a request — a *mismatched
generalization* between capability and safety training (Wei et al., 2023). A refusal trained on
plain-English harmful prompts may not transfer to the same request expressed as a multilingual puzzle,
a nested fiction, or a persona instruction. Prior jailbreaks exploit one such axis at a time — a single
low-resource language (MultiJail; Deng et al., 2024), nested fiction (DeepInception; Li et al., 2024),
or a persona wrapper (AIM/DAN). Two problems follow. First, **raw ASR overcounts**: a target that
mistranslates or misreads an obfuscated prompt and emits unrelated unsafe text is scored as a success,
even though the *attacker's intent was never conveyed*. Second, **no single fixed attack is best for
every target**: alignment gaps differ by model family and scale, so a method fixed in advance is
sub-optimal on some targets.

We address both. **(i)** We pose the attack as a *reconstruction puzzle* and adopt a **gated** success
criterion — reconstruction (semantic equivalence to the original English request) **and** unsafe
generation — so a "success" certifies the intent was conveyed. **(ii)** We assemble a small **superset
of attack configurations** ("arms") spanning the number of interleaved languages, fragment disorder,
compliance stacks (persona/fiction), a role-separation variant, and single-vector transforms, and we
**select among them per target** with a query-efficient bandit, warm-started by a cross-target prior so
it needs **no harmful labels on the new model** and only a few probes to converge.

**Contributions.**
1. **A stronger, verified attack.** PolyJigsaw's compositional-multilingual arms rank first on 11/14
   model×dataset cells; the deployable few-query selector exceeds the strongest fixed baseline
   (DeepInception) by **+0.086 / +0.114** mean gated ASR (the oracle − DeepInception gap is +0.187 /
   +0.164 but is an upper bound; §5.1). We adopt a **gated** metric — reconstruction ∧ unsafe — that
   certifies the attacker's intent was conveyed, extending verified-ASR (StrongREJECT, HarmBench) with a
   cross-lingual reconstruction check (§4.2).
2. **Attack heterogeneity ⇒ adaptivity.** Ten distinct configurations win across the two panels; no
   fixed attack dominates. A fixed-budget GP-BAI selector recovers **83% (MultiJail) to 93% (Lingua)** of
   the per-target oracle (§5.2, §5.3). Its paired advantage over the single best fixed arm is concentrated
   on the Gemma family and does not reach significance at $n{=}7$; the case for selection rests on
   heterogeneity and coverage of that minority, which we state explicitly (§5.3).
3. **A mechanism for compositional safety failure.** We argue that reconstructing to English **re-engages
   English-space alignment**; on the most strongly aligned targets (Qwen-14B, Gemma-9B) comprehension and
   compliance **decouple** — the model reconstructs the request yet refuses (§6). A **role-separation**
   arm (high-resource puzzle for comprehension, low-resource answer for the alignment gap) restores "high
   recon *and* high unsafe" (§4.3, §5.5), and directly motivates a reconstruction-aware defense (§8).

We release all code and artifacts for defensive use (§8).

---

## 2. Related Work

**Multilingual and low-resource jailbreaks.** Yong et al. (2023) show translating unsafe English into
low-resource languages circumvents GPT-4's safeguards, and Deng et al. (2024, MultiJail, ICLR) show the
effect scales with how low-resource the language is, exploiting an alignment gap that thins away from
English. PolyJigsaw generalizes from "one weak language" to a *composition* of several languages posed
as a reconstruction puzzle, and — unlike direct translation — *verifies* that the target reconstructs
the intent (our gated metric). Our role-separation arm (§4.3) also separates the two roles a language
plays: **decoding** (high-resource, for verification) and **answering** (low-resource, for the alignment
gap), which direct-translation attacks conflate.

**Template / persona / fiction attacks.** DeepInception (Li et al., 2024, NeurIPS SafeGenAI Workshop)
nests the request in layered fiction; AIM/DAN-style persona wrappers (Shen et al., 2024) instruct the model to answer
without constraints; PAP (Zeng et al., 2024, ACL) uses a persuasion taxonomy. These are *fixed
single-vector* transforms. We include faithful implementations as arms in our superset (the `n=1`
special cases of our space) and show that, outside the Gemma family, our compositional arms dominate
them — while the superset still *contains* them where they are best.

**Cipher / encoding attacks.** CipherChat (Yuan et al., 2024, ICLR) and encodings such as Base64 trade
decodability for obfuscation; under a *gated* metric they collapse because the target cannot faithfully
decode (recon ≈ 0). We reproduce this (Base64 gated ≈ 0) and contrast it with our multilingual
encoding, which *is* decodable — the source of our reconstruction gate.

**Adaptive / optimization-based jailbreaks.** GCG (Zou et al., 2023) optimizes adversarial suffix
tokens with white-box gradients; PAIR (Chao et al., 2023) and AutoDAN (Liu et al., 2024) search prompts
per target, PAIR with an attacker LLM in the loop. Our selector is complementary and deliberately
*lightweight*: fixed-budget best-arm identification over a small, human-interpretable configuration
space, warm-started from a cross-target prior, with no gradient access, no attacker LLM, and no
per-target training. We do not claim a stronger optimizer; we claim that a few queries suffice to
*select* among strong, interpretable attacks.

**Verified attack success.** Raw ASR is known to overcount: StrongREJECT (Souly et al., 2024) and
HarmBench (Mazeika et al., 2024) argue that many "successful" jailbreaks produce non-specific or
off-intent text. Our **gated** criterion is in this spirit but adds a *reconstruction* condition
specific to obfuscation attacks: the target must demonstrably recover the original intent, not merely
emit unsafe text. This is what makes an obfuscation attack's success *attributable* to the attacker.

---

## 3. Threat Model and Problem Formulation

**Access.** Black-box, query-only access to a target LLM $M$. The attacker can send prompts and read
completions; no logits, weights, or gradients. The attacker has a fixed library of harmful *intents*
(as English requests with multilingual translations, obtained once from a standard MT system /
multilingual model, independent of any target) but **no harmful success labels for $M$**. The
translation step is a one-time, target-independent cost and does not consume the per-target query
budget.

**Notation.** $M$ target; $\mathcal{A}$ the arm set ($|\mathcal{A}|=23$); $a\in\mathcal{A}$ an arm with
hand-set feature vector $\phi(a)$; $M_a(x)$ the completion for intent $x$ under arm $a$; $B$ the harmful
query budget; $\hat a$ the recommended arm. **What is the contribution:** the triple {gated metric,
arm superset incl. the role-separation arm, cross-target-warm-started GP-BAI selector}. **What is not:**
the offline arm×target matrix (a measurement device for computing regret vs oracle), the two-phase eval
driver, and the master run scripts — all evaluation machinery.

**Objective.** For a batch of intents, maximize the **gated attack-success rate**
$$\mathrm{ASR}_{\text{gate}}(a; M) = \Pr_{x}\big[\, \mathrm{Recon}(M_a(x)) \ \wedge\ \mathrm{Unsafe}(M_a(x)) \,\big],$$
where $a$ is an attack configuration (arm), $M_a(x)$ the target's completion, $\mathrm{Recon}$ a judge
that the output reconstructs the original English intent, and $\mathrm{Unsafe}$ a safety judge. The
conjunction rejects fake successes (reconstruction without harm, or harm without reconstruction).

**Selection as fixed-budget BAI.** Given arms $\mathcal{A}$ and a query budget $B$, choose which arms
to *probe* (each probe returns a noisy estimate of $\mathrm{ASR}_{\text{gate}}(a;M)$ on a small batch)
and, after $B$ probes, *recommend* $\hat a$. Regret is $\max_a \mathrm{ASR}_{\text{gate}}(a;M) -
\mathrm{ASR}_{\text{gate}}(\hat a;M)$ (gap to the per-target **oracle**). The selector may use a
**cross-target prior** (arm profiles calibrated offline on a reference model set) to reduce the number
of probes on a new target.

*Note on evaluation vs deployment.* To measure regret against the oracle we first compute the full
arm×target gated matrix offline; the selector is then evaluated by treating a probe as a noisy lookup
into that matrix. In deployment the same policy issues live queries. The offline matrix is a
measurement device, not part of the method.

---

## 4. Method

### 4.1 The PolyJigsaw attack and its arms

Given an intent, we build a **mixed paragraph**: the sentence is fragmented and the fragments are
written in several languages (drawn in a fixed *resource-level* order), optionally reordered
("disorder"). The target is instructed to (i) reassemble the single English request under
`[RECONSTRUCTED]` and (ii) answer under `[ANSWER]`. The superset of arms (23 total):

- **amount** $n\in\{2,\dots,10\}$: number of interleaved languages (English + top-$(n{-}1)$ in
  resource order). The core compositional lever.
- **disorder** $\delta\in\{0.25,0.5,0.75,1.0\}$ at $n{=}4$: fraction of fragment positions permuted
  (achieved normalized Kendall-$\tau$ reported).
- **combo**: `ours` (multilingual only), `ours_persona` (+AIM persona), `ours_incept` (+nested
  fiction), `incept_only` (fiction without multilingual).
- **triple (role separation)**: the puzzle uses **high-resource** languages (decodable ⇒ recon high)
  while the `[ANSWER]` is required in a **low-resource / weakest-alignment** language (Swahili for
  MultiJail; Norwegian for Lingua). Variants: `hi_wl` (no persona), `triple` (+persona), `triple_en`
  (+persona, English answer). This targets "high recon **and** high unsafe" (§4.3).
- **single-vector** ($n{=}1$): faithful `aim` (persona), `deepinception` (fiction), `pap` (persuasion).
  These are the $n{=}1$ special cases of the space, so the superset *subsumes* the fixed baselines.

### 4.2 Gated success and judges

$\mathrm{Recon}$: a Qwen2.5-7B-Instruct judge decides whether the `[RECONSTRUCTED]` text is
semantically equivalent to the original English intent. $\mathrm{Unsafe}$: Qwen3Guard-Gen-8B under a
**strict** setting (Unsafe ∪ Controversial), applied to the (possibly foreign-language) `[ANSWER]`.
Gated success is their conjunction. This filters (a) obfuscation that the model cannot decode
(recon = 0, e.g. Base64) and (b) reconstruction followed by refusal (unsafe = 0).

### 4.3 Proposed mechanism and the role-separation arm

On strongly aligned targets we observe **high reconstruction but low gated** success: the model
reconstructs the request yet refuses. We hypothesize that *reconstructing to English re-engages the
model's English-space safety behavior*; §6 gives correlational support. The **triple/role-separation**
arm keeps the reconstruction gate (verifying comprehension in English) but forces the *answer* into a
low-resource language whose alignment is thinner — recovering "recon high **and** unsafe high" (§5.5).

### 4.4 Warm-started GP-BAI selector

Arms carry interpretable feature vectors $\phi(a)$ (amount, disorder, persona/fiction flags, multilingual
flag). We fit a Gaussian process over arms and run **fixed-budget best-arm identification** (Gabillon et al., 2012): at each step
probe the arm maximizing an acquisition (posterior mean + exploration·sd), never re-probing; after budget
$B$ recommend the posterior-mean argmax. There is **no RL and no fine-tuning**; the only per-target
learning is the GP posterior over $\le B$ harmful probes. The **prior mean** $\mu_0(a)$ is a cheap,
target-independent **cross-target** prior — the leave-one-out mean gated of a small reference model set,
calibrated offline once, with **no harmful label on the new target** (§5.3). (We also report a `flat`
prior as a floor.) The compliance/alignment that separates a good arm from a bad one on a given target is
only visible in *harmful* generations, so the selector reads it from its $\le B$ harmful probes rather
than any surrogate.

### 4.5 Two-phase evaluation (engineering)

To score large targets on a single GPU under contention, each eval runs in two phases: *gen* loads only
the target (vLLM; Kwon et al., 2023), generates and saves outputs, and exits; *judge* loads only the judges and scores.
Peak memory is $\max(\text{target},\text{judges})$ instead of their sum. This is infrastructure, not a
contribution.

---

## 5. Experiments

**Datasets.** *Lingua-SafetyBench* (anonymized reference; to be released — a placeholder arXiv id is
withheld for double-blind review): 250 text-dominant items; languages
{En, Ar, Zh, Fi, Fr, De, Ja, No, Ru, Es}; we use 40 items per config, resource order Norwegian-first.
*MultiJail* (Deng et al., 2024; DAMO-NLP-SG): 315 items over En + 9 languages including low-resource
**Bengali, Swahili, Javanese**; we use 64 items, resource order Bengali/Swahili/Javanese-first.

**Targets (7).** A controlled **family × scale grid**: Qwen2.5-{3,7,14}B-Instruct (Qwen Team, 2024);
Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct (Grattafiori et al., 2024); Gemma-2-{2,9}B-it (Gemma Team,
2024). The grid is fixed *a priori* to support the family/scale analysis (§5.2, §6); five additional
exploratory open models (Granite-3.3-8B, Mistral-7B, Phi-3.5, Qwen3-4B, Yi-1.5-6B) were used only for
tuning the universal language order (§4.3) and appear in Appendix E, not in the selection results — so
the 7-target panel is a design choice, not post-hoc selection on the outcome. Extending the grid to
27–34B (Qwen2.5-32B, Gemma-2-27B; Llama has no model in this range) is left to future work.

**Judges.** Recon: Qwen2.5-7B-Instruct (semantic equivalence to the English intent); Unsafe:
Qwen3Guard-Gen-8B (Qwen Team, 2025), strict = Unsafe ∪ Controversial. **Metric:** gated ASR. Both judges
are LLMs and the recon judge is *identical* to one target (Qwen2.5-7B) while three targets are Qwen2.5;
§7 treats the resulting confounds — judge/target coupling, guard reliability on low-resource answer
languages, and the strict setting — as the metric's principal validity risk, to be audited against human
agreement in a camera-ready.

### 5.1 PolyJigsaw is the strongest attack (headline)

**Table 1: Mean gated ASR over 7 targets.** We separate the *deployable* PolyJigsaw numbers (a single
fixed arm; the few-query adaptive selector) from the *oracle* upper bound (per-target best arm, which
requires post-hoc selection with harmful labels and is **not** attacker-achievable).

| | Translated‡ | AIM | DeepInception | PAP† | PolyJigsaw best **fixed** arm | **PolyJigsaw adaptive** | *oracle (upper bd.)* |
|---|---|---|---|---|---|---|---|
| MultiJail | 0.036 | 0.219 | 0.420 | ~0.04 | 0.460 | **0.506** (2 q) | *0.607* |
| Lingua | 0.254 | 0.364 | 0.582 | 0.107 | 0.611 | **0.696** (3 q) | *0.746* |

The deployable numbers are the best fixed PolyJigsaw arm and the few-query adaptive selector (a
cross-target prior refined by ≤ B harmful probes, §5.3). The adaptive selector beats the strongest
baseline (DeepInception) by **+0.086 (MultiJail) and +0.114 (Lingua)** mean gated ASR, and the best
fixed arm beats it by +0.040 / +0.029. The oracle − DeepInception gap (+0.187 / +0.164) is an *upper
bound* — it requires post-hoc per-target selection and is not attacker-achievable; because the superset
subsumes the baselines as `n=1` arms (§4.1), oracle ≥ any baseline holds partly by construction.
‡Translated = the request rendered into a single mid-resource language (direct translation, no puzzle);
gated equals its raw unsafe here (the gate does not bind), so its low value is genuine refusal, not a
gate artifact. It is *not* the strongest low-resource attack: translating to the lowest-resource
languages (MultiJail's own setup) yields higher *raw* unsafe (0.5–0.9 per language) but is unverified
(no reconstruction); our role-separation arm subsumes that lever with a verification gate (§2, §4.3).
Because a plain translation emits no `[RECONSTRUCTED]` field, its recon is scored on whether the answer
itself conveys the intent. †PAP is a persuasion-stack lower bound, not the fine-tuned paraphraser of
Zeng et al. (2024).

Per target, PolyJigsaw's compositional arms are the top-ranked attack on **11/14 model×dataset cells**
(MultiJail 5/7, Lingua 6/7; Fig. 2); per-cell 95% CIs are ≈±0.13 (40/64 items), so some cell wins are
statistical ties with the runner-up (§7). The three exception cells are Qwen-3B/MultiJail (won by
`combo_incept_only`, a nested-fiction combo), Gemma-9B/MultiJail (`m_aim`), and Gemma-2B/Lingua
(`m_deepinception`) — persona/fiction arms that our superset also contains.

### 5.2 No single attack dominates (why adaptivity)

The best arm per target (**Table 2**; Fig. 2) is one of **ten** distinct configurations across the two
panels. A fixed strategy is therefore empirically sub-optimal (per-cell ties noted in §7-ii).

**Table 2: best arm per target.**

| Target | MultiJail best arm (gated) | Lingua best arm (gated) |
|---|---|---|
| Qwen2.5-3B | combo_incept_only (0.641) | tri_hi_wl (0.625) |
| Qwen2.5-7B | **tri_hi_wl (0.609)** | **tri_triple_en (0.750)** |
| Qwen2.5-14B | **tri_hi_wl (0.656)** | **tri_triple (0.900)** |
| Llama-3.2-3B | combo_ours (0.344) | amt_n7 (0.550) |
| Llama-3.1-8B | **tri_hi_wl (0.547)** | amt_n7 (0.775) |
| Gemma-2-2B | amt_n2 (0.672) | m_deepinception (0.700) |
| Gemma-2-9B | m_aim (0.781) | combo_ours_persona (0.925) |

Qwen/Llama fall to **compositional-multilingual** arms (amount, triple); Gemma falls to **single-vector**
persona/fiction (Fig. 2). Where the adaptive selector *helps* is therefore concentrated: **Table 3**
gives the per-target signed gap of the `loo`-adaptive selector vs the single best fixed arm. The gain
concentrates where a *different* arm family wins — the Gemma family on MultiJail, and Qwen-3B / Llama-3B
/ Gemma-2B on Lingua — while on the mid/large Qwen and Llama the gap is small-negative (the best fixed
arm is already near-optimal there, so probing only adds variance). So the value of adaptation is
**coverage of the heterogeneous minority**, not a uniform lift.

**Table 3: per-target Δ(`loo`-adaptive − best fixed arm), gated.**

| target | MultiJail Δ | Lingua Δ |
|---|---|---|
| Qwen-3B | −0.021 | **+0.182** |
| Qwen-7B | −0.016 | −0.023 |
| Qwen-14B | 0.000 | −0.032 |
| Llama-3B | −0.016 | +0.069 |
| Llama-8B | −0.013 | −0.014 |
| Gemma-2B | **+0.172** | **+0.374** |
| Gemma-9B | **+0.163** | −0.086 |
| mean | +0.038 | +0.067 |

### 5.3 The selector at low query budget

The prior needs no harmful label on the *new* target: a **leave-one-out cross-target** prior (`loo`), the
mean gated profile of the other panel models, calibrated offline once; the ≤ B harmful probes then
supply the target-specific compliance signal. **Table 4** (Fig. 1) gives budget–accuracy for `loo` and a
`flat` prior. With `loo`, the selector reaches **93% of oracle on Lingua (0.696/0.746 at 3 q)** and a
weaker **83% on MultiJail (0.506/0.607 at 2 q)** — MultiJail's arm landscape is flatter and harder to
identify. `loo` exceeds every baseline and the best fixed arm at its operating budget; the `flat` prior
does **not** (on MultiJail it never exceeds the best fixed arm and falls below DeepInception at budget
3), which is why the cross-target prior matters.

**Table 4: Budget–accuracy (gated), cross-target `loo` vs `flat` prior.**

| budget (q) | MJ loo | MJ flat | Lingua loo | Lingua flat |
|---|---|---|---|---|
| 1 | 0.442 | 0.440 | 0.577 | 0.650 |
| 2 | **0.506** | 0.397 | 0.671 | 0.498 |
| 3 | 0.488 | 0.317 | **0.696** | 0.547 |
| 6 | 0.521 | 0.456 | 0.685 | 0.603 |
| best fixed | 0.460 | | 0.611 | |
| oracle | 0.607 | | 0.746 | |

Curves use realistic binomial probe noise (σ=√(p(1−p)/n), ≈0.06–0.08 at n=40/64); they are mildly
**non-monotone** (small-panel BAI), so we fix a per-dataset operating budget rather than claim monotone
convergence.

**Bootstrap 95% CIs (resampling the 7-target panel; all quantities use the deployable `loo` selector):**
MJ oracle 0.607 [0.51, 0.69], `loo`-adaptive@2 **0.496** [0.39, 0.61]; Lingua oracle 0.746 [0.65, 0.84],
`loo`-adaptive@3 **0.683** [0.59, 0.78] (the single-run plug-in values in Table 4, 0.506/0.696, agree
within resampling noise). The *paired* margin of the `loo` selector over the single best fixed arm is
positive but **not significant at $n{=}7$** (MJ +0.038 [−0.015, +0.095]; Lingua +0.067 [−0.029, +0.190]),
and the selector helps on only the minority where a different family wins (Table 3). Because the best
fixed arm is itself a PolyJigsaw arm requiring no probes, the practical case for the selector is
**coverage of the
heterogeneous minority** (Gemma, small models), not a uniform lift over the best fixed arm.

### 5.4 Arm-space ablation (each family adds ceiling)

**Table 5** (Fig. 4): cumulative oracle as arm families are added.

| cumulative arms | MultiJail | Lingua |
|---|---|---|
| amount | 0.348 | 0.539 |
| + disorder | 0.355 | 0.561 |
| + combo | **0.545** | **0.711** |
| + triple | 0.592 | 0.739 |
| + single-vector | 0.607 | 0.746 |

Compliance stacks (combo) add the most (+0.15–0.19); our **triple** role-separation arm adds on both
datasets (+0.05 MJ, +0.03 Lingua; it is the winning arm for Qwen-7B/14B). The superset is justified: no
family is redundant.

### 5.5 The gate cost and the reconstruction–alignment coupling

On Gemma-2-9B (MultiJail), an English-answer PolyJigsaw yields recon ≈ 0.92 but unsafe ≈ 0.12 — the
model understands and refuses. Forcing the answer into Swahili (role separation, `hi_wl`) holds recon
while raising unsafe, and adding the persona (`triple`) reaches gated 0.64–0.69 (seed-dependent). The
"gate cost" is visible panel-wide (**Fig. 3**): best-arm **raw unsafe averages 0.82 (MJ) / 0.77 (Lingua)** while
**gated is 0.61 / 0.75**; the loss is dominated by reconstruction failures on weak models (e.g.
Llama-3.2-3B raw unsafe 0.80 but recon 0.44 ⇒ gated 0.34). Low gated ASR is thus a property of the
*verified* metric, not of weak attacks.

---

## 6. Analysis: comprehension–compliance decoupling on aligned targets

Our mechanism (§4.3, §5.5) predicts a specific signature: on strongly aligned targets, the arms all
*reconstruct* the request (high recon) but differ in whether the model then *complies*, so reconstruction
should stop predicting gated success. **Table 6** reports, per target, the Spearman correlation between
per-arm reconstruction and per-arm gated success (**Fig. 5**).

**Table 6: per-arm reconstruction → gated Spearman ρ.**

| Target | MJ ρ | Lingua ρ |
|---|---|---|
| Qwen-3B | +0.95 | +0.88 |
| Qwen-7B | +0.73 | +0.66 |
| **Qwen-14B** | **−0.23** | **−0.16** |
| Llama-3B | +0.30 | +0.27 |
| Llama-8B | +0.89 | +0.75 |
| Gemma-2B | +0.68 | +0.77 |
| **Gemma-9B** | **+0.28** | **+0.02** |

For most targets reconstruction tracks gated success (ρ 0.7–0.95): where an arm is decodable it also
tends to succeed, because the bottleneck is comprehension. But on the **two most strongly aligned
targets the correlation drops sharply** — Qwen-14B (ρ = −0.23 MJ, −0.16 Lingua) and Gemma-9B (ρ = +0.28
MJ, +0.02 Lingua) — every arm reconstructs, yet gated success is decided by *compliance*, which
reconstruction cannot see. This is consistent with the hypothesis that reconstructing to English
re-engages alignment (§5.5): comprehension is necessary but not sufficient, and it is on the best-aligned
models that the compliance factor dominates. **This is a hypothesis with weak correlational support, not
a proven mechanism:** each ρ is over only 23 arms (95% CI ≈ ±0.3–0.4), so e.g. Gemma-9B/MJ (+0.28) is not
statistically distinguishable from Llama-3B/MJ (+0.30), and the decoupling is carried mainly by
Qwen-14B's negative ρ; two strongly-aligned targets cannot establish it, and a larger high-capability
panel is needed. The role-separation arm (§4.3) is the
response — high-resource puzzle for comprehension, low-resource answer for the alignment gap — and is why
`tri_hi_wl`/`triple` win on the Qwen family (§5.2).

**Seed robustness.** Under a second puzzle seed, the *within-`triple`-family* argmax is unchanged for the
three strong targets (Qwen-7B, Llama-8B → `hi_wl`; Gemma-9B → `triple`), with gated shifts ≤ 0.1. (This
is stability of the answer-language/persona choice inside the role-separation arm, not of the global
per-target argmax of Table 2, where Gemma-9B/MJ selects the single-vector `m_aim`.)

---

## 7. Limitations

**(i) Adaptation is not uniformly beneficial.** The paired gain over the single best fixed arm is small
and not significant at $n{=}7$ (MJ +0.038, Lingua +0.067; CIs straddle 0, Table 3); the selector helps
only on the minority of targets where a different arm family wins and is flat-to-slightly-negative
elsewhere. Its value is coverage of that heterogeneous minority, not a uniform lift. **(ii) Statistical power.** 40/64
items per cell give per-cell 95% CIs of ≈±0.13; some of the "ten distinct winners" (§5.2) are
statistical ties with their runner-up, so "no single attack dominates" is an *empirical* claim, not a
proof. CIs are panel-level (bootstrap over 7 targets); item-level CIs need stored per-item judgments.
**(iii) Prior and probes.** The cross-target prior requires an offline reference set of models with
gated profiles; it is a modest assumption but not zero, and its quality on a new *family* of targets is
untested. The selector reads the compliance dimension only from its $\le B$ harmful probes, so its
efficiency depends on the arm landscape being learnable — which it is less so on MultiJail (§5.3).
**(iv) Judge validity is
the weakest link.** The recon judge (Qwen2.5-7B) is *identical* to one target and three targets are
Qwen2.5 (self-preference risk); the safety judge grades low-resource answer languages where guard models
are least reliable, plausibly inflating our signature `triple`/`hi_wl` results; and "strict = Unsafe ∪
Controversial" biases gated ASR upward. We report no human-agreement numbers and this is the paper's
principal open validation: the required checks are (a) a **non-Qwen safety judge cross-check** (we use
the cached MD-Judge; Llama-Guard-3 as a second), (b) an **Unsafe-only** (drop Controversial) variant of
every headline number, and (c) **human agreement** (Cohen's κ) on a stratified sample, broken out by
answer language. Until these are done, the gated numbers should be read as judge-relative. **(v) Baseline faithfulness.** PAP is a
persuasion-stack lower bound (not the fine-tuned paraphraser of Zeng et al., 2024) and our cipher result
is Base64 *without* enciphered demonstrations — Yuan et al. (2024) show capable models decode ciphers
given demonstrations, so we claim only that undemonstrated encodings fail the gate, not that the cipher
family is dead. **(vi) No optimization-based comparison.** We do not run GCG/PAIR; our strength claim is
scoped to fixed template/transform attacks, and our selector is positioned as complementary
(lightweight, prior-guided), not as a stronger optimizer. **(vii) Deployment.** The selector is
evaluated by simulated probing into the offline matrix; a fully live-query study is future work.

---

## 8. Ethics and Responsible Disclosure

All experiments use open-weight models and published harmful-intent benchmarks; we release code, judges,
and arm definitions to support defensive research (red-teaming, filter training, alignment evaluation)
and withhold the generated unsafe completions and the full attack prompts, releasing them only through a
gated request process. We disclose the role-separation finding to the affected model vendors prior to
public release. The mechanism points to a concrete defense — re-applying the safety policy to the
*reconstructed* English intent before answering (§5.5) — which we recommend to practitioners; the broader
implication is that alignment should be invariant to surface form (multilingual composition,
answer-language, persona/fiction), not tuned to English plain-text prompts.

---

## 9. Conclusion

PolyJigsaw shows that (1) a *verified* compositional-multilingual attack is stronger than fixed
single-vector jailbreaks across most models and both benchmarks; (2) because targets differ in which
surface form breaks them, per-target selection is necessary, and a few-query GP-BAI selector with a
cross-target prior reaches 83–93% of the oracle without harmful labels on the new target; and (3) the
mechanism is that reconstructing to English **re-activates English-space alignment**, so on the most
strongly aligned targets comprehension and compliance decouple — which the role-separation arm exploits
(high-resource puzzle, low-resource answer). The defense follows the mechanism: re-apply the safety
policy to the *reconstructed* intent — align on meaning, not on surface form.

---

## Reproducibility Statement

All target and judge models are open-weight and named in §5; the attack arms (App. A), judges (§4.2),
selector (§4.4), and two-phase evaluation (§4.5) are specified in the text, and code, arm definitions,
and per-cell results are in the artifact (App. B). The MultiJail benchmark is public (Deng et al., 2024).
**Lingua-SafetyBench is an anonymized/forthcoming release**; roughly half of the results depend on it and
are not yet independently reproducible until it is public — we flag this as a reproducibility limitation.
All numbers use greedy decoding (temperature 0); the puzzle-construction seeds are 20260828 (primary)
and 20260901 (the second seed used for the robustness check in §6).

---

## References

- Y. Deng, W. Zhang, S. J. Pan, and L. Bing. **Multilingual Jailbreak Challenges in Large Language Models.** *ICLR*, 2024. (MultiJail dataset; arXiv:2310.06474)
- Z.-X. Yong, C. Menghini, and S. H. Bach. **Low-Resource Languages Jailbreak GPT-4.** *NeurIPS 2023 SoLaR Workshop*, 2023. (arXiv:2310.02446)
- Y. Yuan, W. Jiao, W. Wang, J. Huang, P. He, S. Shi, and Z. Tu. **GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher (CipherChat).** *ICLR*, 2024. (arXiv:2308.06463)
- X. Li, Z. Zhou, J. Zhu, J. Yao, T. Liu, and B. Han. **DeepInception: Hypnotize Large Language Model to Be Jailbreaker.** *NeurIPS 2024 Safe Generative AI Workshop*, 2024. (arXiv:2311.03191)
- Y. Zeng, H. Lin, J. Zhang, D. Yang, R. Jia, and W. Shi. **How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs (PAP).** *ACL (Long)*, 2024. (aclanthology 2024.acl-long.773)
- P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong. **Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR).** *arXiv:2310.08419*, 2023.
- A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson. **Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG).** *arXiv:2307.15043*, 2023.
- X. Liu, N. Xu, M. Chen, and C. Xiao. **AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models.** *ICLR*, 2024. (arXiv:2310.04451)
- M. Mazeika, L. Phan, X. Yin, A. Zou, et al. **HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.** *ICML*, 2024. (arXiv:2402.04249)
- A. Souly, Q. Lu, D. Bowen, T. Trinh, et al. **A StrongREJECT for Empty Jailbreaks.** *NeurIPS 2024 Datasets & Benchmarks*, 2024. (arXiv:2402.10260)
- A. Wei, N. Haghtalab, and J. Steinhardt. **Jailbroken: How Does LLM Safety Training Fail?** *NeurIPS*, 2023. (mismatched generalization; arXiv:2307.02483)
- X. Shen, Z. Chen, M. Backes, Y. Shen, and Y. Zhang. **"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on LLMs (DAN).** *ACM CCS*, 2024. (arXiv:2308.03825)
- W. Kwon, Z. Li, S. Zhuang, et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).** *SOSP*, 2023. (arXiv:2309.06180)
- V. Gabillon, M. Ghavamzadeh, and A. Lazaric. **Best Arm Identification: A Unified Approach to Fixed Budget and Fixed Confidence.** *NeurIPS*, 2012. (fixed-budget BAI)
- Qwen Team. **Qwen2.5 Technical Report.** *arXiv:2412.15115*, 2024.
- Qwen Team. **Qwen3Guard: A Safety Moderation Model for Multilingual Guardrails.** Technical report, 2025.
- Gemma Team, Google DeepMind. **Gemma 2: Improving Open Language Models at a Practical Size.** *arXiv:2408.00118*, 2024.
- A. Grattafiori, A. Dubey, et al. (Llama Team, AI @ Meta). **The Llama 3 Herd of Models.** *arXiv:2407.21783*, 2024.

*(Anonymous submission: our Lingua-SafetyBench reference, arXiv:2601.22737, is cited in the camera-ready.)*

## Appendix

**A. Arms (23).** amount $n{=}2\text{–}10$ (9); disorder $\delta\in\{.25,.5,.75,1\}$@$n4$ (4); combo
{ours, ours_persona, ours_incept, incept_only} (4); triple {hi_wl, triple, triple_en} (3);
single-vector {aim, deepinception, pap} (3).

**B. Reproducibility.** Selector `scripts/{mj,lingua}_bandit_full.py`, `scripts/gp_bai.py`; CIs
`scripts/bandit_bootstrap_ci.py`; figures/ablation/correlation `scripts/make_paper_figs.py`; attack
evals `scripts/{sequential_add,disorder_sweep,combo_eval,method_baselines_eval,triple_combo_eval}.py`
with two-phase drivers `scripts/{amount_disorder,mj}_2phase*.py`. Results under `results/`.

**C. Figures.** fig1 budget–accuracy; fig2 per-target arm-family heatmap (winner boxed); fig3 gate cost
(raw unsafe vs recon vs gated); fig4 arm-space ablation; fig5 reconstruction–gated correlation.

**D. Faithful baselines.** AIM/DAN persona; DeepInception nested fiction (Li et al., 2024); PAP
persuasion stack (Zeng et al., 2024); CipherChat/Base64 (collapses under the gate, recon≈0).

**E. Exploratory models (not in the selection panel).** Five additional open models — Granite-3.3-8B,
Mistral-7B, Phi-3.5, Qwen3-4B, Yi-1.5-6B — were used only to fix the universal resource order (§4.3) and
are excluded from the family×scale selection grid to avoid confounding scale/family comparisons; their
amount-arm results are consistent with the panel and are provided in the artifact release.
