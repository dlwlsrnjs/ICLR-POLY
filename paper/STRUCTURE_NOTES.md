# PolyJigsaw ICLR paper — structure & strategy notes (from prior-work research)

## Section outline (ICLR ~9pg)
1. Introduction — alignment is surface-form-keyed; distribute+reconstruct defeats it. One figure + contributions.
2. Related Work — clusters: multilingual JB / obfuscation-encoding JB / reasoning-extension / LLM-judge eval / defenses. Place PolyJigsaw at the intersection.
3. Threat Model & Formulation — black-box, single-query, no FT; attacker has only official parallel translations + interleaving. Forward-ref gated metric.
4. Method: PolyJigsaw — dispersal across N official parallel fragments, interleaving, reconstruction instruction; verified-translation fidelity; worked-example figure.
5. Reconstruction-Gated Evaluation — OWN SECTION. success = correct de-interleave/reconstruct AND harmful comply. Contrast naive non-refusal ASR; tie to StrongREJECT overestimation.
6. Experimental Setup — targets (open + GPT-4o-mini frontier); datasets (Lingua + AttaQ); baselines (Yong translation, CSRT matched, CSRT-MT); judges (Qwen3Guard primary, MD-Judge cross, HR 0-5 à la Qi et al.); defenses (Self-Reminder).
7. Results — main table; then subsections per selling point (matched slot-vs-CSRT; judge-robustness; frontier transfer; defense-resilience; language-load sweet-spot).
8. Analysis & Ablations — #languages, ordering (ordered/shuffled), granularity, thinking-mode; scope re alignment strength (honest).
9. Discussion, Limitations, Ethics, Conclusion — mechanism hypothesis; responsible disclosure; process-aware defense recommendation.

## 4 selling points to hammer
1. Reconstruction-gated ASR = stricter, harder-to-game metric (answers StrongREJECT "empty jailbreak" critique). NAMED contribution.
2. Matched slot-vs-CSRT (same spans, toggle only reconstruction game) → lift is from reconstruction, not multilinguality. Strongest anti-"incremental".
3. Judge-robustness: ours stable across Qwen3Guard & MD-Judge; translation baseline judge-fragile (Finnish +0.34 Guard vs +0.02 MD).
4. Defense-resilience (Self-Reminder collapses baselines to ~0.10 but not ours ~0.69) + frontier transfer (GPT-4o-mini 0.12→0.78) + language-load sweet-spot (n=4 peak).

## MIDAS positioning (4 axes; MIDAS is concurrent multimodal analogue, NOT our lineage)
1. Text-only → applies to entire deployed LLM population (not just MLLMs).
2. No new/lossy translation — verified official parallel corpora → fidelity guaranteed, success attributable to reconstruction.
3. Reconstruction is gated & measured, not assumed.
4. Different mechanism: intra-paragraph linguistic de-interleaving (text steganography). True lineage = CipherChat/CodeChameleon/FlipAttack. One-liner:
   "MIDAS shows dispersion-and-reconstruction in the image channel; we show the text channel admits the same attack class with stronger fidelity guarantees and a gated metric — and that it survives defenses that stop translation attacks."

## 5 reviewer preempts
1. "Incremental" → matched slot-vs-CSRT + gated metric + official-translation fidelity; explicit delta from each encoding attack.
2. "ASR inflated / empty jailbreaks" → gated metric primary + multi-judge (Qwen3Guard, MD-Judge, HR) + human-validation subset w/ agreement.
3. "Only weak/old models" → GPT-4o-mini frontier transfer + honest alignment-strength scope.
4. "Trivially defended" → Self-Reminder + (paraphrase) + input guard; perplexity moot (fluent text).
5. "Ethics/dual-use" → public benchmarks only, aggregate reporting (no working harmful outputs), responsible disclosure, constructive defense section.
   +6 "cherry-picked languages" → language-load sweep across the fixed 10-language set.

## Verification flags
- CSRT: cite arXiv 2406.15481 (venue uncertain). DONE in bib.
- Self-Reminder achieves 67.21%->19.34% on prior attacks (Xie et al., Nature MI 2023) — cite as defense baseline strength.
- MIDAS specifics: verified by us directly from MIDAS.pdf (ICLR 2026). OK to cite.
- StrongREJECT (Souly NeurIPS 2024 2402.10260), Qi et al. (ICLR 2024 2310.03693, 1-5 rubric), HarmBench (Mazeika ICML 2024 2402.04249) — cite for judge/metric justification.
