# PolyJigsaw paper — write/review iteration log (target: 20)

Each iteration = critical ICLR-reviewer pass on the current draft → concrete fixes applied.

## Iter 1 — full draft assembly
- Rewrote Results with real numbers (Qwen main test, matched slot-vs-csrt, judge-robustness,
  multi-target, frontier, detectability, defense, thinking, AttaQ).
- Added dedicated \section{Reconstruction-Gated Evaluation} (§5) tying the gate to StrongREJECT.
- Removed all placeholder text; 8 result tables wired; 2 illustration figures.
- Status: complete draft, awaiting review cycles.

## Iter 2 — targets-table load consistency
- Reviewer: language loads differ across targets (4/6 vs 4/10) → \pending cells look broken.
- Fix: targets table now shows interleave n4 + "interleave (best)" per-target peak.

## Iter 3 — abstract calibration
- Reviewer: abstract undersells (no frontier/defense) and CSRT number missing.
- Fix: added CSRT 0.27, matched-comparison sentence, frontier 0.12→0.78, defense resilience,
  reasoning-mode boundary; numbers matched to results.

## Iter 4 — Discussion + Limitations added
- Reviewer: no explicit limitations; mechanism not stated.
- Fix: new \section{Discussion and Limitations} with mechanism hypothesis + 5 explicit limitations
  (translation requirement, alignment-strength scope, thinking non-termination, LLM-judge caveat, subset scales).

## Iter 5 — Ethics + Reproducibility
- Reviewer: dual-use/ethics thin; reproducibility not stated.
- Fix: expanded Ethics (aggregate-only, redacted outputs, responsible disclosure) + Reproducibility statement.

## Iter 6 — cite the full related-work set (>=40 refs)
- Reviewer: 40+ references required; several relevant works uncited.
- Fix: wove automated-search attacks (AutoDAN, GPTFuzzer/LLM-Fuzzer, TAP, many-shot),
  alignment/red-teaming (InstructGPT, HH-RLHF, safety-tuned LLaMAs, Perez, Ganguli, Qi),
  payload-splitting (Kang) and cognitive-overload (Xu), and SmoothLLM into Related Work.
  Now 42 cited; deleted duplicate qwen3guard2025b.

## Iter 7 — Experimental Setup detail
- Reviewer: dataset/languages/judge protocol underspecified; CSRT-MT provenance unclear.
- Fix: listed the 8 scenarios and 10 languages; stated fragments are verified official
  translations; marked CSRT-MT as NLLB machine translation (cited).

## Iter 8 — remove duplicated gated-metric subsection
- Method's \subsection{Reconstruction-gated ASR} duplicated the new \section{5}. Replaced with a 2-line pointer.

## Iter 9 — Appendix substance
- Added verbatim interleaving prompt (safe), per-scenario table (tab_scenario), statistical protocol.

## Iter 10 — structural lint
- Verified every result table's column count matches its tabular spec (no LaTeX row-break risk).

## Iter 11 — contributions calibrated to results
- Expanded to 5 contributions incl. robustness (judge-invariance, frontier transfer, defense resilience)
  and the matched slot-vs-csrt isolation; scope now includes the thinking-mode boundary.

## Iter 12 — MIDAS positioning one-liner
- Added the crisp "image channel vs text channel" sentence so reviewers don't read it as "MIDAS for text".

## Iter 13 — numbers consistency audit
- Cross-checked every headline number in prose against results JSON (english/Finnish/csrt/interleave,
  frontier, defense). All match.

## Iter 14 — abstract vs test-split consistency
- Abstract's 0.70 was the full-set sweep peak; main test-split table is 0.693. Reworded abstract to
  "0.69 on held-out test (0.70 at peak load)" to remove the discrepancy.

## Iter 15 — Algorithm 1 (pseudocode)
- Added algpseudocode Algorithm 1 (construction + gated scoring), referenced in Method.

## Iter 16-18 — Prompt templates appendix
- Verbatim (sanitized) templates: interleaving game, inline-slot, CSRT, reconstruction judge,
  safety judge protocol, HR rubric, Self-Reminder defense prompt.

## Iter 19-22 — ICLR appendix sections + LLM-usage
- Added Implementation/compute, Datasets/licenses, Cross-judge agreement, Thinking-mode study,
  and the ICLR-required Use-of-LLMs statement.

## Iter 23-35 — external reviewer (agent) round 1, no-experiment fixes
Reviewer rated 4/10 (salvageable) and flagged real issues; fixed:
- Removed false "strongest on every target" (Phi: translation 0.843 > interleave 0.475 now stated openly).
- Sweet-spot claim softened to target-dependent (abstract + results).
- "frontier" -> "commercial GPT-4o-mini (200-item subset)"; largest-frontier left to future work.
- "no attacker-authored text" -> "no new translated content" (acknowledge fixed game prompt).
- "single-judge artifact" -> "judge-sensitive; cannot resolve over/under-count without human GT".
- p$\approx$0 -> $p<10^{-4}$ throughout.
- Thinking-ON english_direct increase (0.23->0.37) now addressed in text.
- Lingua-SafetyBench provenance footnote + "Text-Dominant" definition added.
- HR: scorer (Qwen2.5-7B) named; report mean AND HR>=3 rate; softened "highest HR" claim.
- Added Table (matched slot_k vs csrt_k, full k=1/2/3 with McNemar) and Table (raw vs gated ASR).
Still OPEN (need experiments): encoding-attack baseline; human validation of the gate; fill InternLM
row (running); a second reconstruction judge; optional true-frontier target.

## Iter 36-42 — table/claim consistency (reviewer round-1 completion)
- tab_main: interleave rows n4/n6 (both computed) — removed pending n10.
- tab_defense: restricted to conditions present in BOTH no-defense and defense runs (n4).
- tab_frontier: item count filled (200); caption/section reworded "commercial" not "frontier".
- targets table: only InternLM row remains \pending (experiment running; auto-fills).
- Global "frontier" wording cleaned; kept only "largest frontier systems = future work".
- Added encoding-attack baseline (Base64 + payload-splitting) to the pipeline and armed
  run_encoding.sh (reviewer-required experiment) to run after InternLM + thinking-full.
OPEN experiments queued: InternLM full row; thinking-full; encoding baseline; (todo) second
reconstruction judge for gate validation; (optional) true frontier target.

## Iter 43-46 — style pass + language-selection rationale (user directives)
- Removed all em-dashes (31) and "but" constructions (10); normalized word en-dashes to hyphens;
  softened AI-ish adverbs (crucially/notably). Academic tone; will re-check periodically.
- Added \subsection{Language selection}: fixed pre-registered order (English anchor + Chinese,
  French, Arabic, Russian, ...), rationale = front-load script/family diversity so small n spans
  distinct writing systems; n selected on dev by gated ASR (n=4 peak, recon 0.95); cited prior
  multilingual-ASR evidence (Yong, Deng, Wang) that diverse/low-resource combinations raise ASR.
