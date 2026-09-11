# Finding — does the willingness axis transfer from a benign probe? (honest, n small)

Analysis: [`transfer_validation.py`](transfer_validation.py) → `results/transfer_validation.json`.
Question: can the benign FalseReject willingness fingerprint pick the **harmful-best** willingness
frame on a target, so we never need a harmful willingness sweep on that target?

## Data
Models with **both** a FalseReject willingness prior (benign) and a harmful full-matrix: **qwen7b,
qwen3b, phi35** → 6 model×dataset points. Harmful per-frame = mean of the C=160 arms' score,
marginalized over comprehension cells, for each base frame {plain, persona, fiction, pap}. Channels:
`unsafe` (comply — what willingness should drive), `verified` (recon-gated success), `recon`.

## Result — the naive benign signal does NOT transfer
Benign **nonrefusal** vs harmful **comply (unsafe)** across the 4 frames:

| channel | argmax agreement (benign-best == harmful-best) | mean Pearson | mean Spearman |
|---|---|---|---|
| unsafe (comply) | **1/6 (0.17)** | **−0.41** | −0.37 |
| verified | 2/6 (0.33) | −0.06 | −0.13 |
| recon | 1/6 (0.17) | −0.03 | −0.03 |

Nonrefusal is **anti-correlated** with harmful comply. Why: on benign over-refusal prompts the
compliant-looking frame is often `pap` (high nonref), while the AIM `persona` frame gets *burned*
(nonref collapses) — yet harmfully `fiction`/`persona` are frequently the strongest. So "pick the
frame the model answers most benignly" is the wrong rule.

## Which benign feature, if any, predicts harmful comply?
Pooled Pearson over all frames × model-datasets (benign feature → harmful `unsafe`):

| benign feature | pooled Pearson |
|---|---|
| nonref | −0.277 (misleading) |
| −hedge | +0.197 |
| words | +0.193 |
| **fiction-structure (`fic`)** | **+0.281** (best, but weak) |

The most transferable benign signal is **whether the model actually builds fiction structure**
benignly (`fic`), consistent with the OR-Bench observation that `fiction` genuinely engages qwen7b.
But every effect is weak (|r|≲0.3) at n=3 models.

## What this means for the paper (honest boundary)
- The **comprehension axis benign-transfers** (established: corr ≈0.84 cross-panel); the **willingness
  axis does not** transfer via a single benign feature, and the intuitive one (nonrefusal) is actively
  misleading.
- So the transfer/stealth story should be scoped honestly: the benign target probe fixes the
  **comprehension** setting; the **willingness** frame is best learned on surrogates as a *weak prior*
  (favor `fiction`-type engagement) and, for a strong claim on a specific target, confirmed with a
  **small harmful willingness check** on that target. Report the harmful-query budget for that check
  separately; do not claim a zero-harmful-query willingness selection.
- Do NOT use benign nonrefusal as the willingness selector feature.

## Caveats
n = 3 models (only these have both a FR prior and a harmful matrix today). When the L40S harmful
matrices land for the 6 big models (all already have FR priors), re-run `transfer_validation.py`:
that raises n to ~9 models and will tighten or overturn these numbers. Treat the current values as
directional, not final.
