# Bottleneck-gated willingness intervention

This pilot tests a conditional claim: willingness framing should be applied
where the comprehension transformation already succeeds but plain verified ASR
still fails. It should not be applied uniformly to every comprehension cell.

For model/dataset tag `m`, comprehension cell `c`, and willingness frame `w`:

```text
observed_need(m,c) = max(0, recon(m,c,plain) - verified(m,c,plain))
effect(m,c,w)      = verified(m,c,w) - verified(m,c,plain)
```

The intervention frame is selected per model/dataset using the other 31
comprehension cells. The evaluated cell therefore does not select its own
frame. Item-level matrices additionally measure rescue and backfire rather than
only comparing aggregate means.

Run:

```bash
/home/jklee/safe_mental-dpo/ICLR-POLY/.venv/bin/python \
  experiments_suite/exp09_bottleneck_gated_willingness/analyze_intervention.py
```

The result is written to
`results/bottleneck_willingness_intervention.json`.
Pass `--include-rows` when a full cell/fold-level artifact is needed; the
checked-in default contains compact summaries and a source-manifest hash.

The observed need score is a retrospective diagnostic because it includes
attack-evaluation outcomes. A deployable low-budget gate must estimate it from
the harmless comprehension prior plus a small, held-out calibration budget;
the present benign willingness non-refusal score is not a sufficient proxy.

The result file also contains a two-fold item cross-fit. In each direction,
one half of the items estimates the bottleneck and selects a model-specific
frame using the other 31 cells; the disjoint half measures intervention effect.
This avoids using the same plain outcomes in both the need estimate and the
effect measurement.

Two effects are reported: forced intervention selects among the four non-plain
frames, while the model gate also allows `plain`. The latter tests whether a
model-specific gate can avoid negative transfer when willingness framing is not
helpful.
