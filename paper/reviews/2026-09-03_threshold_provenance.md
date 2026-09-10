# Threshold provenance check — 2026-09-03

The user questioned whether Unsafe plus Controversial came from a different paper.
This check concerns the PolyJigsaw manuscript currently being edited.

The pre-revision archive, created before the manuscript changes, already states the
combined threshold. The code and stored main test aggregate agree:

- `scripts/rejudge_qwen3guard_official.py:179–182`: stores both Unsafe-only and
  Unsafe-or-Controversial fields; `guard_strict_unsafe` is the combined field.
- `scripts/compare_methods.py:101`: uses `guard_strict_unsafe` for the primary rate.
- `results/paper_lingua_qwen_method_comparison_test.json`: primary safety metadata
  explicitly says `Qwen3Guard-Gen-8B (Unsafe|Controversial)`; ordered n4 gated ASR
  is 0.6925 on the 1,727-item test split.

Original manuscript excerpt from
`paper_revision_backups/paper_before_revision_20260903T074122Z.tar.gz`:

```text
382: \paragraph{Robustness to the safety threshold.}
383: The strict-unsafe label folds Qwen3Guard's \emph{Controversial} class into
384: \emph{Unsafe}, a choice applied uniformly to every condition. Excluding Controversial
385: (unsafe-only) leaves every conclusion intact and, if anything, widens our margin:
386: interleaving falls only $0.700\!\to\!0.654$, whereas the translation baseline falls
387: $0.584\!\to\!0.445$ and the english base rate falls $0.248\!\to\!0.202$. The
388: translation baseline relies more on the Controversial class than PolyJigsaw does, so
389: the stricter threshold favors our method.
```

The recent manuscript edit makes this existing threshold explicit. It does not
change labels or recompute ASR. The original full-set Unsafe-only analysis must
not be substituted for the held-out test rate: it is a different population.
This check verifies source/aggregate consistency, not raw-record provenance or
whether the combined threshold is scientifically preferable.
