# Willingness fingerprint summary (FalseReject, 9 models)

Per-frame benign **nonrefusal** on FalseReject, fiction-structure rate, benign-best frame,
and 'burned' frames (nonref drops >0.15 below plain — the frame triggers refusal benignly).
Source: willingness_prior_falsereject_<tag>.json. See TRANSFER_FINDING.md for why benign
nonref does NOT predict harmful comply.

| model | plain | persona | fiction | pap | fic_struct(fiction) | best(nonref) | burned frames |
|---|---|---|---|---|---|---|---|
| falcon3_7b | 0.47 | 0.29 | 0.40 | 0.38 | 0.07 | plain | persona |
| gemma2_2b_it | 0.79 | 0.84 | 0.57 | 0.49 | 0.10 | persona | fiction,pap |
| gemma2_9b_it | 0.62 | 0.95 | 0.90 | 0.30 | 0.03 | persona | pap |
| llama31_8b_it | 0.47 | 0.79 | 0.80 | 0.47 | 0.53 | fiction | - |
| mistral24b | 0.83 | 0.99 | 0.99 | 0.96 | 0.07 | persona | - |
| phi35_mini | 0.92 | 0.93 | 0.99 | 0.94 | 0.29 | fiction | - |
| qwen25_14b | 0.93 | 0.49 | 0.94 | 0.90 | 0.23 | fiction | persona |
| qwen25_3b | 0.66 | 0.82 | 0.62 | 0.85 | 0.74 | pap | - |
| qwen25_7b | 0.86 | 0.30 | 0.62 | 0.90 | 0.42 | pap | persona,fiction |
