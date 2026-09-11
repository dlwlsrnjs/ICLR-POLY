# L40S restricted-output upload record

The L40S full-matrix collector keeps raw per-response outputs out of public Git history. They are
stored in the owner-only Hugging Face bucket at:

`hf://buckets/jin-kwon/poly/PolyJigsaw/experiments_suite/exp02_panel_collect/results`

Checkpoint verified on 2026-09-11 at 16:26 UTC:

- 62 completed L40S arm aggregates uploaded.
- The matching 62 restricted `_raw/*.jsonl` files uploaded.
- The remote bucket listing contained 125 entries including the collection status file.
- A post-run sync is scheduled to upload the complete result tree after the matrix runner exits.

The public repository records aggregate scores, provenance, and this receipt only. Access-controlled
prompts and raw model responses remain in the private bucket.
