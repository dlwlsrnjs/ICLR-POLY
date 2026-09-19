# Confirmed 331-item benign over-refusal bank

The user confirmed that “311” referred to the existing **331** selected items:
FalseReject 212 + OR-Bench Hard 119. Selection requires at least 5 refusals among
9 screened models. This is a selected over-refusal cohort, not an unbiased
sample of benign requests; non-refusal is not the same as correct fulfillment.

The original 394-item archive remains historical. Use the configs in this
folder for the current panel. All 17 pinned model identities and revisions are
in `models.json`; 34 configs fix `g3_ordered_n2` and the canonical five frames.
MJ uses English+Bengali; LG uses English+Norwegian. Each profile plans 28,135
responses (331 × 5 × 17). Changing the model does not change the puzzle payload.

## Reproduce the connection

From the repository root, set `SOURCE` to the local
`willingness_overrefusal331_17models_20260918` directory and `MODELS` to the
17-model reconstruction manifest:

```bash
ALIGNED=experiments_suite/exp08_willingness394_milmmt_v3/aligned
python3 "$ALIGNED/connect_curated_bank.py" --source "$SOURCE" --models "$MODELS" --out "$OUT/inputs"
python3 "$ALIGNED/audit_curated_bank.py" --inputs "$OUT/inputs" --out "$OUT/CPU_AUDIT.json"
```

The connector verifies the exact frozen source file hash and retains original
text, source IDs, categories, screening counts and rates. Item hashes in each
config prevent accidentally running another same-sized corpus. The generated
translation queue preserves available forward/backtranslation texts and
provenance. Local connected inputs are at:
`target_prior_only_20260916/runs/willingness331_g3_ordered_n2_panel17/inputs`.

## Actual validation and outstanding work

`CPU_AUDIT.json` records 1,655 real LG puzzle constructions: English and
Norwegian fragments reassemble to their source text after whitespace
normalization; all five frames share the same payload. All 34 configurations
correctly block production collection without accepted translations.
This checks the builder, not a model's reconstruction ability.

There are 331 Norwegian translations awaiting semantic QA and 331 Bengali
translations still needed. Translation/backtranslation existence alone does
not certify semantic equivalence. Do not manually relabel these translations
as QA-accepted. After real QA, supply an independently saved accepted-translation
file to `prepare.py`; preserve QA evidence and its hashes alongside it.

```bash
python3 "$ALIGNED/prepare.py" --config "$OUT/inputs/configs/lg_qwen25_7b.json" --items "$OUT/inputs/items.json" --translations "$QA_ACCEPTED" --out "$OUT/lg/qwen25_7b"
```

Every generated job now includes the full prompt, user messages, fragment
records, gold ordering, original English and full source translations. The
collector preserves model responses and termination information; the existing
judge/summarizer requires valid reconstruction and answer judgments.

**No new target-model inference, willingness estimate, or covariance result is
claimed by this connection audit.** Old `g5_ordered_n4` responses are not reused
as `g3_ordered_n2` results. Runtime compatibility of all 17 models is still to be
verified. The saved configurations specify model identities, not evidence that
all 17 model runtimes have passed.
