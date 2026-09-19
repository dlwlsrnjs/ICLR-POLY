# Shared English + Arabic prior

User-selected common-language protocol: `g3_ordered_n2`, English + Arabic,
three fragments per language, ordered within each language, canonical five
frames. Both MJ and LG contain Arabic in their full language inventories.
This protocol deliberately differs from canonical MJ n2 (English+Bengali)
and LG n2 (English+Norwegian). Do not merge it into either canonical prior or
reuse canonical benign reconstruction rates as validation of this protocol.
English originals remain byte-for-byte unchanged. MJ and LG can share this
one language-controlled prior; identical inputs do not require duplicate calls.

All 17 pinned target-model configs are under `configs/`. Existing Arabic
translations/backtranslations cover all 331 selected items. Qwen2.5-32B BF16
semantic QA is being run with CPU offloading, retaining full input, output,
source translation hashes and termination state. Failed/invalid/truncated QA
items are excluded from all five frames and retained for subsequent repair.
No failed item is silently treated as a refusal or translation success.

Local execution directory:
`target_prior_only_20260916/runs/willingness331_shared_en_ar_g3_ordered_n2`.
The `code/` snapshot includes the aligned preparation/collection modules and
these two runner scripts. The QA script takes explicit run/source/model paths.
The continuation script waits for the QA and smoke PIDs, prepares all 17 runs,
and starts only the Qwen7B full collection when enough GPU memory remains.
Remaining models and final reconstruction/refusal judging are pending.

`STRUCTURAL_AUDIT.json`: 1,655 actual English+Arabic prompts checked for
fragment roundtrip preservation and identical payloads across frames.
A separate smoke run uses six older, completed Qwen32-QA MiLMMT translations
of matching originals (30 jobs); this overlap subset is not representative.
The smoke is kept separate from the newly QA-selected main cohort.
Model inference and covariance results are not claimed by the structural audit.
Live stage status is saved locally as `pipeline_status.json`, with separate
`smoke_collect.log`, `qa_arabic.log`, and `qwen7b_full_collect.log` files.
