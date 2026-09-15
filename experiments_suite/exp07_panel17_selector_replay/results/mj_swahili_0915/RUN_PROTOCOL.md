# MultiJail Swahili translation-baseline recollection

- Started: 2026-09-15 UTC
- Scope: 17-model L40S panel
- Dataset: MultiJail
- Items per model: 64
- Method: `translated` only
- Translation language: `Swahili`
- Output tag suffix: `_mj_sw`
- Existing Bengali outputs: preserved in their original result tree
- Runner: `bigmodel_l40s/scripts/run_mj_sw_translation_0915.sh`

The output manifest does not currently serialize `tlang`; therefore the distinct tag, this protocol
record, and the queue status are retained as language provenance. `scripts/closed_compare.py`
validates that Swahili exists in every selected MultiJail row before model inference begins.
