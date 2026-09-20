# Matched cross-model comparison

Primary comparisons now require identical English+Arabic n2 inputs for all 17 models: the same 90 QA-eligible selection item IDs, g in {12,8,5,3}, and ordered/shuffled, giving 720 responses per model. The source bank remains 331 items; this is the common selection subset, not a claim that all 331 items were evaluated in each condition. The frozen IDs and target sampling settings are in `matched_n2_contract.json`.

Temperature 0, top_p 1, seed 0, repetition_penalty 1, max_tokens 1024, and max_model_len 4096 match across all 17 configurations. Each target retains its pinned native tokenizer/chat template. Missing/invalid outputs remain explicit unknowns, not task failures. Compare the same cell and paired item IDs across models, not each model's retrospectively selected best cell.

`matched_n2_report.py` rejects item or frame mismatches and excludes n6 and all validation runs. It preserves denominator and coverage counts, including invalid fulfillment judgments. Previous heterogeneous tables and model-specific historical-n curves are supplementary exploratory results, not primary cross-model comparisons. Gemma27B's additional 591 validation responses are never included in its primary 720-response row.

GLM4-9B and Phi3.5-mini n2 backfill inputs have been prepared and validated (720 jobs each). Their run is `willingness331_matched_n2_backfill_20260920`; the uniform n2 source plan is `willingness331_matched_n2_source_20260920`. The existing GPU1 collection queue owns `FOLLOWUP_COMMAND.json`, executed synchronously after its assigned models finish, before it exits. This keeps the fulfillment supervisor from starting its final audit before the backfill finishes. Follow-up completion only means process completion; per-model outcomes remain in the child results.json.

The current Qwen32 fulfillment pass continues unchanged. After that pass, GPU1 resumes its original four-model collection, then runs the n2 backfill. Model-directory links under that queue make new n2 responses discoverable to the final fulfillment pass. Existing n6 responses and labels remain archived and excluded from the primary comparison. No thresholds were relaxed and no stored labels were rewritten.

Latest primary report: `target_prior_only_20260916/runs/matched_n2_comparison_20260920/primary_report.json`. Regenerate it using the contract, the original/recovery/GPU1 run roots as --sources, and the fulfillment audit directory as --audit.
