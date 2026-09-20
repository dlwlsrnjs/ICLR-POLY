# Two-GPU remaining-model queue (2026-09-20)

The eight unfinished models are assigned without overlap:

| GPU | Models |
| --- | --- |
| 0 | llama32_3b_it, mistral7b, phi35_mini, qwen25_32b |
| 1 | mistral24b, phi3_medium_14b, qwen25_14b, qwen25_3b |

GPU 0 retains `willingness331_adaptive_queue_20260920`. A STOP marker makes the original scheduler exit after the active collection stage, preserving all responses. A detached handoff process waits for that scheduler to exit, removes the marker, and resumes using `adaptive_queue_worker.py --gpu 0 --models ...` in the same run. Completed-stage hashes are checked. The previous command is preserved in `process_before_split.json`; assignment and handoff details are in `PARALLEL_HANDOFF.json` and `GPU0_HANDOFF_COMPLETED.json`.

GPU 1 starts immediately in `willingness331_adaptive_parallel_gpu1_20260920` with `adaptive_queue_worker.py --gpu 1 --models ...`. Both use the original run as `--reuse-run` and the frozen transition bank as `--old-run`. See each run's `process.json` for the complete resume command. Never launch duplicate workers on these assignments. The completed three-model recovery remains in `willingness331_adaptive_recovery_gpu1_20260920`; its results supersede the original three failed-judge statuses.

Each worker continues selection, confirmation, then five willingness frames only if the frozen criteria pass. Sampling, prompts, judge rubrics, and selection thresholds are unchanged. Both run independently of the chat session. GPU space is checked between stages; a server restart requires resuming from recorded commands.
