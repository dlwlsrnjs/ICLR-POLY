# GPU 1 recovery queue

`adaptive_queue_worker.py` uses the same scientific selection and confirmation rules as `adaptive_queue.py`, with explicit `--gpu`, `--models`, and `--reuse-run` arguments. Use a separate `--run` directory and assign only models that the other active worker will not revisit. Do not run overlapping model assignments or restart the original queue over these models while recovery is active.

On 2026-09-20, GPU 1 was assigned `qwen25_7b falcon3_10b falcon3_3b`, whose original runs failed at WildGuard initialization under shared GPU memory pressure. All 720 selection responses per model were verified against the full prompt/configuration and reused (2160 total). Judges run afresh with provenance in the new directory; inference settings and decision thresholds are unchanged. Successful confirmation automatically continues to the five willingness frames. Models without a confirmed transition are reported as such.

Runtime: `target_prior_only_20260916/runs/willingness331_adaptive_recovery_gpu1_20260920`. The original run records `RECOVERY_HANDOFF.json`; recovery records `WORKER_ASSIGNMENT.json`, `process.json`, source hashes, logs, and stage completion markers. Original failed statuses remain historical and must be read alongside recovery results. The detached process survives chat closure but not server restart. Resume using the command in `process.json` after ensuring no instance is active.

GPU 0 continues the original queue. GPU 1 retains memory headroom and checks availability between stages. Other users' processes are left running.
