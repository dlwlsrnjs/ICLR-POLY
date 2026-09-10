# EXP02 Qwen2.5-7B revised-space run status (2026-09-10)

## Scope and inputs

- Target: `Qwen/Qwen2.5-7B-Instruct`
- Space: 292 arms (32 comprehension cells × 9 willingness variants + 4 single-vector references)
- MultiJail: 64 sampled items from the frozen 315-row input
- Lingua-SafetyBench: 40 sampled items from the frozen 250-row input
- Metric: verified ASR = reconstruction-equivalent AND unsafe answer
- Judges: Qwen2.5-7B-Instruct reconstruction judge and Qwen3Guard-Gen-8B safety judge

Controlled benchmark rows and raw generations remain under ignored `private_artifacts/` and
`results/attack/_raw/` paths with mode 0600. They must not be pushed to the public repository.

## Completed checks

- Both collection schemas and translation languages validated (Bengali for MultiJail; Norwegian for
  Lingua); controlled input modes are 0600.
- Offline construction audit: 292 arms for each collection and zero byte-identical prompt groups.
- Harmless 24-item probes completed for both collections.
  - MultiJail: mean benign reconstruction 0.695; selected `g3_shuffled_n6__plain`.
  - Lingua: mean benign reconstruction 0.712; selected `g3_ordered_n2__plain`.
- Four-item end-to-end verified pilots completed for both collections. All aggregate values were
  recomputed from raw `R`, `U`, and `J` rows; no invariant or permission failure was found. These
  pilots validate plumbing only and are not effect estimates.
- Production matrix durable outputs contain 113/292 MultiJail arms and 132/292 Lingua arms. The
  collectors were stopped cleanly at the user's request; one in-flight raw file per collection is
  intentionally excluded from the aggregates and will be overwritten on resume.

## Reliability changes made during the run

- vLLM now exposes the invoking virtualenv's `bin` directory to subprocesses, so FlashInfer can find
  `ninja` even when Python is invoked by absolute path.
- vLLM/FlashInfer caches are routed to `/data1` by the runner instead of the full system partition.
- Resident HF judges use deterministic micro-batches (default 16; lower with
  `--judge-batch-size 8` when GPUs are shared).
- MANIFEST writes are locked, merged, atomic, and persisted after every completed arm. The offline
  `repair-manifest` command recovers registry rows from durable aggregate/raw pairs after interruption.
- Heterogeneity analysis rejects partial matrices by default rather than treating a partial maximum as
  a finished 292-arm oracle.

## Resume

Use a genuinely free GPU pair. Logical `cuda:0` is the target and logical `cuda:1` is both judges.
The commands are idempotent and skip completed aggregate/raw pairs.

```bash
cd /data1/users/ljk98/POLY_AISI
export PATH=/data1/users/ljk98/envs/VLLM-VL-LABEL/bin:$PATH
export HF_HOME=/data1/users/ljk98/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/data1/users/ljk98/runtime_cache/xdg
export VLLM_CACHE_ROOT=/data1/users/ljk98/runtime_cache/vllm
export FLASHINFER_WORKSPACE_BASE=/data1/users/ljk98/runtime_cache/flashinfer

CUDA_VISIBLE_DEVICES=<target>,<judge> \
  /data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python \
  experiments_suite/exp02_panel_collect/collect_mj.py matrix \
  --models Qwen/Qwen2.5-7B-Instruct --judge-device cuda:1 \
  --judge-batch-size 8 --n-items 64 --util 0.45

CUDA_VISIBLE_DEVICES=<target>,<judge> \
  /data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python \
  experiments_suite/exp02_panel_collect/collect_lg.py matrix \
  --models Qwen/Qwen2.5-7B-Instruct --judge-device cuda:1 \
  --judge-batch-size 8 --n-items 40 --util 0.45
```

After both reach 292 arms:

```bash
/data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python scripts/closed_compare.py \
  repair-manifest --root experiments_suite/exp02_panel_collect/results
/data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python \
  experiments_suite/exp03_heterogeneity/analyze.py \
  --root experiments_suite/exp02_panel_collect/results --suffix _mj
/data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python \
  experiments_suite/exp03_heterogeneity/analyze.py \
  --root experiments_suite/exp02_panel_collect/results --suffix _lg
```
