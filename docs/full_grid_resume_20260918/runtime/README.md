# PolyJigsaw — Full Harmful Arm-Grid Collection (portable)

Collect, for every `(item × arm)` on each target model, whether the multilingual-puzzle jailbreak
**reconstructs** (R) and is **unsafe** (U), with `J = R∧U`. This yields the per-sample / per-model /
per-setting jailbreak grid whose per-sample maximum is the **oracle** used to score the blind
benign-probe search.

- **Arm space** = 160 grid = 32 comprehension cells `frag{3,5,8,12} × order{ordered,shuffled} ×
  langs{2,4,6,8}` × 5 frames `{plain, persona, fiction, pap, persona+fiction}`.
- **Datasets**: `lg` (Lingua, 250 items) and `mj` (MultiJail, 315 items). **Languages differ per
  dataset** and are read from `data/<ds>/…/resource_order.json` (MJ: Bengali/Swahili/Javanese/…,
  LG: Norwegian/Finnish/…). The code already keeps them separate.
- **Judges** (HF, pinned revisions): reconstruction = `Qwen/Qwen2.5-7B-Instruct`; safety =
  `Qwen/Qwen3Guard-Gen-8B`. Target runs on vLLM. English answers.
- **Output**: `full_grid/<ds>/<tag>_<ds>__<arm>.jsonl`, one row per item. **Resumable** — an arm file
  with all items is skipped; safe to Ctrl-C and rerun.

## Layout
```
collect_grid.py        # standalone collector (judges inlined; imports build_puzzle from lib/)
run.sh                 # resumable driver; splits work via MODELS / DATASETS env
lib/run_qwen_interleaving_probe.py   # build_puzzle + split_fragments (stdlib only)
data/model_map.json    # tag -> HF model id, gpu util, tp, big/gated flags   (safe to commit)
data/                  # HARMFUL PROMPTS live here — download from the HF PRIVATE bucket, DO NOT COMMIT
```

## GitHub vs Hugging Face private bucket
- **GitHub (code only, no prompts)**: `collect_grid.py`, `run.sh`, `lib/`, `requirements.txt`,
  `README.md`, `data/model_map.json`. The `data/**/harm_grid.jsonl` and `resource_order.json` files
  are gitignored (harmful content) — keep them OUT of git.
- **HF private bucket (data to download before running)** — place under `./data/`:
  | path (under ./data/) | lines | sha256[:16] |
  |---|---|---|
  | `multijail_v1/harm_grid.jsonl` | 315 | 85c0cfeecfe59d3a |
  | `multijail_v1/resource_order.json` | 13 | 5e637378f209db2e |
  | `panel_v2/harm_grid.jsonl` | 250 | 14a69350e63d4d3c |
  | `lang_rank_20260905/resource_order.json` | 32 | 98fd729726e736d5 |

  e.g. `huggingface-cli download <your-org>/polyjigsaw-data --repo-type dataset --local-dir ./data`

## Models
17 targets in `data/model_map.json` (Qwen2.5 {3,7,14,32}B, Llama-3.1-8B, Llama-3.2-3B,
Gemma-2 {2,9,27}B, GLM-4-9B, Mistral-7B-v0.3, Mistral-Small-24B, Falcon3 {3,7,10}B,
Phi-3.5-mini, Phi-3-medium-14B). Gated ones (google/*, meta-llama/*) need a HF token
(`huggingface-cli login`). Big ones (27B/32B/24B/phi3-medium) auto-place judges on `cuda:1`.

## Run
```bash
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
# download data/ from the HF private bucket first (see table above)
huggingface-cli login    # for gated targets + judges

# whole panel on this box:
bash run.sh
# split across servers — give each box a slice of models:
MODELS="qwen25_7b gemma2_9b_it glm4_9b" DATASETS="lg mj" bash run.sh   # server A
MODELS="gemma2_27b qwen25_32b mistral24b" DATASETS="lg mj" bash run.sh # server B (big; 2 GPUs each)
```
Judges load once per (model,ds). One arm ≈ 30–45 s on a 3B target (40 items); scales with model
size and item count (MJ 315 ≈ 1.6× LG per arm). Full panel ≈ 7.5 GPU-days single-GPU; split across
N boxes to divide wall-clock.

## Rebuild matrices
After collection, per (model,ds): `oracle_U[item] = max_arm U`, `oracle_J[item] = max_arm J`.
Stack arms into `matrix[tag][arm] = [U per item]` to match the existing 40/64-item matrices
(now extended to full 250/315).

## Environment
Pinned: torch 2.6.0, vllm 0.8.5, transformers 4.57.6, tokenizers 0.22.2 (see requirements.txt).
Always pin BLAS threads (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`). Do NOT install
`sentence-transformers` here — it upgrades transformers and breaks vLLM 0.8.5.
