# PolyJigsaw (ICLR-POLY)

## Paper

**Title:** *PolyJigsaw: Prior-Transferred Adaptive Jailbreaks Without Target Data via Verified Multilingual Reconstruction*

**Abstract.** Adaptive jailbreaks typically require access the attacker does not have, such as gradients, logits, labeled failures of the target, or an attack policy fitted to the target's data distribution. We show that none of this is necessary. PolyJigsaw attacks a new language model by transferring a prior over a structured space of attack configurations, estimated once from the responses of other models, and then updating it only from the target's own observed responses. The attacker never sees the target's data or internals; each request starts from the transferred prior and forms its own posterior from a few probes, so selection follows how the target actually reacts. The configuration space is compositional and multilingual: a request is split into semantic fragments, each fragment is rendered in a different language, the fragments are interleaved in order or shuffled, and the model is asked to reassemble the request as a puzzle before answering. No fragment carries the full intent. Success is reconstruction-verified: an attempt counts only when the target both reconstructs the original request and produces unsafe content under fixed judges, which removes spurious successes that inflate raw attack-success rate. We evaluate the full 160-configuration grid on Lingua-SafetyBench and MultiJail across a 17-model panel of open instruction-tuned targets and compare against published attacks under identical inputs, judges, and verification. No fixed configuration dominates across targets; the prior-transferred selector approaches the per-target oracle within a small probe budget without target data or harmful labels; and reconstructing to English re-engages English-space alignment, so separating the reconstruction language from the answer language restores compliance on strongly aligned targets. Code, pinned environments, audit tooling, and aggregate artifacts are released for defensive research.

**개요.** PolyJigsaw는 대상 모델의 데이터나 내부 정보 없이, 다른 모델들의 응답으로 추정한 공격 설정 prior를 새 대상에 옮기고 대상의 실제 응답만으로 갱신해 공격을 고릅니다. 요청을 의미 조각으로 나눠 조각마다 다른 언어로 표현하고 섞어서 제시한 뒤 모델이 원문을 복원해 답하게 하며, 원문 복원과 안전하지 않은 응답을 고정된 판정기로 함께 확인하는 재구성-검증 지표로 성공을 셉니다. Lingua-SafetyBench와 MultiJail, 17개 공개 모델에 대해 160개 설정 전체 그리드를 같은 입력·판정기·검증으로 수집해 기존 공격과 비교합니다.

Adaptive per-target multilingual jailbreaks selected on free plaintext, with a reconstruction-gated
(**verified**) success metric. This repo has the code, aggregate results, and paper. Harmful prompts
and raw model outputs are **not** in the public repo; get them from the private bucket or the official
datasets (below).

> Safety: authorized academic AI-safety evaluation only. Do not commit harmful prompts, raw responses,
> tokens, or local paths. Keep the private bucket private.

**Why the earlier run appeared to beat the baselines by a large margin:** read
[`docs/EXPERIMENT_SETUP.md`](docs/EXPERIMENT_SETUP.md). It separates the actual probe-selected arm
from the post-hoc best of more than 100 evaluated arms, and records the exact old environment.

## 1. Get the data

Two restricted, access-controlled datasets (their harmful text is never public here):

- **Lingua-SafetyBench** (Text-Dominant partition) — official source per its license.
- **MultiJail** (Deng et al.) — official source per its license.

Fastest path for a collaborator with access: pull the working tree (code + small dataset inputs +
aggregate results) from the **private** HF bucket `jin-kwon/poly`. Bucket commands need a
bucket-capable `hf` (huggingface_hub ≥ 1.30). Use a separate transfer environment: upgrading the
locked collection environment changes the measured runtime.

```bash
pip install -U "huggingface_hub[cli]"    # gives `hf sync` / `hf buckets`
hf auth login                            # your HF token (bucket is private)
hf buckets ls -R jin-kwon/poly/PolyJigsaw          # see what's there
hf sync hf://buckets/jin-kwon/poly/PolyJigsaw ./PolyJigsaw   # download whole tree
#   subset only:  hf sync hf://buckets/jin-kwon/poly/PolyJigsaw/scripts ./PolyJigsaw/scripts
```

The bucket holds code, `paper/`, `docs/`, `results/` (aggregates + `_raw`), and the small dataset
inputs under `private_artifacts/{multijail_v1,panel_v2,paper_main}/` (harm grids, benign probes,
resource orders). **Model weights are NOT in the bucket** (re-download target/judge models from HF).

Layout after download (see `experiments_suite/BUCKET.md` for the full map):

| path | contents |
|---|---|
| `private_artifacts/multijail_v1/` | MultiJail `harm_grid.jsonl` (harmful, restricted), `benign_probe.jsonl` (harmless FLORES), `resource_order.json` |
| `private_artifacts/panel_v2/` | Lingua same layout |
| `results/lang_rank_20260905/resource_order.json` | Lingua language order |
| `results/**` | aggregate JSON (+ restricted `_raw*` harmful outputs) |

### Extended 17-model confirmatory panel (2026-09-14, L40S)

`experiments_suite/exp06_confirmatory_selector/` holds the runners, distilled summaries and
family-clustered statistics for the 17-model / 7-family / 160-arm panel, plus the generator
`scripts/make_confirmatory_tables.py` for `paper/tab_confirm_*.tex`. The full multi-megabyte
replay records and selector checkpoints stay in the private bucket under
`PolyJigsaw/0913/L40S-only/`; some runners are still only on the L40S box, listed in
`docs/L40S_CODE_GAP_2026-09-15.md`.

If you do **not** use the bucket, obtain the datasets officially and rebuild the local files:
`python scripts/prepare_lingua_text.py --dataset-root <extracted>` then the pilot builders (see `docs/REPRODUCE.md`).

## 2. Environment

```bash
python3.10 -m venv .venv-collection && . .venv-collection/bin/activate
pip install -r requirements-collection.lock.txt
export HF_HOME=<cache> HF_HUB_OFFLINE=0
# judges: reconstruction = Qwen2.5-7B-Instruct, safety = Qwen3Guard-Gen-8B
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-7B-Instruct', revision='a09a35458c702b33eeacc393d103063234e8bc28'); d('Qwen/Qwen3Guard-Gen-8B', revision='4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb')"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python scripts/verify_repro_env.py --strict-reference --require-clean
```

## 3. Reproduce the paper tables (no GPU; from stored aggregates, minutes)

```bash
python scripts/benign_prior_selection.py     # probe recipe (LOO)  -> tab_sel_prior, app_heldin_probe
python scripts/bandit_bootstrap_ci.py        # bandit + CIs        -> tab_sel_budget, paired deltas
python scripts/permodel_technique_tables.py  # per-model results   -> tab_heldin, tab_heldout, heldin_numbers
python scripts/heldout_selector.py           # held-out transfer   -> tab_heldout, heldout_numbers
python scripts/query_efficiency.py           # query efficiency    -> tab_sel_queryeff
python scripts/query_eff_strat4.py           # stratified strategy -> tab_strat, strat_numbers
python scripts/acq_compare.py                # acquisition compare -> tab_acq, acq_numbers
python scripts/robustness_checks.py          # winner's curse etc. -> tab_sel_robust, robust_numbers
python scripts/domain_analysis.py; python scripts/domain_permutation.py; python scripts/domain_selector.py   # domain (negative) -> tab_sel_domain*
python scripts/make_selector_tables.py       # tab_sel_*.tex + selector_numbers.tex
python scripts/audit_panel_integrity.py      # re-derive every table from the 90 per-model files
```

Build the PDF (Tectonic or XeLaTeX):

```bash
mkdir -p paper/build
tectonic --untrusted --outdir paper/build paper/polyjigsaw_iclr2026.tex
```

Table → producer map lives in `docs/REPRODUCE.md` and `experiments_suite/INVENTORY.md`.

## 4. Experiment suite (new, dataset-separated, collects baselines + ours together)

`experiments_suite/` organizes each experiment in its own folder with **separate MultiJail / Lingua
drivers** (the two datasets have different languages/order/translation-language, so the files differ).

```bash
cd experiments_suite/exp01_closed_compare
python mj.py audit                 # offline: build every arm, duplication report
python mj.py probe                 # HARMLESS: benign plaintext setting-selection -> shortlist
python lg.py probe
python mj.py attack                # HARMFUL: ours(shortlist) + baselines, verified scoring, raw 0600
python lg.py attack
```

Arm space = comprehension cells (frag{3,5,8,12} × {ordered,shuffled} × n) × willingness set
(power set of persona/fiction/pap, stackable, + role) + single-vector baselines. `n` auto-caps to a
dataset's language count. See `experiments_suite/README.md`, `INVENTORY.md`, `ARM_EXPANSION_IMPACT.md`.

## 4b. Big/mid models on a separate GPU box (L40S)

Large targets that don't fit the shared box are collected on a dedicated **L40S** box with a
self-contained runbook in [`bigmodel_l40s/`](bigmodel_l40s/README.md). It reuses the same drivers
(vLLM tensor-parallel is wired via `--tensor-parallel`) and writes to its own `bigmodel_l40s/results/`
so the output merges straight back into the panel.

```bash
git clone https://github.com/dlwlsrnjs/ICLR-POLY.git && cd ICLR-POLY
export HF_HOME=/data/hf_cache HF_TOKEN=<bucket-access token>
bash bigmodel_l40s/scripts/fetch_data.sh      # dataset inputs from the private bucket
python bigmodel_l40s/scripts/verify_env.py    # preflight (GPUs, judges, weights, data)
bash bigmodel_l40s/scripts/run_l40s.sh        # collect the 6 assigned models (both datasets, resumable)
bash bigmodel_l40s/scripts/upload_results.sh  # push results back to the bucket, then merge on the shared box
```

Full env setup, dataset acquisition, per-model TP/util, cautions, and the merge-back path are in
[`bigmodel_l40s/README.md`](bigmodel_l40s/README.md).

## 5. Upload changes back to the bucket

```bash
bash scripts/bucket_sync.sh        # excludes secrets/caches; scans before upload
```
