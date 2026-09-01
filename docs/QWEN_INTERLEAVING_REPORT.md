# Qwen Parallel-Language Interleaving — reconstruction-difficulty sweep

마지막 갱신: 2026-08-28

이 문서는 `docs/INTERLEAVING_PUZZLE_HANDOFF.md`의 §12 "Qwen/HF 이전"과 §11 "언어 수
2/4/6/8/10 ablation"을 실제로 수행한 결과다. GPT-4o mini 파일럿을 로컬 open-weights
Qwen으로 이식하고, interleaving 부하(언어 수 × ordered/shuffled)에 따른 **의미
재구성 난이도 곡선**을 측정했다.

## 무엇을 측정했고 무엇을 측정하지 않았나

- 대상 모델: `Qwen/Qwen2.5-7B-Instruct` (local, bf16, greedy/temperature 0).
- probe는 재구성된 요청을 **수행하지 않는다**. fragment 재그룹핑, 순서 복원, 영어
  재구성, toxicity label만 출력한다. 따라서 이 실행 자체는 탈옥을 유발하지 않으며
  benign 데이터로도 안전하게 반복 가능하다.
- 데이터: FLORES-200 dev (benign 병렬 문장, false-positive/재구성 난이도 control).
  공식 tarball에서 받아 10개 언어 열 parquet로 구성.
- **미측정(차단됨):** reconstruction-gated ASR의 유해 arm. Lingua-SafetyBench 원문은
  이 머신에 없고 access-controlled다. 코드는 `--source lingua`로 준비되어 있으며,
  사용자가 데이터셋을 배치한 뒤에만 실행한다(아래 재현 절 참고).

## 연구 프레이밍: 재구성 난이도 ↔ 과부하의 절충

핵심 가설은 "정확한 의미 재구성 이후에도 안전 정렬이 실패하는가"이다. 이를 주장하려면
두 조건이 동시에 성립하는 부하 지점이 필요하다.

1. 모델이 실제로 의미를 복원한다 (semantic reconstruction 높음).
2. 그 부하가 안전 거부를 약화시킨다 (gated ASR > 직접 요청 C0).

부하가 너무 크면 (1)이 무너져 gated 지표가 0이 되고(기존 C3=0/80, K1 등에서 관측),
너무 작으면 wrapper 효과만 남는다. 따라서 먼저 benign에서 **(1)이 유지되는 최대
부하**를 찾아 두는 것이 유해 실험 예산을 쓰기 전의 올바른 순서다. 이 문서는 그 (1)
곡선을 제공한다.

## 결과: benign FLORES 재구성 난이도 곡선

각 셀 = 20개 항목, coarse granularity(언어당 5조각), seed 20260828.
지표는 모두 무해 집계값이다.

| 언어 수 | 조건 | JSON parse | English exact | fragment 배정 정확도 | frag coverage | all-lang exact |
|---:|---|---:|---:|---:|---:|---:|
| 2 | ordered | 1.00 | 0.55 | 0.79 | 0.97 | 0.00 |
| 2 | shuffled | 1.00 | 0.10 | 0.74 | 0.93 | 0.00 |
| 4 | ordered | 1.00 | 0.20 | 0.61 | 0.88 | 0.00 |
| 4 | shuffled | 1.00 | 0.10 | 0.43 | 0.72 | 0.00 |
| 6 | ordered | 1.00 | 0.10 | 0.46 | 0.74 | 0.00 |
| 6 | shuffled | 1.00 | 0.05 | 0.34 | 0.69 | 0.00 |
| 8 | ordered | 1.00 | 0.15 | 0.43 | 0.77 | 0.00 |
| 8 | shuffled | 1.00 | 0.05 | 0.33 | 0.77 | 0.00 |
| 10 | ordered | 1.00 | 0.10 | 0.39 | 0.77 | 0.00 |
| 10 | shuffled | 1.00 | 0.00 | 0.29 | 0.66 | 0.00 |

지표 정의:
- `json_parse_valid`: 엄격 JSON 파싱 성공률(구조 준수).
- `english_text_output_exact`: 재구성 영어가 원문과 정규화 후 정확 일치한 비율.
- `fragment_assignment_accuracy`: gold fragment 수를 분모로 한 언어 배정 정확도.
- `all_languages_exact`: 10개 언어 전부의 조각 순서를 정확히 복원한 비율.

## 해석

- **JSON 구조 준수는 부하와 무관하게 100%** (max_new_tokens=2048로 truncation 제거 후).
  즉 실패는 형식이 아니라 재구성 능력의 문제다.
- **재구성 난이도는 언어 수에 단조 증가한다.** English exact reconstruction은
  2언어 ordered 0.55에서 10언어로 갈수록 급락하고, shuffled은 2언어에서도 0.10으로
  훨씬 낮다. fragment 배정 정확도도 0.79→0.29로 단조 감소한다.
- **ordered ≫ shuffled.** 표시 순서 단서를 제거하면 같은 언어 수에서도 재구성이
  일관되게 낮다. 이는 부하의 상당 부분이 언어 식별이 아니라 순서 추론에서 온다는
  기존 가설과 일치한다.
- **all-language exact는 전 구간 0.00.** 10개(또는 그 이하) 병렬 문장을 조각까지
  완벽히 복원하는 것은 7B 모델에는 사실상 불가능하다. 따라서 안전 실험의 gate는
  전체-정확 복원이 아니라 **English 의미 등가**(semantic equivalence) 기준이어야 한다.
- **함의(절충점):** benign 기준으로 English 재구성이 실용적으로 유지되는 상한은
  대략 **2~4언어 ordered** 구간이다. 유해 arm에서 gated ASR을 주장하려면 이 구간을
  우선 후보로 삼고, 그보다 부하를 올린 6~10언어·shuffled는 "raw ASR은 오르나 재구성
  실패로 gated는 붕괴"하는 기존 C3/K1 패턴을 재현할 가능성이 높다. 즉 이 곡선이 유해
  실험에서 탐색할 K* 범위를 사전에 좁혀 준다.

## 재현

benign 곡선(이 문서):

```bash
export HF_HOME=/home/ubuntu/342/jinkwon/hf_cache CUDA_VISIBLE_DEVICES=0
python3 scripts/run_interleaving_sweep.py \
  --source flores \
  --flores-dataset /home/ubuntu/342/jinkwon/datasets/flores200/dev.parquet \
  --num-items 20 --granularity coarse \
  --language-counts 2 4 6 8 10 --conditions ordered shuffled \
  --workdir private_artifacts/interleaving_sweep_flores \
  --summary results/qwen_interleaving_flores_sweep_summary.json
```

유해 arm(데이터셋 배치 후에만):

```bash
# 1) 공식 절차로 Lingua-SafetyBench를 받아 정렬 텍스트를 만든다
python3 scripts/prepare_lingua_text.py \
  --dataset-root /path/to/Lingua-SafetyBench/extracted/dataset
# 2) 같은 sweep을 --source lingua로 실행
python3 scripts/run_interleaving_sweep.py --source lingua \
  --lingua-dataset ../datasets/lingua_safetybench_text/lingua_aligned_text_all.jsonl \
  --num-items 20 --granularity coarse \
  --language-counts 2 4 6 8 10 --conditions ordered shuffled \
  --workdir private_artifacts/interleaving_sweep_lingua \
  --summary results/qwen_interleaving_lingua_sweep_summary.json
```

## 다음 단계

1. 유해 arm을 동일 grid로 실행해 toxicity recall과 (benign 대비) 재구성 난이도를 비교.
2. benign에서 재구성 ≥80%가 유지되는 최대 언어 수 K*를 고른 뒤, K*에서만
   reconstruction-gated 안전 실험(응답 수행 포함)을 별도 승인·독립 judge로 수행.
3. coarse/fine granularity ablation과 seed 다중화로 방향 안정성 확인.
