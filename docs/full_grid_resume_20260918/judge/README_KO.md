# BF16 판정 전용 2단계 파이프라인

이 폴더는 대상 모델 생성과 판정을 분리할 때 사용하는 **판정 단계만** 포함한다. 대상 모델 응답을 새로 생성하지 않는다.

입력 JSONL에는 `item_id`와 다음 중 하나가 있어야 한다.

- `reconstruction`, `answer`
- `[RECONSTRUCTED]`, `[ANSWER]` 섹션을 포함하는 `raw_output`

판정 모델은 기존 수집기와 동일한 revision으로 고정한다.

- 재구성: `Qwen/Qwen2.5-7B-Instruct`
- 안전성: `Qwen/Qwen3Guard-Gen-8B`
- dtype: BF16

두 GPU 실행 예시:

```bash
POLY_JUDGE_BATCH_SIZE=128 \
  ./run_two_gpu.sh \
  lg \
  /path/to/raw/lg \
  /path/to/judged/lg \
  /home/ubuntu/342/jinkwon/poly/polyjigsaw_grid_collect_runtime/data
```

두 프로세스가 입력 파일 목록을 짝수/홀수 인덱스로 나눠 처리한다. GPU 안에서 OOM이 발생하면 배치를 절반으로 줄여 재시도한다. 출력은 임시 파일에 먼저 기록한 뒤 원자적으로 교체한다.

OOM 축소 순서는 기본 설정에서 `128 -> 64 -> 32 -> 16 -> 8 -> 4`이며, 한 번 축소된 배치 크기는 해당 판정 호출이 끝날 때까지 유지한다.

동일한 입력 SHA-256과 판정 모델 revision으로 완료된 파일은 다시 실행해도 건너뛴다. `--overwrite`를 직접 지정하면 다시 판정한다.

## 5개 대상 모델 전체 실행

`run_five_models.sh`는 `qwen25_14b`, `qwen25_32b`, `gemma2_27b`, `mistral24b`, `phi3_medium_14b`의 LG/MJ raw 응답을 GPU 0/1에서 병렬 생성한 뒤, 생성이 끝나면 데이터셋별 판정을 두 GPU에 반씩 나눠 실행한다.

```bash
nohup ./run_five_models.sh > full_grid_5models.log 2>&1 &
```

기본 캐시는 `/home/ubuntu/342/jinkwon/hf_cache`, 결과 루트는 `target_prior_only_20260916/full_grid_5models_bf16`이다. 환경 변수 `HF_HOME`, `POLY_RUN_ROOT`, `POLY_GPU0_MODELS`, `POLY_GPU1_MODELS`, `POLY_DATASETS`로 덮어쓸 수 있다. raw 파일과 판정 파일은 각각 `raw/<dataset>` 및 `judged/<dataset>`에 저장된다.

## 생성 truncation 재시도

각 `(모델, 데이터셋)`의 160개 arm 생성이 모두 끝난 직후 출력 길이 제한으로 끝난 행만 다시 생성한다.

- 일반 생성은 모델 컨텍스트의 절반을 출력 한도로 사용한다.
- `finish_reason=length` 행만 남은 컨텍스트가 허용하는 최대 출력 길이로 한 번 재시도한다.
- 재시도 결과는 해당 JSONL의 행만 바꾼 뒤 파일 전체를 원자적으로 교체한다.
- 결과 행에는 입력/출력 토큰 수, 종료 사유, 재시도 횟수를 기록한다.
- 요약은 `raw/<dataset>/<tag>_<dataset>__truncation_retry_summary.json`에 기록한다.
- 입력 자체가 모델 컨텍스트보다 길면 조용히 자르지 않고 `prompt_length` 오류로 명시한다.


## 자동 복구 watchdog

`watch_five_models.sh`는 실행 중인 전체 큐와 HF Bucket 동기화 프로세스를 함께 감시한다.

- 기본 30초마다 PID와 실제 명령행을 확인한다.
- 수집/판정 데몬이 사라지면 완료 JSONL을 유지한 채 `run_five_models.sh`를 다시 시작한다.
- HF 동기화 데몬이 사라지면 최종 receipt 유무를 확인한 뒤 다시 시작한다.
- raw 1,600개, judged 1,600개, metadata 1,600개와 최종 HF receipt가 모두 확인되면 종료한다.
- 데이터셋 생성은 기본 3회 자체 재시도하며, 모두 실패하면 판정을 시작하지 않고 watchdog에 제어를 돌려준다.

의도적으로 전체 작업을 중단할 때는 자동 재시작을 막기 위해 watchdog을 먼저 종료해야 한다.
