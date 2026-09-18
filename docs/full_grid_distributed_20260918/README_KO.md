# 160-arm MJ/LG 남은 큐 분산 실행 인수인계

이 디렉터리는 `full_grid_20260917_v1`의 **아직 시작하지 않은 16개 `(model, dataset)` 단위**를 다른 GPU 서버에서도 동일한 프로토콜로 실행하기 위한 인수인계다. GitHub에는 코드·모델 ID·해시·진행 메타데이터만 둔다. 원문 입력, 모델 응답, judge 출력은 커밋하지 않고 비공개 Hugging Face bucket에만 저장한다.

정확한 분할은 `queue_manifest.json`이 기준이다. 체크포인트 시점의 전체 큐는 11모델 × 2데이터셋 = 22단위이며 5단위가 receipt 완료, Llama-3.1-8B/LG가 진행 중, 그 뒤 16단위가 남아 있다. Qwen2.5-3B는 이 11모델 큐에 포함되지 않는다.

## 중복 실행 방지 조건

현재 L40S 프로세스는 시작 시점에 전체 모델 tuple을 메모리에 올렸기 때문에 파일만 수정해도 자동으로 범위가 줄지 않는다. 외부 shard를 시작하기 전에 L40S queue를 다음 네 모델만 포함하도록 안전하게 재개해야 한다.

```text
llama31_8b_it,gemma2_2b_it,gemma2_9b_it,phi35_mini
```

진행 중인 Llama/LG는 arm별 atomic checkpoint로 재개할 수 있다. 다만 실행 중인 프로세스를 무조건 kill하지 말고, 현재 unit이 receipt된 직후 재실행하거나 foreground에 SIGINT를 보내 child 정리 여부를 확인한 다음 재개한다. `queue_manifest.json`의 외부 shard 상태가 `ready_after_local_queue_constraint`인 이유가 이 조건이다.

## 동일하게 고정되는 실험 조건

- arm: `g={3,5,8,12} × ordered/shuffled × n={2,4,6,8} × 5 frames = 160`
- MJ: 315문항, Bengali 시작 언어 순서
- LG: 250문항, Norwegian 시작 언어 순서
- target: BF16, context 8192, greedy, 최대 출력 1024 tokens
- reconstruction judge: `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
- safety judge: `Qwen/Qwen3Guard-Gen-8B@4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb`
- 환경: vLLM 0.8.5, transformers 4.57.6, tokenizers 0.22.2, torch 2.6.0+cu124
- FalseReject와 4개 method baseline은 포함하지 않는다.

2GPU worker는 target과 reconstruction을 병렬 실행한 뒤 judge GPU를 Guard로 재사용한다. 3GPU worker는 target/reconstruction/Guard를 동시에 파이프라인한다. GPU 번호는 컨테이너 내부의 추정 번호가 아니라 해당 호스트에서 `nvidia-smi`에 표시되는 번호를 넘긴다.

## 필요한 비공개 입력

저장소 루트 기준으로 아래 파일을 복원한다. 내용은 GitHub에 올리지 않는다.

| 경로 | SHA-256 |
|---|---|
| `private_artifacts/multijail_v1/harm_grid.jsonl` | `85c0cfeecfe59d3abee90640c076e0b66808e7f3cd54d524a76fae4b18faab75` |
| `private_artifacts/multijail_v1/resource_order.json` | `5e637378f209db2e656f28817416a42b4fd2624206c0cf28ddb83c7199e9e06d` |
| `private_artifacts/panel_v2/harm_grid.jsonl` | `14a69350e63d4d3c47f497de5087cf4ec2a3685085af37443a810a70c772ae62` |
| `results/lang_rank_20260905/resource_order.json` | `98fd729726e736d547b4a431fc827765d9516e9ddf70536adc21098589045446` |

두 judge의 고정 revision도 `HF_HOME`에 먼저 받아 둔다. gated Llama/Gemma 접근 권한과 비공개 bucket 쓰기 권한이 있는 HF 로그인은 서버별로 별도 설정한다. 토큰 문자열을 명령행, 로그 또는 Git에 기록하지 않는다.

## 실행 방법

브랜치를 받은 뒤 고정 환경의 Python과 bucket 지원 HF CLI 경로를 지정한다. 먼저 `plan`, 그 다음 `preflight`, 마지막에 `run`을 실행한다.

```bash
export PY=/absolute/path/to/vllm085/bin/python
export HFCLI=/absolute/path/to/hf
export HF_HOME=/absolute/path/to/poly-hf-cache
export HF_TOKEN_PATH=/absolute/path/to/huggingface/token

$PY bigmodel_l40s/scripts/distributed_full_grid_worker.py plan \
  --owner dist-shard-a \
  --models mistral7b,falcon3_7b \
  --datasets mj,lg \
  --gpus 0,1,2 \
  --hf-bin "$HFCLI" --hf-home "$HF_HOME" --hf-token-path "$HF_TOKEN_PATH"

$PY bigmodel_l40s/scripts/distributed_full_grid_worker.py preflight \
  --owner dist-shard-a \
  --models mistral7b,falcon3_7b \
  --datasets mj,lg \
  --gpus 0,1,2 \
  --hf-bin "$HFCLI" --hf-home "$HF_HOME" --hf-token-path "$HF_TOKEN_PATH"

$PY bigmodel_l40s/scripts/distributed_full_grid_worker.py run \
  --owner dist-shard-a \
  --models mistral7b,falcon3_7b \
  --datasets mj,lg \
  --gpus 0,1,2 \
  --hf-bin "$HFCLI" --hf-home "$HF_HOME" --hf-token-path "$HF_TOKEN_PATH"
```

2GPU 서버는 `--gpus 0,1`처럼 두 번호만 넘긴다. 다른 shard는 `queue_manifest.json`의 `owner`와 `models`를 그대로 사용한다. 임의로 owner를 공유하거나 동일 모델을 두 shard에 넣지 않는다.

각 worker의 기본 로컬 경로와 bucket 경로는 다음처럼 분리된다.

```text
<repo>/full_grid_20260917_v1/<owner>/
hf://buckets/jin-kwon/poly/PolyJigsaw/full_grid_20260917_v1/<owner>/
```

한 `(model,dataset)`의 160 arms가 row-level audit를 통과한 뒤에만 immutable snapshot과 receipt가 업로드된다. 실행이 끊겨도 `_pipeline/<protocol-hash>/`의 atomic stage 파일에서 재개한다. receipt 없는 unit은 완료로 간주하지 않는다.

## 완료 판정과 후속 repair

Primary 완료 조건은 shard별 모든 unit의 local receipt와 bucket receipt가 존재하고, audit에서 160 arms 및 예상 행 수가 모두 맞는 것이다. 1024-token primary snapshot은 수정하지 않는다. 길이 종료 행의 2048-token repair는 모든 primary owner의 완료 receipt를 모은 뒤 별도 snapshot으로 실행한다. Primary와 repair 결과를 덮어쓰거나 같은 receipt namespace에 섞지 않는다.

이 배포의 실행기는 primary 분산 수집을 담당한다. 기존 `truncation_repair_20260917.py`는 원래 단일 owner 경로를 전제로 하므로, 외부 shard 결과를 중앙으로 합치기 전에는 그 스크립트를 외부 서버에서 임의 실행하지 않는다.
