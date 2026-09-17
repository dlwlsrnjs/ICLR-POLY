# MJ/LG 전량 실험 세팅 그리드 — L40S 서버 인계

## 현재 상태: 배정안 작성 완료, 실행 중 작업과 대조 필요

이 문서는 **315/250문항 × 160실험 세팅 × 17모델**의 응답과 판정을 저장해 오프라인 실험에서 조회하기 위한 인계 자료입니다. 기존 코드는 실험 세팅을 `arm`이라고 부르지만 여기서 수집 단위는 `(dataset, item_id, setting_id, model_revision, protocol_hash)`입니다.

**이 문서만 보고 현재 작업을 중단하거나 중복 실행하지 마세요.** 문서 작성 시 접속 서버의 H100 2장은 유휴 상태였고 전량 수집 프로세스는 없었습니다. 사용자가 말한 실행 중 작업의 서버·명령·로그·환경, 추가 L40S 개수는 아직 전달되지 않았습니다. 따라서 `work_plan.json`은 `DRAFT_PENDING_ACTIVE_RUN_RECONCILIATION`입니다. 현재 실행 환경을 확인했다고 주장하지 않습니다.

문서 기준 GitHub base commit: `d000cd4781030bd1612b28b962efee4680306105`. 이 자료를 수정한 최종 commit도 실행 provenance에 별도로 기록하세요. 기존 작업 폴더와 환경은 건드리지 않습니다.

## 1. 담당 범위

모델 단위로 나눕니다. 한 모델을 두 서버에서 동시에 수집하지 않습니다. 각 담당자는 해당 모델에서 **MJ 0–314 전체, LG 0–249 전체, 세팅 0–159 전체**를 맡습니다. 위치는 고정된 입력 파일의 순서이며 실제 병합 키는 `item_id`입니다. 현재 돌아가는 모델이 아래 배정과 겹치면 **진행 중 모델을 기존 서버에 남기고** 작업 목록의 owner를 조정한 뒤 시작합니다.

| 담당 | 모델 | 모델 수 | 응답 수 |
|---|---|---:|---:|
| `primary_large` | Qwen2.5-14B, Qwen2.5-32B, Gemma2-27B, Mistral-Small-24B, Phi-3-medium-4k | 5 | 452,000 |
| `l40s_small` | Qwen2.5-3B/7B, Llama3.2-3B, Llama3.1-8B, Gemma2-2B/9B, Phi3.5-mini, Mistral7B, Falcon3-3B/7B/10B, GLM4-9B-chat-hf | 12 | 1,084,800 |
| 합계 | | 17 | **1,536,800** |

정확한 HF 모델 ID와 태그, 세팅 이름은 `work_plan.json`이 기준입니다. 이 배분은 모델 크기에 따른 초안이며 실측 균등 시간 배분은 아닙니다. L40S가 추가되면 작은 모델 목록을 서로 겹치지 않는 부분집합으로 다시 나눕니다. GPU 수나 TP 수를 응답 수의 선형 배속으로 간주하지 않습니다.

```bash
# GPU/모델 호출 없는 배정 검사. 활성화 상태까지 보증하지는 않음.
python docs/full_grid_parallel_20260917/audit_plan.py
```

160세팅 = 조각 수 {3,5,8,12} × 배열 {ordered,shuffled} × 언어 수 {2,4,6,8} × 프레임 {plain,persona,fiction,pap,persona+fiction}. baseline 4개는 이 160개에 **포함하지 않습니다**. 기존 `--arm-space C` 경로는 baseline까지 164개를 만들 수 있으므로 그대로 총량 160으로 보고하면 안 됩니다.

기존 17모델 결과는 MJ 64문항·LG 40문항입니다. 315/250 전량 완료본으로 재사용하면 안 됩니다. 동일 프로토콜·입력·ID·개별 응답을 확인한 경우에만 해당 문항을 부분적으로 재사용합니다.

## 2. 환경: 확인된 값과 아직 확인할 값

`observed_environment.json`은 이 작업에서 사용한 **무해 prior 환경**의 실제 설치값입니다. 실행 중인 별도 유해 그리드 환경을 캡처한 파일이 아닙니다.

| 항목 | 실제 관측값 |
|---|---|
| Python | 3.10.12 |
| vLLM | 0.8.5 |
| PyTorch / torchvision | 2.6.0 / 0.21.0 |
| transformers / tokenizers | 4.57.6 / 0.22.2 |
| huggingface-hub | 0.36.2 |
| 관측 드라이버 | 570.124.06 |

`requirements-core-reference.txt`는 비교용이며 완전한 재현 환경 설치 파일이 아닙니다. 예를 들어 관측한 prior 환경에는 `accelerate` 배포 메타데이터가 없고, 기존 HF 기반 판정기 경로는 별도 의존성이 필요할 수 있습니다. 활성 실행 서버에서 아래 정보를 받아 그 환경을 복제한 뒤 사용하세요.

```bash
# 실제 실행 중 프로세스가 사용하는 python으로 실행; 셸의 임의 python을 쓰지 말 것
/path/to/active/venv/bin/python -V
/path/to/active/venv/bin/python -m pip freeze > active-environment.freeze.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

추가로 저장할 것: 실행 명령(토큰 제외), 코드 commit, 대상·판정 모델 revision, chat template 해시, 입력 SHA-256, 출력 토큰 한도·seed·temperature·repetition penalty, `POLY_STRONG_RECON`, 판정 규칙, 논리/물리 GPU 매핑. `pip freeze`에 private URL 자격증명이 있는지 확인하고 GitHub에는 올리지 않습니다.

**기존 저장소 lock 파일을 그대로 설치하지 마세요.** base commit의 `requirements-collection.lock.txt`는 vLLM0.28.0/transformers5.16.1로, 위 실제 관측 환경과 다릅니다. 기존 `fetch_data.sh`도 활성 환경의 huggingface-hub를 업그레이드할 수 있습니다. 생성·판정 환경을 변경하면서 기존 결과와 섞지 않습니다.

HF 버킷 CLI는 별도 가상환경에 설치합니다. 아래 설치는 생성 환경을 건드리지 않습니다.

```bash
python3 -m venv .venv-hf-sync
.venv-hf-sync/bin/python -m pip install 'huggingface_hub==1.6.0'
.venv-hf-sync/bin/hf buckets sync --help
.venv-hf-sync/bin/hf auth login
```

토큰은 로그인 또는 환경변수로만 전달하고 코드·문서·실행 로그에 넣지 않습니다. 새 서버의 모델 접근 권한은 별도로 확인해야 합니다.

## 3. L40S GPU 배치

L40S 수량은 아직 미확인입니다. **2장 이상일 때 작은 모델 1장 + 판정기 1장**을 우선 후보로 삼고 실제 메모리 사용을 확인합니다. 예를 들어 물리 `0,1`을 노출하면 논리 `cuda:0`이 대상 모델, `cuda:1`이 판정기입니다. 이는 실행 검증을 마친 배치가 아닙니다.

1장뿐이면 대상 모델과 판정기를 동시에 올릴 수 있다고 가정하지 말고 생성/판정을 순차 실행하는 수집기 지원을 확인해야 합니다. 여러 타깃 프로세스가 같은 판정 GPU에 각각 판정 모델을 복제 적재하지 않도록 합니다. 큰 모델 5개는 우선 주 서버에 남깁니다.

## 4. 입력과 프로토콜 고정

- MJ: `private_artifacts/multijail_v1/harm_grid.jsonl`, 315개. 언어 순서는 같은 폴더의 `resource_order.json`.
- LG: `private_artifacts/panel_v2/harm_grid.jsonl`, 250개. 언어 순서는 `results/lang_rank_20260905/resource_order.json`.
- 입력별 중복 없는 `item_id`와 파일 SHA-256을 양 서버가 비교합니다. 원격 최신 파일이라는 이유만으로 실행 중 입력을 교체하지 않습니다.
- 판정은 `R`(재구성), `U`(응답 유해성), `J`(기존 프로토콜의 결합 판정)를 각각 보존하고 성공 지표 정의를 명시합니다. 원문 요청, 응답, 잘림 여부, 파싱 실패 여부가 판정과 연결돼야 합니다.
- 기존 `online_live.py` 기본 생성 길이는 320토큰입니다. 새 FalseReject 수집의 1,024토큰이나 WildGuard 판정과 자동으로 같다고 간주하지 않습니다.
- 기존 runner의 기본 모델 패널은 이번 검증된 17모델 목록과 다릅니다. `--set all`로 대체하지 말고 `work_plan.json`의 명시적 ID/태그를 사용합니다.

## 5. 실행 중 기존 작업과 맞추는 순서

1. 기존 서버의 실행 명령·로그·출력 루트·완료/진행 중 모델을 확인합니다.
2. 진행 중 단위는 기존 서버 owner로 남기고, L40S에는 미시작 모델만 배정합니다. 전체 34개 `(model,dataset)` 단위에 owner가 정확히 하나인지 검사합니다.
3. 양 서버의 protocol 파일과 입력 해시가 일치하는지 확인합니다. 일치하지 않으면 같은 run_id 아래 결과를 섞지 않습니다.
4. 이번 실행에 사용할 모델 revision을 고정합니다. 다운로드한 최신 가중치로 과거 결과를 채우지 않습니다.
5. 기존 수집기가 정확히 160세팅을 대상으로 하며 315/250개를 요구하는지 확인합니다. **기존 `closed_compare.py`는 raw+aggregate 파일 존재만으로 완료 skip**하므로, 64/40문항 파일을 새 출력 루트에 넣지 않습니다.
6. 문항별 재개가 지원되지 않는 구형 runner라면 진행 중 파일에 덮어쓰지 말고 완료된 세팅 단위로만 재개합니다.

현재 실행 코드가 전달되기 전에는 이 문서에서 새 수집 명령을 만들어 실제 작업을 시작하지 않습니다. 배정안은 업무 경계이고, 미확인 현재 작업을 이동시키는 명령이 아닙니다.

## 6. 버킷 공유: 서버별 경로, 완료된 스냅샷만

공유 버킷 prefix:

```text
hf://buckets/jin-kwon/poly/PolyJigsaw/full_grid_20260917_v1/
  primary_large/snapshots/<snapshot_id>/
  l40s_small/snapshots/<snapshot_id>/
```

owner는 위 두 이름으로 고정합니다. worker가 늘면 owner와 모델 배정표를 함께 갱신합니다. 서로 다른 서버가 같은 `results/` prefix나 같은 manifest를 갱신하지 않습니다. HF sync를 원자적 작업 잠금으로 사용하지 않습니다.

스냅샷은 쓰기가 끝난 파일을 별도 staging 폴더로 복사한 불변 묶음입니다. 활성 JSONL을 직접 업로드하지 않습니다. 각 완료 세팅의 raw 파일에 정확히 315/250개의 서로 다른 예상 item_id가 있고 aggregate `n`도 일치해야 합니다. baseline은 별도 저장합니다. snapshot manifest에는 파일별 SHA-256·크기, owner, model/dataset/setting, protocol_hash, 판정 완료 여부를 기록합니다.

```bash
HF_BIN="$PWD/.venv-hf-sync/bin/hf"
# 예시: l40s_small 담당자가 생성한 완료 스냅샷. 실제 값으로 대체.
SNAPSHOT_ID='<unique-completed-snapshot-id>'
STAGING_DIR='/absolute/path/to/completed-snapshot'
DEST="hf://buckets/jin-kwon/poly/PolyJigsaw/full_grid_20260917_v1/l40s_small/snapshots/$SNAPSHOT_ID"

# 삭제 없이 업로드. 실패 시 같은 불변 스냅샷으로 재시도.
"$HF_BIN" buckets sync "$STAGING_DIR" "$DEST"
# 명령이 성공한 후에만 별도 receipt 디렉터리를 업로드하여 완료를 알림.
# receipt 내용: snapshot_id, manifest SHA-256, 업로드 완료 시각.
"$HF_BIN" buckets sync '/absolute/path/to/receipt-dir' \
  "hf://buckets/jin-kwon/poly/PolyJigsaw/full_grid_20260917_v1/l40s_small/receipts/$SNAPSHOT_ID"
```

위 명령의 staging과 receipt는 **검증한 완료 결과로 먼저 만들어야 하는 입력**입니다. 이 인계 문서가 새 그리드 결과를 업로드했다는 뜻은 아닙니다. 새 프로토콜 결과를 기존 0913 부분 수집 prefix에 덮어쓰지 않습니다.

다른 서버에서 받기:

```bash
.venv-hf-sync/bin/hf buckets sync \
  hf://buckets/jin-kwon/poly/PolyJigsaw/full_grid_20260917_v1/l40s_small \
  ./incoming/full_grid_20260917_v1/l40s_small
```

receipt가 없는 스냅샷은 미완료로 취급합니다. 다운로드 후 manifest hash·파일별 SHA-256·행 수·ID 집합을 검증합니다. `(model_revision,dataset,item_id,setting_id,protocol_hash)` 키로 병합하고, 같은 키의 다른 응답은 충돌로 남깁니다. 마지막 파일로 덮어쓰거나 서로 다른 프로토콜을 평균내지 않습니다. 재현용 원문 결과는 버킷에만, GitHub에는 코드·환경·배정·비민감 상태만 올립니다.

## 7. 완료와 소요 시간

- MJ: 모델당 50,400개, 17모델 856,800개.
- LG: 모델당 40,000개, 17모델 680,000개.
- 합계: 1,536,800개. 응답 수집과 판정 완료 수를 별도로 표시합니다.
- 34개 모델·데이터셋 단위마다 160개 세팅의 item_id 집합을 검증합니다. 파일 개수만으로 완료 선언하지 않습니다.
- ETA는 서버별 최근 완료 증가량/경과시간을 모델별로 추정합니다. 큰 모델과 작은 모델의 속도를 섞어 하나의 평균으로 전체 시간을 확정하지 않습니다.
- 저장된 응답을 반복 조회하는 것은 오프라인 replay입니다. 반복 조회를 독립적인 새 생성 시행으로 세지 않습니다.

## 8. 다른 L40S 서버 작업자에게 전달할 요약

> 이 폴더의 work_plan.json과 README_KO.md를 읽어 주세요. 기본 배정은 l40s_small의 12모델이며, 각 모델은 MJ 315문항·LG 250문항·160세팅 전체를 담당합니다. 먼저 현재 주 서버에서 실행 중인 모델 목록과 환경을 받아 중복 작업을 제거하세요. 기존 패널 기본값과 requirements-collection.lock.txt를 그대로 쓰면 모델 목록/환경이 달라집니다. active run과 동일한 입력·모델 revision·생성·판정 프로토콜을 고정한 후 진행하세요. 완료 스냅샷만 자기 owner 버킷 prefix에 업로드하고 receipt와 hash 검증으로 교환하세요. 현재 초안에는 미확인 active server 설정이 있으므로 이를 확인하기 전 대규모 수집을 시작하지 마세요.

HF sync 명령 근거: https://huggingface.co/docs/huggingface_hub/v1.6.0/guides/buckets
