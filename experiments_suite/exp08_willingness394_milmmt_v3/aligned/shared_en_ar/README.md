# 영어·아랍어 고정 이해축 / 331개 과잉거부 prior

처음 언급된 311개는 사용자 확인을 거쳐 **331개**로 확정했습니다.
이 폴더에는 실행 코드뿐 아니라 실제 영어 원문, 기존 번역·역번역과 선별 근거가 포함됩니다.

- FalseReject 212개 + OR-Bench Hard 119개. 9개 선별 모델 중 5개 이상이 거절한 문항입니다.
- 영어 + 아랍어, `g3_ordered_n2`: 언어별 3조각, 총 6조각, 언어 내부 순서 유지.
- 의지 프레임: `plain`, `persona`, `fiction`, `pap`, `persona+fiction`.
- 영어 원문을 고정하고 5개 프레임의 조각 내용도 동일하게 유지합니다.
- 대상 모델 응답 한도는 **max_tokens=1024**입니다. `[RECONSTRUCTED]`와 `[ANSWER]`를 합한 생성 토큰 한도이며, 입력을 포함하는 문맥 길이는 max_model_len=4096입니다.
- 이전 320토큰 실행 폴더는 그대로 보존하고 새 OUT에서 실행합니다. 번역 QA의 판정 출력 한도(160토큰)는 별도입니다.
- 17모델의 모델명·revision·샘플링 설정은 `configs/`에 고정했습니다.

**이 설정은 공통 언어 비교 실험입니다.** 기존 MJ 기본 n2는 영어+벵골어,
LG 기본 n2는 영어+노르웨이어입니다. 영어+아랍어 결과를 기존 기본 n2 결과와
섞거나, 기존 benign 재조합 정확도가 여기서도 검증된 것처럼 해석하지 않습니다.
두 데이터셋에 사용할 공통 prior는 한 번 수집하면 됩니다.

## 포함된 데이터

| 파일 | 내용 |
|---|---|
| `data/items.json` | 실제 331개 영어 원문, source ID, 범주, 거절 모델 수·비율, 선별 순서 |
| `data/original_items.json` | 이전 331개 은행의 원본 파일 |
| `data/original_items_manifest.json` | 원본 출처 경로와 체크섬 |
| `data/translations.jsonl` | 노르웨이어·핀란드어·아랍어 각각 331개: 번역·역번역, 전체 입력 텍스트, 종료 상태, 토큰 수, MiLMMT revision |
| `data/translation_metadata.json` | 기존 번역 실행 메타데이터 |
| `DATA_MANIFEST.json` | 공개 데이터 파일 크기·SHA256 |
| `smoke_inputs/` | 이전 Qwen32 QA를 통과한 아랍어 6문항 및 QA 입력·출력 근거 |

993개 번역 모두 동일한 영어 원문에 연결됨을 확인했습니다. 번역이 존재한다는
것과 의미 QA를 통과했다는 것은 별개입니다. QA 실패 문항은 모든 프레임에서
함께 보류하고 원문/번역/판정 기록을 남깁니다. 무응답·잘림·QA 실패를 거절이나
성공으로 치환하지 않습니다. 데이터의 범주명은 출처를 그대로 보존했습니다.
원 데이터의 이용 조건은 FalseReject 및 OR-Bench의 원 배포 조건을 따릅니다.

## 환경

모델 실행에 NVIDIA GPU와 모델 가중치가 필요합니다. 가중치는 이 저장소에
포함하지 않습니다. 로컬에서 사용한 환경은 Python 3.10, vLLM 0.8.5,
PyTorch 2.6.0, Transformers 4.57.6, tokenizers 0.22.2입니다.
저장소 루트의 `requirements-collection.lock.txt`와 환경 검사기도 이 실제 환경으로
정정했습니다. 근거는 [수집 환경 문서](../../../../docs/COLLECTION_ENVIRONMENT.md)에 있습니다. 모든 17모델의 실행 호환성이
검증된 것은 아닙니다. 특히 대형 모델은 아래 7B용 메모리 설정으로 실행할 수 없습니다.
CPU 데이터 검증·퍼즐 생성은 표준 Python으로 가능합니다.

저장소 최상위에서 시작합니다:

```bash
P=experiments_suite/exp08_willingness394_milmmt_v3/aligned/shared_en_ar
bash "$P/run.sh" verify

export PYTHON=/absolute/path/to/vllm-environment/bin/python
export OUT=/absolute/path/to/new_run
export MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
export QA_MODEL_PATH=/absolute/path/to/Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
```

### 1. 먼저 실제 30응답 테스트

```bash
GPU=0 bash "$P/run.sh" smoke
```

6문항 × 5프레임입니다. 기존 QA와 겹치는 소표본이므로 전체 은행의 성능이나
17모델 공분산으로 해석하면 안 됩니다.

### 2. 331개 아랍어 의미 QA

```bash
GPU=1 bash "$P/run.sh" qa
```

Qwen32 BF16을 단일 전용 L40S에서 실행할 때 검증한 기본값은 CPU offload
25 GiB, GPU memory utilization 0.95, max_num_seqs 16,
max_num_batched_tokens 2048, batch size 32입니다. 이 설정으로 331개 판정을
완료했으며 284개가 통과하고 47개가 보류됐습니다. 공유 GPU에서는 여유 VRAM에
맞춰 값을 낮추거나, 가능하면 두 개의 빈 GPU에서 tensor parallel을 사용하세요.
실행 서버의 RAM·VRAM 여유를 먼저 확인하세요. 공유 GPU에서는 다른 프로세스의
사용량에 따라 이 설정도 실패할 수 있습니다. 영어·아랍어·역번역의 의미와
언어를 판정하며, `OUT/qa/judgments.jsonl`에 전체 입출력·종료 사유를 보존합니다.
중단 후 동일 명령으로 이미 기록된 판정을 이어서 처리합니다.
설정을 바꿨다면 새 OUT을 사용해 실행 기록을 구분하세요.

### 3. QA 통과 문항으로 17모델 입력 생성

```bash
bash "$P/run.sh" prepare
```

각 모델의 `manifest.json`, `jobs.jsonl`, `blocked.json`이 생성됩니다.
최대 331 × 5 = 1,655응답/모델이며, 보류 문항은 별도로 기록합니다.
각 job에는 영어 원문·실제 프롬프트·메시지·언어별 번역·조각·정답 순서·해시가 있습니다.

### 4. Qwen7B 본 수집

```bash
GPU=0 MODEL_TAG=qwen25_7b MEMORY_UTILIZATION=.85 BATCH_SIZE=8 \
  bash "$P/run.sh" collect
```

`MODEL_TAG`와 `MODEL_PATH`를 해당 config의 모델·revision으로 함께 바꾸면
다른 모델의 수집 입력을 사용할 수 있습니다. 17모델 동시 실행 명령이 아닙니다.
각 모델의 메모리와 vLLM 호환성을 확인한 뒤 순차 실행해야 합니다.
수집 결과는 `OUT/panel/<model_tag>/responses.jsonl`에 전체 입력·출력·종료
사유·토큰 수와 함께 저장됩니다. 같은 입력·런타임이면 중단 지점부터 재개합니다.
입력이나 런타임이 달라지면 새 OUT을 사용합니다.

여러 고정 snapshot을 중단 재개 가능한 순차 큐로 돌릴 때는
`code/run_panel_queue.py`에 `--model TAG=/snapshot/path`를 반복해 전달합니다.
`--after-pid`로 선행 큐 종료를 기다릴 수 있고, 각 큐는 별도
`--status-file`과 모델별 로그를 남깁니다. 제한된 임시 저장소에서 마지막 대형
모델을 하나씩 staging하려면 `code/run_remaining_models.py`, 전체 수집 후 GPU
두 장에서 reconstruction/WildGuard를 병렬 판정하고 prior를 집계하려면
`code/run_judge_queue.py`를 사용합니다. 두 보조 실행기는 완성 행 수를 검사하며,
불완전한 모델의 가중치나 결과를 자동 삭제하지 않습니다.

## 판정 및 의지축 해석

수집 후 상위 `aligned/judge.py`로 reconstruction 및 WildGuard 판정을 실행하고,
`aligned/summarize.py`로 재조합 성공·정상 종료·유효한 답변이 있는 공통 문항을
기준으로 5프레임 non-refusal을 산출합니다. 자세한 인자는 상위 README와
각 스크립트의 `--help`를 참고하세요. 기존 judge 실행기는 GPU 1을 사용하므로
번역 QA와 동시에 같은 GPU에서 실행하지 마세요.

Non-refusal은 답변의 정확성/충실도와 동일하지 않습니다. 선정된 과잉거부
은행이므로 일반 무해 질문 전체로 일반화할 수 없습니다. 공분산은 실제로
판정된 모델×5프레임 결과가 모인 후 계산해야 하며, 아직 보고하지 않습니다.

## 검증 및 실제 실행 상태

`STRUCTURAL_AUDIT.json`: 실제 영어·아랍어 퍼즐 1,655개에서 원문 복원과
프레임 간 조각 동일성 확인. 이것은 생성 코드 검증이며 모델 재조합 성공률은 아닙니다.
`RUN_STATUS.json`은 기록 시각의 스냅샷입니다. 초기 0.30 설정은 KV 캐시를
확보하지 못해 실패했습니다. 이후 전용 L40S에서 30응답 smoke를 완료했고
(30/30 정상 종료, 필수 두 section 존재), 위의 조정된 설정으로 Qwen32 QA
331건도 완료했습니다. 대상 패널 수집은 중단 재개 가능한 `collect.py`와
`code/run_panel_queue.py`로 순차 실행합니다. 공유 GPU에서는 높은 utilization을
그대로 사용하지 말고 실제 여유 VRAM을 먼저 확인하세요. 실제 진행 여부는 실행
폴더의 `pipeline_status*.json`, 모델별 로그, 응답 행 수로 확인합니다.
