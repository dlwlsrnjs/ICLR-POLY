# 다음 담당자가 어디서 이어서 해야 하는가

## 작업을 나누는 방법

| 담당 범위 | 해야 할 일 | 완료 기준 |
|---|---|---|
| A. 수행 판정 품질 | 현재32B pass를 보존·관찰. 인용 오류 표본을 분류하고 새 버전 rubric/parser를 소규모로 검증한 후 오류 응답 재판정 | 오류를 숨기지 않은 커버리지와 근거가 남고, 대표 표본의 요청 수행 여부를 사람이/별도 검토로 점검 |
| B. 동일 조건 데이터 | 공통 n2 90문항×8조건 계약 확인. 미완료 target 및 GLM/Phi 보충 수집 완료 | 17모델 모두 같은720 job 키 구성에 대응하는 응답 보유; 무효·절단도 별도 보존 |
| C. 이해축 선택 | 재구성 R, 거절 F, 실제 수행 Y를 별도로 분석. 후보를 선정 split에서 고르고 확인 split에서 검사 | 최적 조건이라고 부를 근거 또는 기준상 후보 없음이 명확함. 사후 gate 완화 금지 |
| D. 의지축 prior | 합의된 anchor만 고정하고 plain/persona/fiction/pap/persona+fiction 수집 | 5프레임 공통 문항과 품질 커버리지 확인. 비거절과 실제 수행 지표를 구분 |
| E. 패턴·공분산 | 동일 문항 벡터를 프레임 간 정렬하고 공분산/상관/불확실성을 계산 | 결측 처리와 분모가 명시되고 재현 가능. 현재는 이 단계 미도달 |

모든 작업을 동시에 GPU에 올릴 필요는 없습니다. 한 GPU의 새 judge/target 실행은 기존 프로세스와 supervisor의 자동 재개까지 확인하고 배정합니다. 다른 사용자의 GPU 작업에는 종료 신호를 보내지 않습니다.

## 서버 경로

공통 root: `/home/ubuntu/342/jinkwon/poly/target_prior_only_20260916/runs/`

| 하위 경로 | 용도 |
|---|---|
| `willingness331_model_specific_overlap8_v2_20260920` | 최종 8언어 번역/QA 및 이전 calibration |
| `willingness331_transition_refusal_20260920` | 초기 2조건 전이 비교, 원본 입력, 고정 분할/기준 |
| `willingness331_adaptive_queue_20260920` | GPU0 8조건 탐색 및 결과 |
| `willingness331_adaptive_recovery_gpu1_20260920` | Qwen7/Falcon10/Falcon3 판정 복구 결과 |
| `willingness331_adaptive_parallel_gpu1_20260920` | GPU1 수집 큐, FOLLOWUP_COMMAND.json |
| `willingness331_matched_n2_source_20260920` | 공통 n2 보충용 source plan과 입력 |
| `willingness331_matched_n2_backfill_20260920` | GLM/Phi n2 입력720개씩; 미완료 |
| `matched_n2_comparison_20260920` | CONTRACT.json 및 primary_report.json |
| `fulfillment331_audit_20260920` | 현재32B 수행 판정 전체 기록 |
| `model_settings_review_20260920` | 이전 조건별 기술통계. n6/선정 조건 차이를 주의하고 참고용으로만 사용 |

## 실제 입력/출력과 판정 근거

각 수집 run의 `models/<tag>/selection/` 또는 `validation/`:

- `manifest.json`, `jobs.jsonl`: 설정, 원래 요청, 퍼즐 prompt/messages, fragment/gold-order 등.
- `responses.jsonl`: target 실제 전체 출력, 출력 토큰, finish_reason, 모델 revision, 응답 해시.
- `collection_metadata.json`, `reuse.json`: 환경/template/config 해시와 캐시 재사용 출처.
- `judge/reconstruction/restricted_reconstruction_audit.jsonl`: 재구성 의미 동등성 판정.
- `judge/outputs/wildguard.jsonl`: 거절 판정과 raw 근거/확률 관련 필드.
- `curve_analysis.json`, 모델 상위의 `confirmation.json`: 전이 후보·bootstrap·검증 결과.

현재 수행 audit:

- `input_manifest.json` → immutable `inputs_<hash>.jsonl` 경로. 원문, full response, ANSWER, target prompt/messages, input/response hash 보존.
- `source_links.json`: 복구 중복을 포함한 원래 run/key와 R/F 판정 연결.
- `judge_protocol.json`, `SOURCE_MANIFEST.json`, `allocation.json`: rubric/code/환경 핀과 GPU 할당.
- `judgments.jsonl`: full/partial/none/uncertain, exact quotes, valid/error, raw 출력과 모든 재시도. 배치마다 flush/fsync.
- `status.json`: 판정 처리량; `supervisor_status.json`: 전체 단계.
- `joined_audit.jsonl`, `summary.json`: judge pass 완료 시 생성/갱신. 진행 중에는 아직 없거나 이전 pass일 수 있음.
- `assistant_spotcheck.json`: 실제 답 대신 계획/문제 분해만 제시한 사례2개를 확인한 기록. 대표 표본도 사람 gold도 아님.

원본 영어와 실제 ANSWER를 보고 판단합니다. `[RECONSTRUCTED]`의 문장 자체를 실제 답변으로 채점하지 않습니다. unknown/invalid를 거절·미수행·재구성 실패로 임의 치환하지 않습니다.

## 최소한의 읽기 전용 확인

```bash
cd /home/ubuntu/342/jinkwon/poly
cat target_prior_only_20260916/runs/fulfillment331_audit_20260920/status.json
cat target_prior_only_20260916/runs/fulfillment331_audit_20260920/supervisor_status.json
tail -n 10 target_prior_only_20260916/runs/fulfillment331_audit_20260920/judge_round1.log
nvidia-smi
```

각 런의 `process.json`에 PID와 argv 배열이 저장됩니다. 재시작 전에 PID가 실제 해당 스크립트인지, child가 살아 있는지, STOP/PAUSE/FOLLOWUP 및 다른 supervisor가 있는지 확인합니다. **현재 실행 중인 작업에 저장된 command를 다시 실행하면 중복 GPU 작업이 생길 수 있습니다.** 채팅 종료와 무관하게 detached 프로세스는 계속되지만 서버 재부팅 후 자동 부팅 서비스는 설치하지 않았습니다.

`fulfillment_supervisor.py`는 1차 judge subprocess 실패 시에도 원래 수집을 재개하는 finally가 있습니다. judge만 종료하면 수집이 다시 올라올 수 있으므로 사용자가 GPU를 비우라고 할 때에는 supervisor와 자기 child를 함께 제어해야 합니다. 최근 일시 중지 후 재개 이력은 `USER_RESUME.json`과 보관된 user_stop 디렉터리에 있습니다.

## 알려진 문제와 다음 판단

1. **인용 오류 약40%:** 결과 저장은 정상이어도 판정 품질은 미완료입니다. exact quote 누락/markdown 차이/잘못된 인용, schema 오류를 분리하십시오. 단순히 체크를 없애거나 기존 valid를 뒤집지 말고, 새로운 parser/rubric 버전과 원문 근거 검증을 갖춘 재판정 결과를 따로 남겨야 합니다. 현재 자동 2차 pass는 새 응답 처리용이며 기존 invalid를 자동으로 전부 고쳐주는 큐가 아닙니다.
2. **Mistral7B:** head_dim None 초기화 오류. 수정 후 작은 pilot와 환경/소스 변경 해시 기록 필요. 아직 수정·재검증 완료가 아닙니다.
3. **Phi3.5:** 80개 저장 후 중단/재개, 이후 GPU 공유 중 KV cache OOM. 보존된 응답과 실패 로그를 확인한 뒤 재개해야 합니다.
4. **Mistral tokenizer 경고:** regex 관련 경고가 있어 fidelity 검토가 필요합니다. 현재 코드가 자동으로 해결했다고 주장하지 않습니다.
5. **GPU0 Qwen32 target:** 메모리 대기 중. 이 작업이 끝나지 않으면 supervisor의 최종 수행 판정 refresh도 대기할 수 있습니다.
6. **모델별 최적값:** 90문항에서 성공 수1~2개 차이는 확정적인 우열 근거가 아닙니다. 이전 추천표를 validated optimum으로 사용하면 안 됩니다.
7. **범위 조정 이력:** 사용자가 현재32B의 역할을 다시 확인한 뒤 “보충 수집 확대가 성급했다”는 설명이 있었으나, FOLLOWUP은 삭제되지 않았습니다. 현재판정만 원하는지 후속수집까지 원하는지는 실제 다음 지시를 반영해 큐 범위를 명시해야 합니다. 이 문서화 작업에서는 큐를 변경하지 않았습니다.

## 코드 위치와 확인 범위

[aligned/model_specific](../experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific/)의 `adaptive_queue.py`, `adaptive_queue_worker.py`, `transition331.py`, `queue_judge.py`, `reconstruction_judge.py`, `fulfillment_audit.py`, `fulfillment_supervisor.py`, `matched_n2_report.py`가 핵심입니다. target 렌더/수집은 상위 `prepare.py`, `collect.py`, `run_io.py`, `grid_contract.py`를 사용합니다.

기존 단위검증은 전이 bootstrap/gate, GPU 예산, 판정 CLI, quote 검사, blinded input, uncertainty 분모, matched cohort 검사에 대해 실행됐습니다. 이것이 실제 17모델 end-to-end 완료나 판정 정확도 검증을 의미하지 않습니다. 이번 요청에서는 문서·설정 스냅샷만 만들고 새 수집·재판정·환경 변경은 하지 않았습니다.
