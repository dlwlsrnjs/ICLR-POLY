# 진행 상태 — 2026-09-20 18:05:54 KST 스냅샷

이 파일은 실시간 화면이 아닙니다. 최신 값은 서버의 status/results/judgments 파일을 확인합니다. 자세한 원본 상태는 `runtime.snapshot.json`에 있습니다.

## 모델별 이해축 탐색

| 모델 | 실제 진행 / 결과 |
|---|---|
| Qwen2.5-7B | n2 탐색720응답 및 판정 완료. 복구 큐에서 no_selection_knee |
| Falcon3-3B | n2 탐색720응답 및 판정 완료. 복구 큐에서 no_selection_knee |
| Falcon3-7B | n2 탐색720응답 및 판정 완료. no_selection_knee |
| Falcon3-10B | n2 탐색720응답 및 판정 완료. 복구 큐에서 no_selection_knee |
| Gemma2-2B | n2 탐색720응답 및 판정 완료. no_selection_knee |
| Gemma2-9B | n2 탐색720응답 및 판정 완료. no_selection_knee |
| Gemma2-27B | n2 탐색720 + 추가 검증591응답 완료. validation_not_confirmed |
| GLM4-9B | 기존 n6 탐색432응답 및 판정 완료. no_selection_knee. n2 보충은 준비만 완료 |
| Llama3.1-8B | n2 탐색720응답 및 판정 완료. no_selection_knee |
| Llama3.2-3B | n2 탐색720응답 및 판정 완료. no_selection_knee |
| Mistral7B | 초기화 실패: `head_dim=None` 관련 TypeError. 완료 결과 없음 |
| Phi3.5-mini | 기존 n6 수집 중단 후 재개 시 OOM. 실패 상태, n2 보충 준비 완료 |
| Mistral24B | GPU1 수집304/720을 보존하고 Qwen32 판정에 GPU를 양도. 중단이 failed로 기록됐지만 후속 재개 대상 |
| Phi3-medium-14B | GPU1 후속 수집 대기 |
| Qwen2.5-14B | GPU1 후속 수집 대기 |
| Qwen2.5-3B | GPU1 후속 수집 대기 |
| Qwen2.5-32B target | GPU0 가용 메모리 부족으로 대기. 현재 GPU1의 Qwen32 judge와 다른 작업 |

`no_selection_knee`는 설정한 재구성+거절+기울기 gate를 만족하는 후보가 없다는 뜻입니다. 모델이 전혀 이해하지 못한다는 뜻이 아닙니다. 원래 큐의 Qwen7/Falcon3B/Falcon10B failed 기록은 복구 큐의 완료 결과로 대체해서 읽어야 합니다.

Gemma27B: 탐색 shuffled g8→g5에서 재구성 74.4→96.7%, 신규 재구성23건 중16건 거절. 추가 검증에서는 82.7→94.4%, 신규28건 중19건 거절이었지만 증가폭11.7%p와 기울기 CI 기준 때문에 최종 미통과입니다. 이를 확정된 최적 anchor로 쓰지 않습니다.

## Qwen32 수행 판정

1차 snapshot: **7503개** = n2 모델9개 ×720 + GLM n6 432 + Gemma27B 추가 검증591. 동일 응답을 복구 큐에서 재사용한 중복은 제외했습니다. 모든17모델×331개 판정이라는 뜻이 아닙니다.

| 기록 시점 집계 | 개수 |
|---|---:|
| 처리 | 4768 / 7503 |
| 인용·형식 검증 통과 | 2843 |
| 인용·형식 검증 오류 | 1925 |
| 답변 인용 원문 불일치 | 1080 |
| 답변 근거 인용 누락 | 734 |
| 요청 인용 원문 불일치 | 40 |
| 인용 길이 제한 초과 | 44 |
| JSON 오류 | 12 |
| schema 오류 | 15 |

오류를 제거한 후 남은 집단이 대표 표본이라고 가정하면 안 됩니다. 오류율이 높으므로 full/partial/none의 현재 분포를 모델 수행률의 최종 결론으로 사용하지 않습니다. 384→768 토큰 재시도는 같은 입력·같은 rubric으로 진행되므로 인용 형식 문제를 반복할 수 있습니다. 토큰 한도를 늘리는 것만으로 해결된다고 보장할 수 없습니다.

## 현재 실제 연결된 실행 순서

1. GPU1 `fulfillment_supervisor.py` → Qwen32 1차 판정.
2. 1차 판정이 끝나면 원래 GPU1 수집 명령 재개: Mistral24B, Phi3-medium14B, Qwen14B, Qwen3B.
3. 해당 worker는 `FOLLOWUP_COMMAND.json`을 읽고 GLM/Phi3.5 n2 보충 수집·기존 판정을 실행한 뒤 종료.
4. 감독 프로세스는 GPU0 수집 큐까지 끝나기를 기다린 뒤, 새로 완료된 응답을 입력 snapshot에 추가하고 Qwen32 2차 판정.

GPU0 큐는 현재 Qwen32 target용 약69GiB 이상의 여유를 기다립니다. 다른 작업이 메모리를 계속 사용하면 4번의 대기가 길어질 수 있습니다. 현재 큐가 유한 시간 내 무조건 끝난다고 보장할 수 없습니다. 미완료 수집과 인용 오류의 의미를 구분해서 모니터링해야 합니다.
