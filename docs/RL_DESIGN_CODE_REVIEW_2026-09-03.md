# 강화학습 선택기 설계·코드 검토 — 2026-09-03

대상: `scripts/train_selector_gpu.py`, `scripts/collect_selector_gpu.py`,
`scripts/run_selector_gpu_queue.py`, `scripts/build_selector_gpu_data.py`,
`tests/test_selector_gpu.py`. 목적: 지금 코드가 (a) 그대로 돌아가는가, (b) 공동
목적(재구성+ASR)으로 돌릴 수 있는가, (c) 설계가 타당한가.

## 결론 요약

- 현재 코드는 **무해 재구성 어휘 대리 보상**에서는 돌아가고 단위검사도 통과한다.
  그러나 그 목적에서는 PPO와 고정 설정이 같은 값(0.8767)이라 RL 이점이 없다.
- **공동 목적(gated ASR)으로는 지금 코드로 학습할 수 없다.** 세 곳에서 막힌다.
  수집기가 유해 작업을 거부하고, 트레이너가 유해 provenance를 예외 처리하며,
  보상으로 읽는 필드가 무해 대리 점수뿐이다. 대상/계열 수도 부족하다.
- 설계 자체(분할, 정보 누출 차단, 특징, 순차 정책)는 견고하다. 막는 것은 목적
  연결과 규모다.

## 1. 그대로 돌아가는가 (무해 목적)

돌아간다. `main()`은 CUDA 필수, 계열 보류 fold, seed 3개, 지도학습→PPO→continued
비교, 결과 자동 집계까지 구현돼 있다. 정보 누출·순열 등변성·PPO 갱신 검사 4개가
있다(`tests/test_selector_gpu.py`). 다만 `results/selector_gpu_20260903/REPORT.md`에서
train128_12 기준 fixed=0.8767, ppo=0.8767로 **동일**하다. 무해 대리 점수가 포화되고
단일 고정 설정이 이미 최적에 가깝기 때문이다. 즉 "돌아가지만 가설을 증명하지 못한다".

## 2. 공동 목적으로 돌릴 수 있는가 — 구체적 blocker 3개

1. **수집기가 유해를 거부.** `collect_selector_gpu.py:34`
   `if any(j["risk_type"] != "benign" ...): raise`. 출력 행도
   `reconstruction_proxy`와 `risk_type:"benign"`만 쓴다(`:108–117`). 안전 판정이 없다.
   → 공동 J를 만들려면 유해 생성 + 재구성 judge + Qwen3Guard(답변만)를 돌려
   셀별 J∈[0,1]을 산출하는 수집 경로가 필요하다. 이번 factorial 러너가 그 생성·판정을
   이미 수행하므로, 그 산출을 트레이너 입력 형식으로 바꾸면 된다.
2. **트레이너가 유해 provenance를 예외 처리.** `train_selector_gpu.py`의 `load_data`는
   `value = row["reconstruction_proxy"]`로 보상을 읽고,
   `if row["risk_type"] != "benign": raise ValueError`로 유해 행을 거부한다.
   → 보상 필드를 `joint`(=R AND U 비율 또는 문항별 0/1의 평균)로 바꾸고, provenance
   검사를 목적에 맞게 파라미터화해야 한다. 표는 [items × configs]로 값만 J이면
   나머지 학습 코드는 그대로 쓸 수 있다.
3. **규모 가드.** 기본 `--require-targets 12`, fold는 계열 수만큼이며 각 fold가
   test·validation 계열을 하나씩 보류하므로 계열이 최소 4개는 필요하다. 이번 공동
   factorial은 6개 대상, 계열 3개(Qwen, Phi, Falcon)라 계열 보류 학습이 성립하지 않는다.
   → 공동 관측을 더 많은 계열로 확장해야 전이 주장을 할 수 있다.

## 3. 설계 타당성 (공동 목적으로 옮겼을 때)

**타당한 부분.**
- 분할: 계열 보류 + 원문 split 분리. calibration(8문항)과 보상(32문항)이 episode
  내에서 분리되고, 최종 test 점수는 정책 입력·업데이트에 들어가지 않는다.
- 정보 누출 차단: `Policy.forward`가 미관측 셀을 prior로 덮어 관측만 인코더에 넣는다.
  `no_feedback` 대조군이 순차 이력의 기여를 분리한다.
- 특징: 각 설정 벡터가 언어 one-hot + n/10 + **fragments/12** + 배열 one-hot을
  담는다. 즉 조각 수 축을 정책이 활용할 수 있다. factorial 결과상 조각 수가 가장 큰
  단일 축이므로 이 특징이 실제로 의미가 있다.
- 보상: PPO 종료 보상이 추천 설정의 보상-문항 평균이다. 값이 J이면 이는 정확히
  reconstruction-gated ASR을 최대화한다.

**보완할 부분.**
- **앵커 위치 하드코딩.** `initial_state`가 설정 인덱스 `[0, C-1]`을 항상 관측된
  앵커로 둔다. 공동 설정 집합에서는 의미 있는 값싼 앵커(예: english_direct)를 그
  위치에 배치하거나 앵커 선택을 일반화해야 한다. 아니면 "무료 2개"가 임의 설정이 된다.
- **scalar J의 표현력.** J 하나로 R과 U를 합치므로 "ASR 최대, 단 재구성 τ 이상"
  같은 제약형 목적은 표현하지 못한다. 사용자의 우선순위(감지 회피하며 ASR)는 J가
  obfuscated-yet-reconstructable를 보상하므로 J로도 담기지만, 제약형을 원하면 보상
  설계를 바꿔야 한다.
- **`random` 명명.** `rollout`의 random은 무작위 질의 후 `(4·prior+8·관측)/(4+8)`
  argmax로 고른다. 순수 무작위가 아니라 prior 수축 결합이다. 보고에 명시하거나 이름을 바꾼다.
- **replay 현실성.** 정책이 받는 결과는 미리 수집한 표에서 재생한 값이다. 기록에 없는
  조합은 알 수 없고, 같은 표에서 만든 수천 episode가 독립 대상 수를 늘리지 않는다.
  불확실성을 episode 수로 축소하면 안 된다(문서에 이미 명시됨, 유지 필요).
- **규모·검정력.** 공동 목적에서는 모델별 최적이 실제로 갈린다(이번 factorial에서
  4/5 모델은 조각 3개가 최적, qwen25_14b만 조각 8개; panel LOO regret 0.129). headroom은
  있으나 6개/3계열로는 전이를 입증하지 못한다. 계열 10개 이상 목표가 필요하다.

## 권장 다음 단계

1. 공동 factorial 산출(`private_artifacts/frag_factorial_20260903`)을 트레이너 입력
   형식(J 값 테이블 + 설정 특징 + 원문 split)으로 변환하는 어댑터를 추가한다.
   기존 `load_data`를 복제·수정하되, 유해 provenance 검사와 응답 해시 검증은 유지한다.
2. `Policy`/`rollout`/`ppo_updates`는 값 텐서만 J로 바꾸면 재사용 가능하다. 앵커 인덱스와
   `random` 명명만 손본다.
3. 계열 수가 4 미만이면 계열 보류 대신 대상 보류 LOO로 시작하고, 공동 관측을 더 많은
   계열로 확장한 뒤 계열 보류로 승격한다.
4. 무해 재구성률과 공동 목적을 섞어 보고하지 않는다. 최종 평가는 보류 계열·보류 문항의
   gated ASR·raw ASR·재구성률·조건부 ASR과 불확실성을 함께 낸다.
