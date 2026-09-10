# 에이전트식 적응 상태 탐침 (경량, judge 불필요) — 2026-09-04

사용자 제안: 상대 모델이 뭔지 몰라도, 이전 응답을 보고 발전적으로 프롬프트를 골라
상태를 파악. 무거운 모델이 꼭 필요하지 않을 수 있다.

## 아이디어

구조 정책은 대상에 대해 두 숫자만 필요하다: A=평문 순응(정렬), C=재구성 능력(난이도 임계).
고정 배터리 대신 적응적으로 추정한다.
- A: 평문 유해요청을 적응 샘플링, **거부 패턴 감지기**로 채점(안전 judge 불필요), SE 작아지면 조기 종료.
- C: 재구성 난이도를 **이분탐색(staircase)** 으로 조절, **어휘 재구성 점수(token-F1≥0.8)** 로 채점
  (재구성 judge 불필요), pass율이 0.5를 지나는 난이도를 C로.
무거운 judge(Qwen2.5-7B, Qwen3Guard-8B)는 마지막 설정 확정 때만 사용.

## replay 검증 (18개 완료 대상, harm_gen 저장값 사용)

- 정렬 A: 평균 |추정−실측| = 0.089
- 능력 C: 평균 |추정−실측| = 0.103
- 평균 총 탐침 질의 = 28.9 (고정 배터리 ~35보다 적음), 그리고 **탐침 단계에 judge 모델 0개**

## 의의

- 상대 모델 정체를 몰라도 소수 질의로 (A,C)를 정확히 복원. 이게 구조 정책 warm-start 입력.
- 탐침에 7B/8B judge 불필요 → 사용자가 말한 "무거운 모델이 필요하지 않을 수도"가 성립.
- online_live에 `--agentic` 옵션으로 연결: 적응 탐침 → (A,C) → 구조 warm-start → GP-UCB.

## 코드
- scripts/agentic_probe.py: 컨트롤러(estimate_alignment/estimate_capability), ReplayProbeEnv(검증),
  LiveProbeEnv(라이브), agentic_fingerprint.
- scripts/online_live.py: --agentic 경로.
- 한정: 거부 감지기는 패턴 기반이라 완벽하지 않음(우회 표현 존재). C는 난이도 격자 해상도에 의존.
  최종 설정 확정은 여전히 정식 judge로 채점한다.
