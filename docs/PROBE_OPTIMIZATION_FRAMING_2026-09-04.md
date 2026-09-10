# 프레이밍: 적응 탐침으로 잠재 능력 곡선 추출 + 정보·목적 결합 최적화 — 2026-09-04

우리 방법을 확립된 두 문헌 계열로 규정한다(신규성은 방법이 아니라 도메인 + 공동 목적).

## A. 대상의 잠재 능력·정렬을 최소 질의로 추출 (CAT/IRT · 적응 심리물리)

- 능력 C = "재구성 pass율이 0.5를 지나는 난이도(역치)". 이는 적응 심리물리의
  **level-set threshold estimation**과 동일. 우리 staircase(이분탐색)가 그 추정기다.
- 정렬 A = 평문 순응률. 적응 샘플링 + SE 조기 종료.
- 채점은 경량(어휘 재구성 점수, 거부 패턴) — 무거운 judge 불필요.
- 관련: CAT/IRT(문항 1.3%로 전수 근사, npj Digital Medicine 2026), IRT for AI Safety
  (2608.05086), continuous-score adaptive eval(2601.13885), adaptive nonparametric
  psychophysics / Bayesian psychometric field (PMC5839980), Fisher-info 추적 최적성(2510.07862).

### 검증에서 배운 것 (정직)

28개 대상 replay:
- staircase(level-set): C 오차 0.078, C-질의 16.7, 총 ~28.7질의. 정렬 A 오차 0.071.
- 파라메트릭 IRT/CAT(고정 기울기 k=4, MFI 문항선택): C 오차 0.212, C-질의 24. **더 나쁨.**
- 원인: 고정 기울기 가정의 모델 오설정 + 이산 난이도 격자(16 rung). 나온 결론은
  "우리 데이터에선 비모수 level-set(staircase)가 파라메트릭 IRT보다 견고하다"이다.
  IRT를 쓰려면 대상별 기울기 추정과 난이도 스케일 캘리브레이션이 필요하다.
- 그래서 파이프라인은 staircase를 유지한다(estimate_capability). IRT 버전
  (estimate_capability_irt)은 비교용으로만 남긴다.

## B. 지문 파악(정보 이득) + 공동 ASR(목적)을 함께 최적화

- 탐침 단계는 정보 이득, 최종 단계는 공동 ASR을 최적화한다. 이 둘을 한 목적으로 저울질하는
  정식 틀이 **Active Inference의 Expected Free Energy**(정보이득 + 실용가치)다(2603.29003).
- 우리 하이브리드: 적응 탐침으로 (A,C,M) 추출 → 구조/확장 정책 warm-start(질의0)
  → 온라인 GP-UCB(공동 ASR 미세조정) → 최종 judge 확정.

## 논문에서의 규정

"CAT/level-set 적응 탐침으로 대상의 잠재 재구성 능력·정렬을 최소 질의로 추출하고,
정보 이득과 reconstruction-gated ASR 목적을 결합해 대상별 설정을 온라인으로 구축한다."
방법 구성요소는 확립된 것이며, 기여는 (1) 탈옥 설정 선택 도메인, (2) 재구성+ASR 공동 목적,
(3) 무거운 judge 없이 반복 질문만으로 능력 파악, (4) 지문 warm-start + 온라인 적응 하이브리드.
