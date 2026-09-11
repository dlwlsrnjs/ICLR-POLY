# 무해 prior 강화 설계 (2026-09-10)

## 진단 (qwen7b MJ, 292 공간)
- corr(셀 재구성, verified) = **0.525**. 이해 신호는 유효.
- corr(현재 prior, verified) = **0.386**. 현재 prior가 recon보다 나쁨.
- 원인: 현재 prior = recon × willingness, 그런데 **benign willingness 신호가 포화**(nonrefusal 분산=0.0,
  persona/fiction/pap 전부 1.0). helpful 모델은 무해 프레임을 다 받아줘 어떤 스택이 **유해** 정렬을 뚫는지
  구분 못 함. plain의 willingness가 1.0으로 잡혀 plain을 과대평가 → 오히려 해가 됨.
- 그러나 verified는 willingness에 크게 의존: persona+fiction 0.466 ≫ plain 0.190.

## 강화안

### 1) 포화 인지 prior (무료, 구현·검증됨)
축별 benign 신호의 프레임 간 분산이 ~0이면 그 인자를 **버리고 미지로 처리**(균등 평균 + 높은 불확실성)
→ 탐색이 그 축을 캐게 함. 효과: prior_info 0.386→**0.517**, πBO AUC 0.64→0.666. 순수 이득.
구현: 분석 단계에서 `benign_recon_by_cell` + `frame_signals` 분산으로 prior 재구성(재수집 불필요).

### 2) 비포화 willingness probe (실질 해법, 무해 추가 실행 필요)
이진 nonrefusal은 helpful 모델에서 포화. **컴플라이언스 깊이의 연속 신호**로 대체:
- 경계 근접(합법·교육적 dual-use-adjacent) 무해 요청에서 프레임별
  (a) hedge/면책 밀도(정규식), (b) 지시 완결성(구체 단계 vs 모호), (c) 거부토큰 logprob(가능 시).
- 난이도 램프(점점 민감하지만 합법)로, **어느 프레임이 hedge 없이 가장 깊게 응하는지** 측정.
- 이건 helpful 모델도 포화 안 되는 gradient → 유해 willingness를 예측. 무해 콘텐츠만 사용.

### 3) πBO + 축별 적응 가중 (무료)
πBO 감쇠 prior에 (1)의 축별 신뢰를 결합: 포화 축은 prior 미반영, 유효 축만 πBO로 가속.
prior가 좋으면 2쿼리 근오라클, 포화면 구조 탐색으로 폴백(random에 안 짐).

### 4) (참고, 비-무료) 교차모델 구조 prior
스택>단일>plain 규칙을 패널 LOTO로 학습해 약한 구조 bias. 타 모델 유해데이터 사용이라 "무료" 주장엔
못 넣고, 상한 참조로만.

## 권고
(1)+(3)을 기본 selector에 반영(무료, 즉시), (2)를 무해 추가 probe로 수집해 helpful 모델의 willingness
포화를 근본 해결. 검증은 패널이 차는 대로 prior_info 축으로 다모델 대조.

## 온라인 축 버리기 = 수집 중 적용 + 애블레이션 (2026-09-10)
- 축 버리기 **결정은 무해 probe 시점**(유해 쿼리 전)에 프레임/셀 분산으로 내려진다 → 후처리가 아니라
  수집 중(온라인) 적용. 유해 예산을 정보 있는 축에 집중.
- 애블레이션(`axis_drop_ablation.py`): 셀마다 drop vs no-drop selector의 예산곡선/AUC와 버린 축을 기록.
- qwen7b MJ 결과: 버린 축=willingness, ucb_add_pibo AUC 0.640→0.666(**+0.026**). 패널이 차면
  "버린 축 ↔ 효과"를 다셀 집계해 "수집 중 포화축 제거가 효과적"을 근거로 제시.
- 산출물: results/axis_drop_ablation*.json, strategy_bakeoff.json (셀별 comp/will_informative 기록).

## 온라인 적응 subspace BO (SAAS-lite, 2026-09-10) — 채택 방법
무해 1회-drop을 넘어, 유해 관측에서 축 관련도를 L1-희소 ARD로 매 스텝 학습해 무의미 축을 온라인
프루닝(ARD/SAASBO 계열, Eriksson&Jankowiak 2021 정신). 구현 `adaptive_subspace.py`.
- qwen7b MJ: random/무해-drop 대비 우위(k8 0.655 vs random 0.523, 무해-drop 0.581; 오라클 0.719의 91%).
- **활성 축을 유해 관측에서 발견**: 최종 active=['fiction'] — 무해 probe가 못 본 willingness 핵심을 스스로 찾음.
- 배포형: 매 스텝 active subspace를 로그. 무해 warm-start(πBO)는 초기 안내, 관련도는 온라인 학습.
