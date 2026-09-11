# Arm-space ablation — 우리 세팅(공간 C)의 근거 (2026-09-11)

## 질문
탈옥 설정 공간을 어디까지 키울까? 네 후보를 held-in·held-out 패널에서 비교해 결정한다.
모든 공간은 수집한 292 매트릭스의 부분집합이라 추가 수집 없이 부분집합으로 평가한다.

| 공간 | 구성 | arm 수 |
|---|---|---|
| A | 이해(조각 F) × 언어(n) × 배열 (의지 없음, plain) | 32 |
| B | A + 의지 약하게: 셀당 {plain, persona, fiction} 단일 프레임 | 96 |
| **C (채택)** | B + pap + persona×fiction 2-스택: {plain,persona,fiction,pap,persona+fiction} | 160 |
| D | A × 의지 전체 멱집합(스택 전부) | 288 |

지표(셀=모델×데이터셋): 오라클(전-arm 최대 verified), benign-only(무해 prior argmax의 verified,
표적 유해 0인 스텔스 모드), adaptive@8(온라인 subspace GP-UCB 유해 8쿼리 달성치).

## held-in 근거 (qwen2.5-7B, 두 데이터셋 평균) — 현재 확정분
| 공간 | arm | 오라클 | benign-only | adaptive@8 |
|---|---|---|---|---|
| A | 32 | 0.489 | 0.357 | 0.455 |
| B | 96 | 0.684 | 0.613 | 0.650 |
| **C** | 160 | 0.772 | 0.613 | **0.675** |
| D | 288 | 0.784 | 0.588 | 0.675 |

읽는 법:
- 오라클(천장)은 의지 키울수록 상승하나 **C→D는 정체**(0.772→0.784).
- adaptive@8(실제 달성치)은 **C에서 포화**(C=D=0.675) — 중간 이상 키워도 예산 8에선 이득 없음.
- benign-only는 **D(전체 스택)에서 하락**(0.588) — 무해가 못 고르는 arm이 늘어 희석.
→ **C가 스위트 스팟**: 천장 대부분 확보 + 탐색 최고치 + 무해 스텔스 유지, 비용은 D의 절반.

## 결정
**채택 = 공간 C (이해×언어×배열 × 의지 중간).** 근거: 천장/탐색 이득은 C에서 사실상 포화하고,
D의 전체 스택은 추가 비용과 benign-only 희석만 낳는다. A(의지 없음)는 재구성은 되지만 거부하는
모델에서 천장이 낮아 부족하다.

## held-out 확인 (수집 중, 자동 갱신)
held-out 모델(mistral7b, phi35, falcon3)을 292로 수집해 같은 A/B/C/D를 비교한다. 결과는
`results/space_ab_compare.json`(모델·데이터셋·held-in/out별 오라클/benign-only/adaptive)에 저장되고,
`/tmp/space_ab_result.log`에 요약된다. held-out에서도 C가 A·B를 이기고 D와 동급이면 C로 최종 확정.

## 산출물
- 코드: `experiments_suite/exp04_budget_queryeff/space_ab_compare.py`
- 데이터: `results/attack/*`(292 전-arm verified), `results/benign/*`(무해 prior)
- 결과: `results/space_ab_compare.json`
- 관련: `PRIOR_STRENGTHENING.md`(포화 축 제거·온라인 subspace), `ARM_EXPANSION_IMPACT.md`

## C-공간 기본 selector (2026-09-11)
C 안에서 측정한 결과 **`ucb_add_pibo` = comp×will 가산(factorized) 커널 GP-UCB + πBO 감쇠 prior**가
가장 일관적. (qwen7b: MJ AUC 0.706/@3 0.507, LG AUC 0.911/@3 0.756 — 두 데이터셋 상위.)
- 가산 커널 > 등방 커널(구조 일반화). πBO는 prior 유효 시 가속(LG 0.852→0.911), 포화 시 구조 탐색 폴백.
- TTTS/무거운 온라인 프루닝은 C에서 이득 적음(이미 무의미 스택이 제거된 공간이라).
- `online_selector.py`의 라이브 selector가 이 방식(가산 커널 + πBO)이라 기본으로 채택.

## 서술(narrative) 권고
"D까지 만들고 C로 되돌렸다"가 아니라 **C를 방법으로 제시 + A/B/C/D를 정당화 ablation으로** 둔다.
- 본문: 공간 C(이해×언어×배열 × 의지 중간)를 설정 공간으로 도입.
- ablation(부록/표): A→B 급등(의지 포함이 필수), B→C 상승, **C→D 정체(달성 ASR 동일) + D는 benign-only
  희석·비용 2배** → "의지 확장은 C에서 수확 체감, 그 이상은 selector가 소예산에서 못 살리고 무료 prior만
  해친다"는 원리로 서술. C는 그 원리의 operating point로 자연히 도출.
- 표현: "sweet spot"보다 "the operating point beyond which returns diminish and the free prior degrades".
