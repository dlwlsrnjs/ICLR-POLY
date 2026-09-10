# Arm 공간 확장(23 → 292)이 기존 실험 방식에 미치는 영향

23-arm(amount/disorder/combo/triple/single-vector)에서 **292-arm**(재구성 32셀 × willingness 9,
스택 포함)로 넓히면 아래 방법들이 바뀌어야 한다. 단순히 arm만 늘리는 게 아니라 여러 분석의
전제가 달라진다.

## 1. 선택기(GP-BAI) 특징 인코딩 — **재설계 필요**
- 기존 특징 = [n/10, δ, is_disorder, single-vector one-hot]. 292 공간은 **(F, arrangement, n) 재구성
  좌표 + willingness 멀티핫(persona/fiction/pap) + role 플래그**로 확장해야 커널이 이웃 arm을 옳게
  일반화한다.
- 스택(persona+fiction 등)은 멀티핫의 합으로 표현 → 커널이 "부분적으로 겹치는 willingness"를 잇는다.
- 영향 표: `tab_sel_queryeff`, `tab_sel_budget`, `tab_strat`, `tab_acq` 전부 재계산.

## 2. 오라클 인플레이션 / 승자의 저주 — **더 커짐, 디바이어싱 강화 필수**
- 오라클 = 292개 중 최대(40~64문항). arm이 23→292로 늘면 max의 상향 편의가 **훨씬 커진다**.
- 기존 winner's-curse 보정(`tab_sel_robust`, `robust_numbers`)의 split-half/worst-case 보정폭이
  커지고, **디바이어싱된 오라클 이득이 더 줄 수 있다**(음의 값 가능). 문항 수를 늘리거나 보정을
  강화하지 않으면 "오라클의 X%" 주장 신뢰도가 떨어진다.
- → 292에서는 문항 수 상향(예: 64→128) 또는 반복 평가 권장.

## 3. 무해 prior / probe 레시피 — **셀×프레임으로 재정의**
- prior가 이제 (셀 재구성 recon) × (willingness 프레임 순응)로 292 벡터를 만든다. 재구성은 셀당 1회
  측정(32) → 292로 매핑. willingness는 프레임/스택별 순응.
- held-in LOO 레시피 선정(`benign_prior_selection.py`, `tab_sel_prior`, app_heldin_probe)을 292에서
  다시 돌려야 한다. 포화 모델(GPT-4o)에서는 recon이 안 갈리므로 willingness 순응이 주도(이미 반영).

## 4. 이질성 / 계열-승자 검정 — **재계산, 약화 가능**
- distinct winners는 arm이 많아지면 자연히 늘어(292 중 최선) "고정 최적 불가"가 더 강해 보이지만,
  1-SE 내 동률(ties)도 급증한다. `permodel`·permutation p값(`\pjfamp`)을 292에서 다시 계산해야 하고,
  다중비교로 **약해질 수 있다**. soft-range(3–8) 대신 더 넓은 밴드가 필요.

## 5. gate 규칙 — **불변(확인 완료)**
- 292 교차 arm은 전부 퍼즐 보유 → gated. 스택 willingness도 퍼즐 위라 gated. n=1 단일벡터
  baseline만 ungated. 기존 규칙 그대로 적용된다(추가 분류 불필요).

## 6. 수집 비용 / 통계력 — **패널 재수집이 병목**
- full-matrix(전 arm) × 16모델 × 2데이터셋 × 문항 = 대규모. 292 전수는 며칠. 대안: (a) 전 arm은
  소수 대표 모델만, (b) 나머지 모델은 phase-1 shortlist만 유해 평가(오라클은 근사). 논문 오라클
  주장을 유지하려면 최소 대표 모델에서 전수 필요.

## 7. 논문 서술/표 — **갱신 목록**
- 재계산 대상: `tab_headline, tab_heldin, tab_heldout, tab_sel_permodel, tab_sel_ablation,
  tab_sel_prior, tab_sel_queryeff, tab_sel_budget, tab_sel_robust, tab_strat, tab_acq,
  heldin/heldout/robust/strat/acq_numbers, factorization_numbers`.
- 공간무관(대체로 유지): `tab_encoding, ppl_*, tab_gateval, tab_safetyjudge, domain_*`.

## 권고 순서
1. 대표 모델 소수(예: qwen7b·gemma9b·llama8b + 폐쇄 1)에서 **292 전수** 수집 → 오라클/이질성 검증.
2. 특징 인코딩·winner's-curse·probe 레시피를 292로 재구현(위 1·2·3).
3. 나머지 패널은 shortlist 유해 평가로 비용 절감.
4. 표/수치 매크로 재생성 후 논문 서술 조정.
