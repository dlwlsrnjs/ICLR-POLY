# 개정 계획 (2026-09-10) — 어떻게 고칠 것인가

이번 세션에서 정한 방향과, 그것이 논문 본문·어펜딕스·표에 어떻게 반영되는지 정리한다. 근거·영향은
[experiments_suite/ARM_EXPANSION_IMPACT.md], 실험↔표 매핑은 [experiments_suite/INVENTORY.md],
폐쇄모델 실험은 [experiments_suite/exp01_closed_compare], 재현은 [docs/REPRODUCE.md] 참조.

## 1. 핵심 변경 방향

1. **arm 공간을 comprehension × willingness 교차곱으로 확장(23 → 292).**
   - 재구성 셀 32 = frag{3,5,8,12} × {ordered,shuffled} × n{2,4,6,8}. `n`은 데이터셋 언어 수로 자동 상한.
   - willingness = {persona, fiction, pap}의 멱집합(plain 포함 8, **스택 허용**) + role = 셀당 9.
   - 32 × 9 = 288 + 단일벡터 baseline 4 = 292. 이렇게 하면 selector가 "더 어려운 재구성 + 순응 프레임"을
     함께 고를 수 있다(기존 23-arm은 willingness가 n=4 퍼즐 위에만 있어 불가능했다).
   - 중복 감사: byte 동일 0, 한 셀을 4 willingness가 공유(의도된 교차). `closed_compare.py audit`.

2. **데이터셋별 코드 분리.** MultiJail(10언어, 저자원 우선, tlang=Bengali)과 Lingua(10언어, tlang=
   Norwegian)는 언어·order·번역언어가 다르다. 드라이버를 `mj.py`/`lg.py`로 분리하고 엔진이 컬렉션
   자동 해석 + tlang 존재 검증(없으면 translated가 조용히 영어로 폴백하는 버그 차단).

3. **폐쇄모델 전이 실험 정식화.** held-in 동결 selector를 GPT-4o 등에 적용. 무해 평문 probe로
   세팅 선택(재구성 포화 시 willingness 순응이 주도) → shortlist → 유해 확정 풀이 + baseline 비교.
   과정을 §transfer에 서술(방법 문단은 exp01 문서의 초안 사용).

## 2. 표·섹션별 수정

| 표/수치 | 어떻게 바뀌나 |
|---|---|
| tab_headline, tab_heldin, tab_heldout, tab_sel_permodel | 292 공간 전-arm 매트릭스로 재계산. 오라클↑, 고정최선↓ 예상 → 적응 이득 재평가 |
| tab_sel_ablation | amount/disorder/composition 축 기여를 교차곱 기준으로 재분해 |
| tab_sel_prior, app_heldin_probe | probe 레시피 LOO 선정을 292에서 재실행. 포화 모델은 willingness 순응이 warm-start를 담당함을 명시 |
| tab_sel_queryeff, tab_sel_budget, tab_strat, tab_acq | GP 특징 인코딩을 (F,arr,n)+willingness 멀티핫으로 확장 후 예산·질의효율 재계산 |
| tab_sel_robust (winner's curse) | 292개 max는 인플레이션이 커짐 → 디바이어싱 강화, 문항 수 상향 검토. "오라클의 X%" 재보정 |
| heterogeneity p값(\pjfamp) | arm 급증으로 동률↑ → permutation 검정 재계산, soft-range 확대 가능(약화 주의) |
| tab_closed / tab_g4axis | 폐쇄모델 결과를 exp01 수집물로 갱신. verified 기준 head-to-head |
| gate 규칙(§3) | 불변. 교차·스택 arm 전부 퍼즐 보유→gated, n=1 단일벡터만 ungated |
| tab_encoding, ppl_*, tab_gateval, domain_* | 대체로 공간무관, 유지 |

## 3. 정직성 경계(유지)

- 성공 단위는 **verified**(재구성 AND unsafe). raw 비거부로 성공 주장 안 함.
- 적응 vs 고정최선 paired Δ는 여전히 소표본에서 비유의일 수 있음 → 과대주장 금지.
- 오라클 이득은 winner's-curse 보정 후 보고. 292에서 보정폭이 커짐을 명시.
- 폐쇄모델은 전이 사례(calibration 포함)이며, 매칭된 전략 비교가 아니면 그렇게 표기.
- probe는 무해, 유해는 최종 평가에서만. 무해 probe 요청 수·유해 요청 수 분리 보고.

## 4. 수행 순서

1. 대표 모델 소수(qwen7b·gemma9b·llama8b + 폐쇄 1)에서 **292 전수 수집**으로 오라클/이질성 검증.
2. GP 특징 인코딩·winner's-curse·probe 레시피를 292로 재구현.
3. 나머지 패널은 shortlist 유해 평가로 비용 절감(전수는 대표 모델만).
4. 매크로 재생성(`*_numbers.tex`) → 표 갱신 → PDF 재빌드.
5. 본문 서술을 위 변경에 맞춰 조정, 정직성 문단 유지.

## 5. 산출물 위치(버킷/레포)

- 코드·수집기·분석기: `experiments_suite/`(exp01 완료, exp02 수집기·exp03 분석기 파일럿 통과).
- 데이터셋 입력: `private_artifacts/{multijail_v1,panel_v2}/`(harm_grid·benign_probe·order).
- 최신 논문: `paper/polyjigsaw_iclr2026.{tex,pdf}`(2026-09-10 재빌드, 미해결 참조 0).
