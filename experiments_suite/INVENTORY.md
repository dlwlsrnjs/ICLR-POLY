# 논문 실험 인벤토리 — 표/실험 → 코드 → 수집 상태

논문 본문·어펜딕스의 모든 표를 실험 단위로 묶고, 각각을 스위트 폴더·생성 스크립트·상태에 매핑한다.
정직한 현재 상태: **exp01(폐쇄)만 새 스위트에 데이터셋 분리·292 공간으로 있고**, 나머지 표는 아직
`scripts/`의 옛 flat 스크립트(대부분 23-arm, MJ/LG를 내부에서 함께 처리)로 만들어진다. 아래 표의
"스위트 상태"가 그 이관 진행도다.

## 공유 자산: 패널 292-arm verified 매트릭스

본문 핵심표 상당수(이질성·예산·held-out)는 **모델별 전-arm verified 매트릭스** 하나를 공유한다.
그래서 `exp02_panel_collect`가 이 매트릭스를 수집하고(모델 루프 × probe + full-matrix attack),
exp03/04/05가 그 매트릭스를 분석만 한다. 292 공간 패널 수집은 16모델×2데이터셋 유해 재수집이라
공유 GPU 수일이 든다(코드는 준비, 실행은 GPU 예약 후).

## 매핑

| 실험(스위트) | 논문 표/수치 | 측정 | 생성 스크립트(현행) | 데이터셋 처리 | 상태 |
|---|---|---|---|---|---|
| exp01_closed_compare | tab_closed, tab_g4axis | 폐쇄모델 우리 vs baseline (verified) | `closed_compare.py`(신규) | **분리(mj/lg)** | ✅ 완료·파일럿 |
| exp02_panel_collect | (매트릭스 원천) | 모델별 전-arm recon/unsafe/verified | `closed_compare.py --all-arms`(신규) | **분리** | 🟡 코드 완료·수집 GPU 대기 |
| exp03_heterogeneity | tab_headline, tab_heldin, tab_sel_permodel, tab_sel_ablation, tab_sel_persona, tab_sel_metrics, heldin_numbers | 고정최적 불가·모델별 최선·persona 축·gate 비용 | `permodel_technique_tables.py`, `make_selector_tables.py` | 내부 공용→분리 예정 | 🟡 분석기 이관 |
| exp04_budget_queryeff | tab_sel_queryeff, tab_sel_budget, tab_strat, tab_acq, tab_sel_batch, query/strat/acq_numbers | 예산-정확도, 질의효율, 전략·획득함수 | `query_efficiency.py`, `query_eff_strat4.py`, `acq_compare.py`, `gp_bai.py` | 내부 공용→분리 예정 | 🟡 분석기 이관 |
| exp05_heldout | tab_heldout, heldout_numbers, heldout_ablation, tab_sel_prior | held-out 전이·probe 선정 | `heldout_selector.py`, `benign_prior_selection.py` | 내부 공용→분리 예정 | 🟡 분석기 이관 |
| exp06_mechanism | factorization_numbers, tab_mechanism, tab_sel_corr | 이해×순응 인수분해 | factorization 스크립트 | 대체로 공간무관 | 🟡 |
| exp07_stealth_ppl | ppl_numbers, ppl_nonqwen_numbers, tab_encoding | perplexity 스텔스, 인코딩 대비 | ppl/encoding 스크립트 | 공간무관 | 🟡 |
| exp08_judge | tab_gateval, tab_safetyjudge, robust_numbers | judge 교차검증·저자원 번역판정·승자저주 | `judge_crosscheck*.py`, `robustness_checks.py` | 공간무관/내부 | 🟡 |
| exp09_domain | tab_sel_domain, tab_sel_domainperm, tab_sel_domainsel, domain_numbers | 도메인 순열(음성)·적응비용 | `domain_analysis.py`, `domain_permutation.py`, `domain_selector.py` | 내부 | 🟡 |
| exp10_appendix_controls | tab_granularity, tab_scenario, tab_langpairs, tab_languages, tab_matched, tab_mtrobust, tab_defense, tab_detect, tab_control, tab_rawgated, tab_difficulty | 조각세분·시나리오·언어쌍·방어/탐지 등 | 각 전용 스크립트 | 내부 | 🟡 |

## 원칙(모든 실험 공통)

- 데이터셋별 드라이버 분리(`mj.py`/`lg.py` 또는 `collect_mj.py`/`collect_lg.py`): 언어·order·tlang이
  다르므로. 엔진이 컬렉션 자동 해석 + tlang 존재 검증.
- 수집은 `results/`에 `MANIFEST.json` + 집계 JSON + `_raw`(0600). 분석은 MANIFEST/집계만 읽는다.
- 무해(probe/analysis)는 Claude 실행 가능, 유해(attack/full-matrix)는 연구자 실행.
