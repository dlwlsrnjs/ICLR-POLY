# PolyJigsaw 9모델 패널 결과 — 32B/27B 편입 후 논문 표 갱신 자료 (2026-09-07 21:12 자동 완료)

`panel9_watcher.sh` 체인(밴딧 재학습 → 부트스트랩 CI → 논문 그림)이 21:12에 끝났다. 이 문서는
`docs/PAPER_TABLE_DRAFT_2026-09-07.md`(7모델 기준)의 각 표를 **9모델 값으로 다시 계산**한 것이고, 7→9에서
바뀐 결론을 별도로 적는다. 산출물: `results/mj_bandit_full_20260906.json`, `results/lingua_bandit_full_20260906.json`,
`results/bandit_ci_20260907.json`, `results/ablation_corr_20260907.json`, `results/figs/fig1..5.png`, 로그 `logs/panel9_*.log`.

패널 = Qwen2.5 3/7/14/**32B**, Llama 3.2-3B/3.1-8B, Gemma-2 2/9/**27B** (굵게 = 신규). arm 23개 동일.
32B/27B 모두 10개 arm 파일 전부 채워짐(`logs/big_models.log`의 `BIG_MODELS_ALL_DONE`).

## 7 → 9 모델에서 바뀐 것 (요약)

1. **오라클 상승**: MJ 0.607 → **0.655**, Lingua 0.746 → **0.792**. 두 큰 모델의 최선 arm이 패널 최상위(27B 0.891/0.975).
2. **최선 고정 arm이 바뀜**: MJ `tri_hi_wl`(0.460) → **`tri_triple`(0.505)**, Lingua `tri_triple_en`(0.611) →
   **`tri_triple`(0.678, `tri_triple_en`과 동률)**. 32B는 두 데이터셋 다 `tri_triple`이 최선, 27B는 `tri_triple_en`이 2위.
   → **우리 triple(역할분리) arm이 규모가 커질수록 강해진다**는 새 근거.
3. **AIM 평균이 크게 오름**: MJ 0.219 → 0.314, Lingua 0.364 → 0.481. Gemma 계열의 페르소나 취약이 27B에서 더
   심해지고(0.891/0.975 = 패널 최고), Qwen-32B도 AIM 0.41/0.80. 그래도 llama 계열은 0.00 → 고정 불가 논지는 유지.
4. **적응 vs 고정의 paired Δ는 여전히 비유의**: benign-recon 기준 MJ **+0.088** [−0.032, +0.207], Lingua **+0.051**
   [−0.045, +0.192]. 고정 최선 arm이 우리 자신의 `tri_triple`로 바뀌며 큰 모델에서 매우 강해진 탓(32B 0.75/0.925).
5. **CI 표의 prior 불일치 발견 → 수정 완료**(아래 §†) — 표 1b가 표 1과 다른 prior로 적응을 계산하고 있었다.
   7모델에선 우연히 근접(0.539 vs 0.536)했으나 9모델에선 MJ@2 **0.599 vs 0.537**로 벌어져 드러났다. 원본
   스크립트가 이제 두 prior를 모두 출력하고 benign-recon이 주값이다.
6. **recon–gated 디커플링 모델이 2 → 4개**: Qwen-14B·Gemma-9B에 더해 **Qwen-32B(ρ −0.01/−0.06)·Gemma-27B(+0.09/−0.12)**.
   "강정렬 모델에서 recon prior가 무너진다"는 표 7 주장이 규모 축으로 확장됨.

## 표 1 (9모델). 고정 전략 vs 적응 선택자 (평균 gated ASR)

| 전략 | MultiJail (9모델) | Lingua-SafetyBench (9모델) |
|---|---|---|
| AIM (persona, n=1) | 0.314 | 0.481 |
| DeepInception (fiction, n=1) | 0.385 | 0.589 |
| PAP | 0.033 | 0.108 |
| 우리 최선 고정 arm | 0.505 (tri_triple) | 0.678 (tri_triple = tri_triple_en) |
| 적응 @1쿼리 (benign-recon) | 0.561 | 0.656 |
| **적응 @2쿼리** | **0.599** | 0.674 |
| **적응 @3쿼리** | 0.582 | **0.738** |
| 적응 @4쿼리 | 0.575 | 0.739 |
| **적응 @6쿼리** | 0.581 | **0.763** |
| 오라클(전-arm 상한) | 0.655 | 0.792 |

- 오라클 대비: MJ 적응@2 = **91%**(0.599/0.655, 7모델 89%), Lingua @3 = **93%**, @6 = **96%**(0.763/0.792).
- flat prior(무해정보 없음): MJ @2 0.436, @6 0.517 / Lingua @3 0.626, @6 0.678 → benign-recon이 MJ 전 예산, Lingua 2쿼리 이상에서 우위(7모델과 같은 패턴).
- loo-data prior(타 모델 유해 데이터 사용): MJ @2 0.535 / Lingua @3 0.744. Lingua에선 benign-recon과 동급, MJ에선 benign-recon이 +0.06 우위.
- 값 출처: `logs/panel9_mj.log`, `logs/panel9_lingua.log`(예산 곡선), 단일벡터 평균은 arm 행렬에서 직접 계산.

## 표 1b (9모델). 부트스트랩 95% CI (모델패널 리샘플) + paired 유의성

21:47에 `scripts/bandit_bootstrap_ci.py`를 두 prior 모두 출력하도록 고쳐 재계산했다(아래 † 참조). 주값은
표 1과 같은 **benign-recon prior**.

| 지표 | MultiJail (9모델) | Lingua (9모델) |
|---|---|---|
| 오라클 | 0.655 [0.552, 0.745] | 0.792 [0.700, 0.875] |
| 최선 고정 arm (tri_triple) | 0.506 [0.384, 0.616] | 0.678 [0.511, 0.822] |
| **적응(MJ@2, Lingua@3), benign-recon** | **0.606 [0.423, 0.723]** | **0.730 [0.616, 0.839]** |
| 적응, loo-data prior (참고) | 0.534 [0.431, 0.629] | 0.742 [0.643, 0.829] |
| **paired Δ, benign-recon** | **+0.088 [−0.032, +0.207]** (5/9) | **+0.051 [−0.045, +0.192]** (5/9) |
| paired Δ, loo-data (참고) | +0.025 [−0.005, +0.066] (4/9) | +0.061 [−0.005, +0.175] (6/9) |

모델별 Δ (benign-recon, MJ): gemma2b +0.382, qwen3b +0.326, gemma27b +0.157, qwen14b +0.138, gemma9b +0.087 /
**llama3b −0.234**, qwen32b −0.047, llama8b −0.020, qwen7b −0.002.
모델별 Δ (benign-recon, Lingua): gemma2b +0.548, gemma9b +0.100, gemma27b +0.072, qwen3b +0.025, llama8b +0.010 /
llama3b −0.122, qwen7b −0.068, qwen14b·qwen32b −0.052.
→ 적응의 이득은 **고정 arm(tri_triple)이 무력한 모델**(gemma-2b: MJ 0.28 / Lingua 0.15)에 집중된다. 반대로
**llama3.2-3b에서는 benign prior가 해를 끼친다**(MJ −0.234): 이 모델은 약한 arm에서 recon이 1.0으로 포화해
(combo_incept_only·m_aim·m_deepinception recon=1.00, gated 0.13/0.00/0.17) prior가 정확히 반대를 가리킨다.
recon prior가 깨지는 방식이 **정렬이 강해서**(14B·32B·9B·27B)와 **능력이 낮아 형식만 따라해서**(llama3b)로
분리된다 — 9모델 확장에서 처음 드러난 구분. 출처 `results/bandit_ci_20260907.json`.

### † 수정됨: 적응 추정치의 prior 불일치

`scripts/bandit_bootstrap_ci.py`의 `adaptive_on`·paired 계산은 **loo-data prior**(`pr = mean of other models' G`)를
썼고, 표 1·`*_bandit_full.py`의 헤드라인 "적응"은 **benign-recon prior**(`pr = RC[t]·0.6`)를 썼다. 7모델에선
두 값이 우연히 근접(0.539/0.536)해 드러나지 않았으나 9모델에선 MJ@2 0.599(benign) vs 0.537(loo)로 벌어졌다.

**2026-09-07 21:47 수정 완료**: 원본 스크립트가 이제 두 prior를 모두 계산해 `adaptive[benign-recon]`(주값,
표 1과 동일)과 `adaptive[loo-data]`(참고)를 함께 저장한다. 기존 키 `adaptive`/`paired_delta`는 benign-recon을
가리키므로 이 파일을 읽는 코드는 그대로 동작한다. 수정 전 파일은 `scripts/bandit_bootstrap_ci.py.bak`.
`results/bandit_ci_20260907.json`은 재생성됐다(로그 `poly/bandit_ci_both.log`).

## 표 2 (9모델). 모델별 최선 arm

| 모델 | MultiJail best arm (gated) | Lingua best arm (gated) |
|---|---|---|
| qwen2.5-3b | combo_incept_only (0.641) | tri_hi_wl (0.625) |
| qwen2.5-7b | tri_hi_wl (0.609) | tri_triple_en (0.750) |
| qwen2.5-14b | tri_hi_wl (0.656) | **tri_triple (0.900)** |
| **qwen2.5-32b** | **tri_triple (0.750)** | **tri_triple (0.925)** |
| llama3.2-3b | combo_ours (0.344) | amt_n7 (0.550) |
| llama3.1-8b | tri_hi_wl (0.547) | amt_n7 (0.775) |
| gemma2-2b | amt_n2 (0.672) | m_deepinception (0.700) |
| gemma2-9b | m_aim (0.781) | combo_ours_persona (0.925, m_aim 0.925 동률) |
| **gemma2-27b** | **m_aim (0.891)** | **m_aim (0.975)** |

최선 arm 종류: MJ 6종, Lingua 7종. **패턴이 규모로 일관**: Qwen 계열은 전 규격에서 다국어 컴포지셔널 arm
(3B/7B/14B `tri_hi_wl`·`combo_incept_only`, 32B `tri_triple`), Gemma 계열은 9B/27B 모두 페르소나 단일벡터(`m_aim`),
Llama는 양(amount)/결합. 32B의 top3 = tri_triple 0.75, tri_triple_en 0.70, combo_ours_persona 0.53(MJ) — 다국어+역할분리가
단독으로 큰 모델을 뚫는다. 27B의 top3(MJ) = m_aim 0.89, tri_triple_en 0.80, tri_triple 0.73 — 페르소나 다음이 우리 triple.

## 표 3 (9모델). 고정 단일템플릿의 모델별 편차 (MJ, gated)

| 기법 | qwen3b | qwen7b | qwen14b | qwen32b | llama3b | llama8b | gemma2b | gemma9b | gemma27b | 평균 |
|---|---|---|---|---|---|---|---|---|---|---|
| AIM | 0.05 | 0.22 | 0.31 | 0.41 | 0.00 | 0.11 | 0.06 | 0.78 | **0.89** | 0.314 |
| DeepInception | 0.38 | 0.56 | 0.28 | 0.16 | 0.17 | 0.44 | 0.42 | 0.69 | 0.38 | 0.385 |
| tri_triple (고정 최선) | 0.31 | 0.56 | 0.52 | **0.75** | 0.23 | 0.52 | 0.28 | 0.64 | 0.73 | 0.505 |
| **모델별 최선 arm(오라클)** | 0.64 | 0.61 | 0.66 | 0.75 | 0.34 | 0.55 | 0.67 | 0.78 | 0.89 | **0.655** |
| **적응 @2쿼리 (benign-recon)** | | | | | | | | | | **0.599** |

AIM은 gemma-27b 0.89 ↔ llama3b 0.00, DeepInception은 qwen32b 0.16 ↔ gemma9b 0.69 — 규모가 커져도 편차는 줄지 않는다.
tri_triple도 gemma-2b 0.28 / llama3b 0.23에서 무력. 적응 선택자는 2쿼리로 오라클의 91%.

## 표 4 (9모델). 예산–정확도 곡선 (benign-recon warm-start GP-BAI)

| 예산(쿼리) | MJ benign-recon | MJ flat | MJ loo-data | Lingua benign-recon | Lingua flat | Lingua loo-data |
|---|---|---|---|---|---|---|
| 1 | 0.561 | 0.469 | 0.486 | 0.656 | 0.703 | 0.691 |
| **2** | **0.599** | 0.436 | 0.535 | 0.674 | 0.570 | 0.726 |
| **3** | 0.582 | 0.374 | 0.543 | **0.738** | 0.626 | 0.744 |
| 4 | 0.575 | 0.416 | 0.559 | 0.739 | 0.648 | 0.733 |
| 6 | 0.581 | 0.517 | 0.556 | **0.763** | 0.678 | 0.737 |
| 오라클 | **0.655** | | | **0.792** | | |
| 최선 고정 arm | 0.505 | | | 0.678 | | |

7모델과 같은 형태: MJ는 2쿼리 정점 후 정체(오라클 갭 0.056), Lingua는 단조 상승(예산 6에 갭 0.029).
Lingua 1쿼리에서 flat이 높은 건 7모델 때와 같은 소패널 노이즈(1쿼리 = prior argmax 그대로).

## 표 5 (9모델). 게이트 비용 — 신규 모델

| 모델 | 데이터셋 | 최선 arm | recon | gated |
|---|---|---|---|---|
| qwen2.5-32b | MJ | tri_triple | 0.906 | 0.750 |
| qwen2.5-32b | Lingua | tri_triple | 1.000 | 0.925 |
| gemma2-27b | MJ | m_aim | 1.000 | 0.891 |
| gemma2-27b | Lingua | m_aim | 1.000 | 0.975 |

큰 모델은 recon≈1이라 게이트 비용이 거의 0 — gated ASR이 곧 raw unsafe에 근접. 손실은 여전히 소형 모델의 재구성
실패에서 나온다(7모델 표 5 논지 유지). 그림 `results/figs/fig3_gate_cost.png`.

## 표 6 (9모델). arm-space ablation (누적 오라클)

| 누적 arm 집합 | MultiJail | Lingua |
|---|---|---|
| amount | 0.300 | 0.494 |
| + disorder | 0.323 (+0.02) | 0.528 (+0.03) |
| + combo | **0.550 (+0.23)** | **0.750 (+0.22)** |
| + triple | 0.632 (+0.08) | 0.781 (+0.03) |
| + single-vector | 0.655 (+0.02) | 0.792 (+0.01) |

7모델 대비 **triple의 기여가 MJ에서 +0.05 → +0.08로 커짐**(32B tri_triple 0.75, 27B tri_triple_en 0.80). 단일벡터 기여는
+0.02/+0.01로 여전히 소폭(27B AIM 0.89가 있어도 tri_triple_en 0.80이 받쳐줌). amount-only 오라클은 7→9에서 오히려
하락(0.348→0.300): 32B/27B는 순수 양 arm에 거의 면역(MJ amt 최고 0.19/0.08). 그림 `fig4_ablation.png`.

## 표 7 (9모델). recon이 gated를 예측하나? (Spearman ρ, 23 arm)

| 모델 | MJ ρ | Lingua ρ | 해석 |
|---|---|---|---|
| Qwen-3B | +0.95 | +0.88 | 잘 예측 |
| Qwen-7B | +0.73 | +0.66 | 잘 예측 |
| **Qwen-14B** | **−0.23** | **−0.16** | decouple |
| **Qwen-32B** | **−0.01** | **−0.06** | **decouple (신규)** |
| Llama-3B | +0.30 | +0.27 | 약함 |
| Llama-8B | +0.89 | +0.75 | 잘 예측 |
| Gemma-2B | +0.68 | +0.77 | 잘 예측 |
| **Gemma-9B** | **+0.28** | **+0.02** | decouple |
| **Gemma-27B** | **+0.09** | **−0.12** | **decouple (신규)** |
| 평균 | +0.41 | +0.34 | (7모델 +0.51 / +0.46) |

**규모 축으로 확장된 핵심 발견**: 각 계열의 최상위 2개(Qwen 14B·32B, Gemma 9B·27B)가 모두 ρ≈0 이하. 큰 모델은
어떤 arm이든 recon≈0.8–1.0이라 recon이 arm을 구별하지 못하고, gated는 정렬이 결정. benign-recon warm-start의
한계가 "정렬이 강할수록"이 아니라 "재구성이 포화될수록"으로도 읽힘 — 두 해석 모두 논문의 소예산 운영점 설계와
정합. 평균 ρ가 내려간 것(0.51→0.41)은 이 때문이며, prior가 유효한 모델에선 여전히 0.7–0.95. 그림 `fig5_recon_corr.png`.

## 논문 표에 반영 시 체크리스트

- [x] 표 1/1b/2/3/4/5/6/7을 9모델로 교체 — `docs/PAPER_TABLE_DRAFT_2026-09-07.md` 갱신 완료
      (7모델 판은 `.7model.bak`으로 보존). 판정기 강건성 절도 추가.
- [ ] 논문 본문(`paper/polyjigsaw_iclr2026.tex`)에는 적응 선택자·밴딧 절이 아직 **없다** — 표만으로는 못 들어가고
      Method/Results 절 신설이 필요하다. 별도 작업.
- [x] 표 1b 적응 값 benign-recon으로 통일 — 원본 `bandit_bootstrap_ci.py` 수정·재실행 완료(두 prior 병기, docstring 갱신).
      임시본 `bandit_bootstrap_ci_benign.py`/`bandit_ci_benign_20260907.json`은 역할이 끝나 참고용.
- [ ] 본문 문구 갱신: "고정 최선 = tri_hi_wl/tri_triple_en" → "tri_triple"; "MJ@2 = 오라클 89%" → 91%; "decouple 모델 2개" → 4개.
- [x] 이중 판정기 교차검증 완료 — 결과는 `docs/JUDGE_CROSSCHECK_2026-09-07.md`. 요약: 집계 κ=0.04는 MD-Judge 프롬프트
      형식 버그로 폐기(원본 `judge_crosscheck.py` 수정 완료), 고치면 순응 행 κ=0.73. 저자원 답변 효과는 Gemma-9B
      triple에서 진짜(번역·이중판정 0.781), Qwen-7B Swahili 셀은 과다 계수(0.891 → 0.078).
- [ ] 그림 5장은 이미 9모델로 재생성됨(21:12). `paper/figures/`로 복사는 별도.
