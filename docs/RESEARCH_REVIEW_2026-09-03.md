# PolyJigsaw 연구 재개용 검토 — 2026-09-03

현재 판단: 주요 실험 결과와 reconstruction-gated 평가를 중심으로 논문을 이어갈 근거는 있다. 반면 capability-only 법칙 및 BO 질의 효율은 현재 구현과 확대 패널 결과로 확정할 수 없다. 가장 최근 작업은 capability와 alignment를 분리해 설명력을 검증하는 후속 분석이다.

## 확인 범위와 상태

- 인수인계, README, 논문 구조/리뷰 기록, 현재 본문, 주요 표, 실험 프로토콜, 최적화·적합·집계·후속 실행 코드를 검토했다.
- `results/` 직속 JSON 82개는 모두 파싱된다. 이는 파일 형식 확인이며, 모든 수치의 원시 응답 재채점을 뜻하지 않는다.
- 주요 test 집계, 기존 5/6모델 패널 결과, 최신 panel_v2 완료 상태를 대조했다.
- BO 함수의 제어 흐름은 실제 함수를 AST로 추출하고 모형 GP와 합성 데이터로 검증했다. 새 대상 모델 질의나 GPU 실험은 실행하지 않았다.
- 일부 private_artifacts 하위 경로는 OS 파일 권한으로 읽을 수 없다. 원시 응답 전체와 사람 라벨은 검증하지 못했다. 권한을 변경하지 않았다.
- 저장소는 상위 `poly/`가 아니라 `PolyJigsaw/`에 있다. 최근 커밋 이후 수정/미추적 파일이 많으므로 Git HEAD만으로 현 상태를 복원할 수 없다.
- 기존 논문·실험 코드는 수정하지 않고 이 검토 문서만 추가했다.

## 연구가 진행된 경로

1. 초기 C0–C3 조각화/다국어화 파일럿: raw ASR 상승에도 의미 재구성 실패가 컸다. 초기 README의 부정적 결론은 이 단계에 해당한다.
2. VL 카드/게임 실험: wrapper 효과와 번역/정렬 병목을 구분했고, ANSWER만 안전 판정하도록 평가를 수정했다.
3. 공식 병렬 문장 interleaving: 현재 text-only 방법과 reconstruction-gated ASR이 중심이 됐다.
4. 본 논문 실험: Lingua, AttaQ, 다수 대상 모델, 독립 판정기, 통제군, 방어 및 번역 품질 실험을 추가했다.
5. capability–difficulty 설명 및 BO 시뮬레이션: 소수 모델에서 얻은 관계를 예측 법칙으로 확장했다.
6. 최신 작업: 모델 패널을 확장하자 관계가 약해져 `panel_v2`, `alignment_probe`, `capability_probe`, `fit_two_axis.py`로 이동 중이다.

`README.md`, `paper/STRUCTURE_NOTES.md`, `paper/REVIEW_LOG.md`는 유용한 이력이지만 최신 본문/결과를 모두 반영하지 않는다.

## 현재 확인되는 실험 근거

출처: `results/paper_lingua_qwen_method_comparison_test.json`. 아래 값은 저장된 자동 판정 결과이며 사람 검증된 정답률이 아니다.

| 조건 | 실제 N | gated ASR, primary | gated ASR, MD | 재구성률 |
|---|---:|---:|---:|---:|
| English direct | 1,727 | 0.2525 | 0.2409 | 1.0000 |
| Finnish translation | 1,727 | 0.5848 | 0.2814 | 1.0000 |
| Interleaving, 본 논문 주 조건 | 1,727 | 0.6925 | 0.6312 | 0.9456 |
| CSRT all | 649 | 0.2712 | 0.2804 | 1.0000 |
| CSRT k2 | 314 | 0.2834 | 0.2994 | 1.0000 |
| Slot k1 | 649 | 0.5516 | 0.4961 | 0.8644 |

주요 장점은 단순 비거부와 의미 재구성을 구분한 평가, wrapper 통제, 판정기 간 민감도 비교다. 다만 모든 조건이 동일한 1,727개 항목에서 평가된 것은 아니다. paired 통계는 별도의 common-item 집계와 연결해 읽어야 한다.

## 최적화 주장에 직접 영향을 주는 확인 사항

### 1. BO 질의 수의 계수 및 관측 처리 오류

근거: `scripts/optimize_difficulty.py:31–46`.

- 초기 설정을 `order`에 넣은 뒤 매 반복 새 설정을 추가하고 `t`를 반환한다. 두 번째 설정에서 처음 최적값을 발견해도 1을 반환한다.
- 첫 설정의 성공 여부는 새 설정을 추가하기 전에 확인하지 않는다. 따라서 현재 값에 일률적으로 1을 더하는 것으로 해결되지 않는다.
- 아직 새 관측으로 GP에 반영하지 않은 설정의 `ytrue`를 바로 조회해 성공 여부를 확인한다. 이는 offline replay의 사후 진단으로는 가능하지만 실제 온라인 정지 판단과는 다르다.
- 반복마다 `ytrue[order] + noise`를 새로 계산하여 과거 관측값까지 바뀐다. 재관측 비용은 계산하지 않는다.
- 실패 시 `iters+1`을 반환하여, 예산 내 미도달과 실제 도달 시간을 구분하지 않은 채 평균한다.
- random search는 전체 후보에서 첫 최적값까지의 기대 횟수, BO는 제한 예산과 실패 대체값을 사용한다. 비교 대상 통계가 일치하지 않는다.

합성 데이터 검증: 최적값이 두 번째 선택 설정에 있을 때 반환값은 1이었다. 다음 검증에서는 같은 최초 관측의 GP residual이 반복 사이 `-0.947078…`에서 `-0.987995…`로 변했다. 이는 실제 연구 결과를 재계산한 값이 아니라 코드 오류를 확인한 최소 예다.

### 2. 설정 평가 횟수와 모델 질의 횟수는 다르다

근거: `scripts/analyze_panel.py:104–116`, `scripts/optimize_difficulty.py:24–34`.

BO는 이미 계산된 조건별 gated ASR에 합성 Gaussian 잡음을 더해 관측으로 사용한다. 하나의 조건별 ASR에는 다수 문항이 들어 있다. 따라서 현재 실험은 집계 성능표에 대한 offline replay이며, 본문 821–839행의 “harmful queries 1–5회”를 그대로 뒷받침하지 않는다.

최신 panel_v2에서 완료된 각 harmful 조건의 N은 250이다. benign probe도 기본 40문항 × 10조건이므로 “하나의 probe”는 모델 호출 한 번과 다르다. 비용은 설정 수, 문항별 대상 모델 호출 수, 판정 호출 수, 토큰/시간을 구분해야 한다. 아직 실행하지 않은 작은 배치 평가의 비용을 기존 결과에 소급 적용하면 안 된다.

### 3. 본문은 4모델, 결과는 5/6모델

| 근거 | 패널 | capability–slope 상관 |
|---|---:|---:|
| 현재 본문/`tab_panel.tex` | 4 | 0.92라는 주장 |
| `paper_panel_theory_aligned5.json` | 5, Mistral 제외 | 0.8924 |
| `paper_panel_theory_all6.json` | 6, Mistral 포함 | 0.4765 |

6모델 결과에서 Qwen2.5-7B의 warm/vanilla 기록은 6.21/2.54, Mistral은 6.88/6.51이다. 이 수치는 위 BO 오류가 있는 기존 산출물이므로 최종 성능으로 인용하면 안 된다. 다만 저장된 결과 자체도 모든 모델에서 warm-start 우세를 보이지 않는다는 사실은 분명하다.

6모델 LOO에서 Phi의 기록된 regret은 0.264, 즉 26.4%p다. “격자에서 옆 설정”이라는 설명은 손실이 작다는 보장이 아니다. Mistral을 제외하는 것은 연구 범위의 선택일 수 있으나, 포함/제외 결과를 함께 공개하고 사후 제외 여부를 설명해야 한다.

`paper_panel_theory.json` 또한 현재 6모델 값이므로 4모델 표와 대응하지 않는다. `make_fig_panel.py`는 이 경로를 읽는다. 현 PDF의 생성 시점까지 검증하지 않았으므로 그림 내용은 별도 확인이 필요하다.

### 4. 능력 지표와 모형 식별성

근거: `analyze_panel.py:49–71`, 기존 패널 JSON.

- 모형은 `D=n+delta·I(shuffled)`, `R=sigmoid(k(C-D))`를 가정한다.
- 5/6모델 모두 `k`가 설정한 하한 0.05에 붙는다. 6모델에서 D의 최대값은 약 24.57인데 C는 약 43.49–92.69다. C는 관측 범위 밖의 50% 재구성 지점을 외삽한 값이다.
- 이것만으로 모형이 무효라고 결론 내릴 수는 없지만, C를 안정적인 일반 능력 점수로 해석하기 전에 불확실성/경계 민감도를 확인해야 한다.
- 거의 모든 격자에서 재구성이 포화된 모델은 “최적 난이도가 관측 범위 밖”이라는 결론을 자동으로 주지 않는다. 재구성률만으로 조건부 응답 성향을 알 수 없기 때문이다.

확률 항등식 `P(R∩U)=P(R)P(U|R)` 자체는 성립한다. 그러나 sigmoid 형태, 공통 난이도 축, 능력에 따른 compliance 예측, 인과적 “safety re-check 억제”는 각각 별도 경험적 가설이다. 조건마다 재구성에 성공한 항목 집합도 달라질 수 있다. `paper_comply_deconfound.json`의 공통 성공 집합 분석은 유용하지만 인과 식별을 끝내지는 않는다.

### 5. 예측 규칙 및 검증 절차의 불일치

- `analyze_panel.py:82`는 연속 최적화 구간 상한을 21로 고정하지만 6모델의 실제 격자 최대 D는 약 24.57이다. LOO의 연속해 근접 설정과 BO prior의 격자 argmax가 서로 다른 선택을 할 수 있다.
- `fit_two_axis.py:93`은 상한 25를 사용한다. 연속해에 가장 가까운 설정은 일반적으로 이산 후보의 예측 목적함수 최대값과 동일하다고 보장되지 않는다.
- 본문과 코드 주석은 “closed form”이라고 부르지만 구현은 `minimize_scalar(method='bounded')`를 사용하는 수치 최적화다. [SciPy 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html)는 이를 bounded Brent 기반 스칼라 최소화로 설명한다.
- 패널의 k/delta/C는 held-out 모델의 benign 데이터까지 합쳐 적합하고 a(C)/b(C)만 LOO한다. 유해 test label 유출과 동일한 것은 아니지만 완전한 inductive LOO는 아니다. 신규 모델의 benign calibration을 허용하는 평가인지 명시해야 한다.
- 다른 모델의 harmful grid로 compliance 관계를 학습한다. 그러므로 “유해 supervision 없음”은 전체 절차가 아니라 held-out 대상의 적응 단계에 한정되는 주장이다.
- 두 축 분석의 `A_en_panel`, `A_fi_panel`은 평가 패널의 직접 요청 성능이다. 독립 dev probe와 같은 것으로 취급할 수 없다.
- `fit_two_axis.py`는 여러 feature 조합의 LOO 결과를 비교한다. 그 결과를 보고 최선의 feature를 고르면 그 선택 자체에 대한 추가 검증이 필요하다. best fixed도 전체 평가 모델에서 고른 사후 기준이다.

## 가장 최근 후속 작업의 상태

검토 시점의 산출물 존재 여부 기준이며, 프로세스가 종료되었는지 판단한 것은 아니다.

- panel_v2에서 benign/harmful 집계가 모두 존재: Falcon3-7B, Mistral-7B, Phi-3.5, Qwen2.5-14B, Qwen2.5-32B, Qwen3-8B.
- benign 집계만 존재: Qwen2.5-3B, Qwen2.5-7B.
- OLMo-2-7B 디렉터리는 있으나 위 집계는 없음.
- `paper_alignment_probe.json`, `paper_overrefusal_probe.json`, `paper_capability_probe_decomp.json`, `paper_two_axis_law.json`은 아직 없음.
- `alignment_probe/scoring_v2.log`, `capability_probe/gpu0.log`, `gpu1.log`는 확인 시점에 비어 있었음.
- `chain_capability.sh`는 alignment scoring의 완료 문자열을 기다린다. 완료 결과가 없는 상태에서 두 축 모형의 성공을 주장할 수 없다.

`fit_two_axis.py`의 방향은 재구성 능력과 안전 응답 성향을 분리하는 것이다. benign 분해는 영어 재조립 능력 C_en과 언어별 이해 M을 측정하고, alignment probe는 과잉 거부와 직접 요청 응답 성향을 따로 측정한다. 이 개념적 분리는 타당한 다음 가설이지만 성능 개선은 아직 검증되지 않았다.

## 최적화 외 본문 정합성 문제

1. 본문 439행 근처의 동일 1,727개 항목 비교 표현과 `tab_main.tex`는 조건별 실제 N을 누락한다. CSRT/slot은 649 또는 314개다. paired 검정과 표의 전체 평균 차이를 혼동하지 않아야 한다.
2. 같은 단락의 “모든 baseline HR≥3 비율 ≤0.16”은 집계와 다르다. English는 0.2588, CSRT k2는 0.2643이다. 제안 방법 0.414보다 낮다는 것과 0.16 이하라는 것은 다른 주장이다.
3. `compare_methods.py:137–141`은 현재 split 안에서 최선 번역을 고르고, `make_paper_tables.py:107–109`도 test 집계에서 interleave 최댓값을 선택한다. Qwen의 Finnish는 dev/test가 일치하지만 생성 절차가 dev 선택을 강제하지 않는다. dev-selected 결과와 test-best/oracle 결과를 구분해야 한다.
4. 본문은 BO를 완성된 기여로 설명하면서 Discussion 943행 이후에는 완전한 adaptive controller를 미래 작업으로 남긴다. 평가 범위를 통일해야 한다.
5. machine translation robustness를 보고하면서 Limitations/Ethics에는 공식 번역만 사용하고 새 번역이 없다는 문구가 남아 있다.
6. 본문의 모든 표가 한 스크립트로 재생성된다는 설명과 달리 `make_paper_tables.py` 실행 목록에는 panel/two-axis 등 모든 표가 들어 있지 않다.
7. 원 데이터의 dev/validation/test와 본 논문용 40/60 dev/test가 중첩된 분할이다. panel_v2는 전자의 test 960개 중 250개를 사용하도록 작성됐다. 동일한 이름의 test를 같은 모집단으로 취급하지 않아야 한다. 원시 ID의 전면 교차검증은 하지 않았다.
8. 표의 일부 baseline은 문헌 기법의 대표 변형이다. 원 기법 전체 재현이라는 표현은 구현 범위 검증과 구분해야 한다. 이번 검토에서는 선행논문 전체와 재현 코드를 대조하지 않았다.

## 논문 작업의 권장 순서

1. 기존 주요 실험과 탐색적 최적화를 명확히 구분하고, 본문의 현재 강한 예측/효율 주장을 보류한다.
2. BO 평가의 계산 단위, 계수, 실패 처리, 관측 보존을 먼저 바로잡아 재현 가능한 측정으로 만든다. 수정 전 숫자를 토대로 알고리즘 간 우열을 결정하지 않는다.
3. 현재 두 축 후속 실험의 완료 산출물을 확보한 뒤, 데이터 분할·probe 비용·모델 패널 범위를 고정한 상태에서 설명력과 일반화를 판단한다.
4. 모델별 regret와 불확실성을 중심으로 예측의 유용성을 평가한다. 사후 oracle 도달 여부만으로 실용적인 정지 규칙을 주장하지 않는다.
5. 논문 표/본문의 N, 모델 수, 판정기, split, 수치 출처를 연결하고 사람 검증 미완료를 유지한다.

지금 지지되는 논문 방향은 “재구성 능력과 안전 응답 성향이 서로 다른 역할을 하며, 재구성 성공을 조건으로 안전 실패를 평가해야 한다”이다. “능력 하나로 모든 모델의 최적 난이도를 예측하고 소수 실제 질의로 도달한다”는 더 강한 주장은 현재 증거보다 앞서 있다.
