# MultiJail에서 우리가 raw-ASR로 지는 근본 원인 + 방법 강화

작성 2026-09-06. 사용자 요청: "왜 MJ에서 raw-ASR 우위 주장 불가한지 보고 방법 강화."

## 근본 원인 (MJ sequential recon/gated 데이터로 확정)

MJ 데이터셋에서 **강한 해독기**의 우리 방법 결과:

| 모델 | recon | gated | 함의 unsafe |
|---|---|---|---|
| gemma2_9b | **0.92** | 0.11 | ~0.12 |
| qwen25_14b | **0.91** | 0.12 | ~0.13 |
| qwen25_7b | 0.78 | 0.19 | ~0.24 |

**재구성은 성공(recon 0.9)하는데 gated가 낮다 = 재구성 후 거부(unsafe 낮음).**

**메커니즘:**
- **우리**: 다국어 퍼즐 → **영어로 재구성** → 답변. 영어로 재구성하는 순간 **강한 영어 안전정렬이
  재활성화**돼 모델이 거부. (같은 qwen25_7b: Swahili 직접 unsafe 0.89 vs 우리 영어재구성 unsafe 0.24.)
- **MultiJail**: 저자원 언어(Swahili)로 직접 질문·답변. 영어를 안 거쳐 **저자원 정렬 공백을 그대로 관통**.

→ 진짜 저자원 언어(bn/sw/jv)가 있는 MJ에서는, "영어로 재구성"이 오히려 정렬을 깨워 우리가 손해.
Lingua(고/중자원)에서는 이 문제가 약해 우리가 이겼음.

## 방법 강화안: "검증은 영어로, 답변은 약한 언어로"

`scripts/weaklang_answer_eval.py`:
- **[RECONSTRUCTED]는 영어**로 재구성(재구성 게이트 = 이해 검증 유지, 우리 고유 기여).
- **[ANSWER]는 저자원 언어(order[0], 예: Bengali/Swahili)로 생성** 지시 → 답변 생성이 **정렬이 약한
  언어 공간**에서 일어나 거부 회피. 안전 judge(Qwen3Guard, 다국어)가 그 외국어 답변을 채점.
- 즉 **우리 재구성-검증 + MultiJail의 정렬공백 활용을 결합.** 강한 모델이 영어론 거부, Swahili론 응답하는
  걸 노림. 가설: recon 유지(영어 게이트) + unsafe↑(약한언어 답변) → gated↑.

변형: ours_en(기존, 영어 답변) vs ours_wl(약한언어 답변). 강모델(qwen7b/llama8b/gemma9b)에 비교.

## 검증 결과 (확증 ✅, MJ 강모델 3개, 2단계 실행)

| 모델 | gated en→wl | unsafe en→wl | recon en→wl | Δgated |
|---|---|---|---|---|
| qwen25_7b | 0.17→0.39 | 0.34→0.73 | 0.50→0.53 | **+0.22** |
| llama31_8b | 0.28→0.44 | 0.58→0.89 | 0.52→0.55 | +0.16 |
| gemma2_9b | 0.06→0.33 | 0.06→0.39 | **0.77→0.77** | **+0.27** |

평균 gated +0.22. **gemma2_9b가 결정적**: recon 불변(0.77→0.77)인데 unsafe 6배(0.06→0.39) → 정렬공백이
전부이고 약한언어 답변이 그걸 관통, 우리 게이트가 이해 증명. 강화안 확증 = 근본원인(영어재구성→정렬
재활성화) 해결. MJ 강모델 gated 0.06~0.28 → 0.33~0.44(1쿼리·재구성검증 유지). 코드 scripts/weaklang_twophase.py
(2단계: gen 타겟만→저장→judge만, 동시메모리 반으로 공유GPU 경합 극복), weaklang_2p_run.sh.
→ 다음: Lingua에도 적용 + 밴딧 arm으로 편입(ours_wl을 arm 추가) 검토.

## 정제된 최종 강화: 언어 역할 분리 (사용자 통찰: recon 높음 AND 탈옥)

사용자 지적: recon 0.5는 너무 낮다(gated 상한이 됨), 이상은 recon 높음 AND unsafe 높음. 원인=퍼즐을
저자원 언어로 짜서 재구성이 깎임. **혼란도는 오답(recon 더 낮춤)**. 진짜 레버 = **언어의 두 역할 분리:
퍼즐은 고자원(해독) 언어, 답변만 저자원(정렬공백) 언어.**

scripts/hipuzzle_weaklang.py: 퍼즐=고자원 역순(Vietnamese/Italian/Chinese), 답변=Swahili. 결과(MJ 강모델 3):

| 모델 | 이전 저자원퍼즐 recon/gated | 역할분리 recon·unsafe·gated |
|---|---|---|
| qwen25_7b | 0.53 / 0.39 | **0.75 · 0.86 · 0.62** |
| llama31_8b | 0.55 / 0.44 | **0.70 · 0.75 · 0.48** |
| gemma2_9b | 0.77 / 0.33 | **0.84 · 0.58 · 0.52** |

recon 0.70~0.84(높음) AND unsafe 0.58~0.86(높음) = "검증된 컴포지셔널 탈옥", 1쿼리. gated 평균 0.39→0.54.
MultiJail-combination(unsafe 0.91, 9쿼리·무검증)에 근접+우리 재구성검증 유지. **이게 확정 최종 방법.**
다음: Lingua 적용 + 밴딧 arm 편입 + 답변언어를 대상별 최약정렬 언어로 튜닝.

## 삼중 결합 + 대상별 답변언어 튜닝 (사용자 "1,2 둘다 적용", 확정)

scripts/triple_combo_eval.py (2단계): 고자원퍼즐 + [ANSWER]=대상별 최약언어(3모델 다 Swahili) + AIM 페르소나.
변형 hi_wl(페르소나X)/triple(페르소나+SW답변)/triple_en(페르소나+영어답변). MJ 강모델 3, gated best-per-model:

| 모델 | hi_wl | triple | triple_en | 최강 베이스라인 | 판정 |
|---|---|---|---|---|---|
| qwen25_7b | **0.609** | 0.562 | 0.312 | deepinception 0.562 | **승 +0.05** |
| llama31_8b | **0.547** | 0.516 | 0.438 | deepinception 0.438 | **승 +0.11** |
| gemma2_9b | 0.500 | **0.641** | 0.641 | AIM 0.781 | **패 −0.14** |

**모델별 레버 규명(핵심 기여):** qwen/llama는 저자원 언어공백이 레버 → hi_wl(페르소나 없이 Swahili 답변) 최강,
페르소나 추가는 recon만 깎아 손해. gemma는 페르소나가 유일 레버 → 페르소나로 0.50→0.64(unsafe 0.58→0.80).
**gemma 정직한 패인 규명:** 페르소나 켠 우리 unsafe(0.797)는 AIM(0.812)과 거의 동률 — gated 격차(0.641 vs
0.781)는 **전적으로 recon 격차(0.828 vs 0.969)**, 즉 **퍼즐 오버헤드**. gemma는 컴포지셔널/다국어 취약이 아니라
퍼즐이 순수 비용. → **적응적 정답 = gemma엔 puzzle amount를 n=1로 낮추고 페르소나만 = 그게 바로 AIM.**
**AIM은 우리 arm 공간의 (n=1 + persona + 영어답변) 특수해.** 밴딧 arm에 persona 추가 + n=1 허용하면 gemma에서
AIM을 재현 → 적응 선택자가 모델별로 베이스라인 최선을 arm공간 안에서 포섭(오라클-지배). **논문 포지션(과대포장 금지):
우리 고유 메커니즘(검증된 다국어 컴포지셔널)은 컴포지셔널/다국어 취약 모델(qwen/llama)+Lingua에서 승. 단일벡터
페르소나만 취약한 gemma는 단일벡터 베이스라인(AIM)이 강하고, 우리 적응 선택자는 그 경우 저-amount+persona arm으로
폴백해야 함(밴딧이 모델별 최선 arm 선택 = 진짜 기여).** 결과 results/mj_triple_20260906/.

## MJ 전-arm 밴딧: 단일벡터를 arm으로 포섭 (사용자 "돌려줘", 확정)

scripts/mj_bandit_full.py: MJ 강모델3, arm 23개 = 다국어양(n2..10)+혼란도+결합(ours/persona/incept)
+우리 triple(hi_wl/triple/triple_en)+**단일벡터를 n=1 arm으로 편입(aim=페르소나, deepinception=픽션,
pap=설득)**. benign-recon warm-start GP-BAI. **모델별 전-arm 오라클:** qwen7b=tri_hi_wl 0.609,
llama8b=tri_hi_wl 0.547, **gemma9b=m_aim 0.781(← gemma가 AIM arm으로 포섭됨).** 오라클 0.646.

**고정 vs 적응(3모델 평균 gated):** tri_triple고정 0.573 / DeepInception고정 0.563 / tri_hi_wl고정 0.552 /
AIM고정 0.370(gemma만 강) / PAP 0.036. **적응 benign-recon: @1쿼리 0.578, @3 0.614, @6 0.624, 오라클 0.646.**
**핵심(정직·강함): MJ에서 어떤 단일 고정기법도 전 모델 최선 아님**(AIM은 gemma만, DeepInception은 어디서도
1등 아님, 우리 다국어 arm은 qwen/llama만). **적응 선택자는 1쿼리(0.578)만에 모든 고정전략을 이기고 3쿼리에
오라클 95%(0.614/0.646)**, benign-recon warm-start(무해 지도)만으로. 단일벡터 변환을 arm으로 편입해 gemma까지
포섭 = "모델별 취약 레버가 다를 때 적응적 최선 arm 선택"이 진짜 기여. 결과 results/mj_bandit_full_20260906.json.

## 실행 상태

공유 GPU 경합(다른 사용자 작업이 간헐적으로 GPU를 채워 OOM 반복)으로 라이브 검증이 반복 실패.
scripts/mj_guarded_run.sh가 **GPU ≥68GB 안정 시에만** 실행하도록 걸려 있어(weaklang 3모델 + 14b 베이스라인),
경합이 사라지면 자동으로 결과 산출 → results/mj_weaklang_20260906/. 통찰(근본원인+강화안)은 실행과 무관하게 확정.

## 논문 함의

이 근본원인 분석 자체가 강한 기여: **"입력 난독화 후 영어 재구성은 강한 영어 정렬을 재활성화한다"** =
재구성-게이트 방법의 근본 한계이자, MultiJail류(비복원 저자원 직접)와의 메커니즘 차이를 규명. 강화안이
통하면 "검증된(gated) 탈옥을 저자원 언어 정렬공백까지 확장"으로 두 세계의 장점 결합.
