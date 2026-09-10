# PolyJigsaw 논문 표 — 최종본 (9모델, 일관 게이트 규칙, 2026-09-07 재측정)

**이 문서는 이제 참고용이다.** 논문 `paper/polyjigsaw_iclr2026.tex`가 정본이고, 표는
`scripts/make_selector_tables.py`가 아티팩트에서 직접 생성한다(`paper/tab_sel_*.tex`,
`paper/selector_numbers.tex`). 본문에 인용되는 수치도 전부 그 매크로를 쓰므로 표와 본문이 어긋날 수 없다.

## 이 판에서 바뀐 것

1. **채점 규칙 통일**(`scripts/arm_scoring.py`). 요청을 영어 평문으로 주는 arm(AIM·DeepInception·PAP·
   fiction only)은 재구성 게이트를 **일률 면제**, 요청을 숨기는 arm(우리 전부)은 **엄격 유지**. 이전에는
   "모델이 태그를 냈는지"에 따라 걸리기도 안 걸리기도 하는 출력 의존 규칙이었다
   (`docs/DATA_AUDIT_2026-09-07.md`). 새 규칙은 baseline에 가장 유리한 쪽이다.
2. **몬테카를로 잡음 제거.** 예산 곡선 reps 40 → 200, 그리고 화면 출력과 JSON 저장이 같은 draw를 쓰도록
   수정(이전에는 같은 RNG를 이어 소비해 로그와 JSON이 ±0.01 달랐다).
3. **운영점 통일.** CI 표의 예산을 두 데이터셋 모두 3쿼리로 맞췄다(이전에는 MJ 2 / Lingua 3).
4. 위 세 가지를 반영해 밴딧·CI·그림·표를 전부 재생성했다.

## 최종 수치

| 전략 | MultiJail (9모델) | Lingua (9모델) |
|---|---|---|
| PAP | 0.090 | 0.197 |
| AIM (게이트 면제) | 0.491 | 0.658 |
| DeepInception (게이트 면제) | 0.450 | 0.639 |
| 우리 최선 고정 arm (tri_triple) | 0.505 | 0.678 |
| 적응 @1쿼리 | 0.543 | 0.658 |
| 적응 @2쿼리 | 0.421 | 0.664 |
| **적응 @3쿼리 (운영점)** | **0.624** | **0.721** |
| 적응 @6쿼리 | 0.651 | 0.783 |
| 오라클 | 0.675 | 0.803 |

오라클 대비 @3 = **92% / 90%**, @6 = **96% / 98%**.

**prior 비교(@3):** benign 0.624 vs 무정보 0.370 vs 타모델유해 0.549 (MJ) / 0.721 vs 0.680 vs 0.770 (Lingua).
MJ에서는 **무해 프로브가 타 모델의 유해 결과보다 낫다**. 예산 2에서 MJ가 0.421로 내려앉는 비단조성은
그대로 보고한다(첫 프로브가 사전분포를 무너뜨린 뒤 두 번째로 복구가 안 되는 구간).

**부트스트랩 CI(@3, 9모델 리샘플):** MJ 적응 0.616 [0.482, 0.729] vs 고정 0.506 [0.384, 0.616],
paired Δ **+0.107** [−0.012, +0.243] (5/9). Lingua 적응 0.724 [0.622, 0.824] vs 고정 0.678,
Δ **+0.048** [−0.057, +0.206] (4/9). **둘 다 비유의** — 논문에서도 그렇게 쓴다.

**모델별 최선 arm(8종):** qwen3b fiction-only(0.734/0.650), qwen7b role-split(0.609)/role-split+persona-EN(0.750),
qwen14b role-split(0.656)/role-split+persona(0.900), qwen32b role-split+persona(0.750/0.925),
llama3b combination(0.344)/amount n7(0.550), llama8b role-split(0.547)/amount n7(0.775),
gemma2b AIM(0.734/0.775), gemma9b AIM(0.812)/combination+persona(0.925), gemma27b AIM(0.891/0.975).
우리 arm이 이기는 모델 5/9(MJ), 6/9(Lingua). 나머지는 단일벡터를 arm으로 편입했기 때문에 커버된다.

**arm 계열 ablation(누적 오라클):** amount 0.300/0.494 → +disorder 0.323/0.528 → **+combination 0.569/0.769**
→ **+role separation 0.642/0.794** → +single-vector 0.675/0.803. 결합이 최대 기여, 우리 역할분리가 그다음.
amount 단독은 32B·27B에 거의 무력(MJ 최고 0.188/0.078).

**recon–ASR Spearman:** qwen3b +0.74/+0.95, qwen7b +0.65/+0.48, llama8b +0.79/+0.48, gemma2b +0.65/+0.74 는
잘 예측. **qwen14b −0.00/+0.04, qwen32b +0.11/−0.09, gemma9b +0.25/+0.02, gemma27b +0.23/+0.05 는 붕괴** —
각 계열의 큰 모델. 무해 prior의 한계이자 정렬의 측정치.

**판정기 강건성:** 저자원 답변을 직접 판정하면 신뢰 불가(같은 행에 Refusal=Yes와 Unsafe가 105/384).
번역 후 판정하면 두 판정기가 순응 행에서 κ=0.73으로 일치. Gemma-9B 역할분리는 0.812 → 0.781로 거의 불변
(진짜), Qwen-7B는 0.891 → 0.078로 붕괴(과다 계수). 상세는 `docs/JUDGE_CROSSCHECK_2026-09-07.md`.

## 논문 구조 (2026-09-07 개편)

본문 9쪽: ① 서론 ② 관련연구 ③ 검증된 공격성공(게이트 규칙 포함) ④ 구성공간 23개 ⑤ 무해 warm-start 선택자
⑥ 실험(고정 불가 → 3쿼리로 near-oracle → ablation → prior 붕괴 → 판정기) ⑦ 논의·한계 ⑧ 결론.
부록: 단일타깃 인터리빙 연구 전체(언어부하 곡선, CSRT 매칭, 인코딩 baseline, 상용모델, 탐지·방어,
thinking 모드, 시나리오별), 무결성 감사, 프롬프트 원문, 파이프라인.

## 재현

```bash
python scripts/audit_panel_integrity.py     # 90개 입력 파일 불변식 검사
python scripts/mj_bandit_full.py            # MJ 밴딧 (arm_scoring 규칙 적용)
python scripts/lingua_bandit_full.py
python scripts/bandit_bootstrap_ci.py       # 두 prior CI, 예산 3
python scripts/make_paper_figs.py           # results/figs/fig1..5
python scripts/make_selector_tables.py      # paper/tab_sel_*.tex + selector_numbers.tex
```
