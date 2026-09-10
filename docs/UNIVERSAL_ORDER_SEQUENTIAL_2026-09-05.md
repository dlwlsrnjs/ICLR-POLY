# 보편 언어 순서 + 순차 그리디 추가 (universal prior + online greedy)

작성 2026-09-05. 사용자 설계: 대상마다 무해 프로파일을 새로 뜨지 말고, **10개 언어에서 어떤 언어가
보편적으로 탈옥에 좋은지 순서를 미리 고정**한 뒤, 온라인에서 그 순서대로 **하나씩 누적 추가하며
joint ASR로 확인 → 안 오르면 멈추는** 그리디. 유효 6대상(zephyr7b·falcon3_7b 제외)에 라이브 적용.

데이터셋: Lingua-SafetyBench text-dominant([[polyjigsaw-dataset-grounding]]). 방법 골격 MIDAS.

## 1단계 — 보편 순서 확정 (results/lang_rank_20260905)

유효 6대상에서 `EN+단일언어`(9개, n2) 유해 joint를 재고 대상 평균. 두 집계(평균 gated / 평균
per-target rank)가 머리(German·Chinese·Spanish)와 꼬리(Finnish)에서 일치.

**보편 순서 = German > Chinese > Spanish > Russian > Arabic > Japanese > Norwegian > French > Finnish**

- 차이는 완만(gated 0.45–0.56). 압도적 단일 "최악 언어"는 없음.
- **Finnish 최하**: recon 0.683으로 재구성이 깨져 unsafe(0.667)가 높아도 joint 최저. ACL이 말하는
  저자원 언어가 오히려 재구성 게이트에 걸림. 최고 지렛대는 중자원·해독가능·정렬약한 German/Chinese/Spanish.
- Lingua-SafetyBench "non-HRL/non-Latin이 더 위협"과 정합하되, 우리 joint(R AND U)에서는 재구성
  게이트가 저자원 언어를 거른다는 **확장**.

## 2단계 — 순차 그리디 추가 (results/sequential_add_20260905)

고정 보편 순서로 EN+top-k(k=1..9)를 누적하며 유해 joint 측정. 그리디 정지: running best 대비
margin(0.05) 넘게 하락이 patience(2) 연속이면 정지, best-so-far 반환(recon<0.4도 정지).

| 대상 | 그리디 선택 | 그리디 J | 쿼리 | 오라클 | 고정(k4=n5) |
|---|---|---|---|---|---|
| phi35 | k1(n2) | 0.650 | 3 | 0.650 | 0.475 |
| qwen25_7b | k6(n7) | 0.650 | 9 | 0.650 | 0.575 |
| mistral7b | k1(n2) | 0.625 | 3 | 0.625 | 0.550 |
| yi | k2(n3) | 0.525 | 5 | 0.525 | 0.450 |
| qwen3_4b | k4(n5) | 0.650 | 7 | 0.675 | 0.650 |
| granite | k4(n5) | 0.700 | 9 | 0.700 | 0.700 |

**그리디 0.633 vs 단일 고정 최선(k=4) 0.567 = +0.067, 이김 4/6, 평균 6.0 라이브쿼리/9,
오라클 대비 regret 0.004.**

- 강한 해독기(qwen25_7b)는 순서대로 n7까지 밀고, 약한 해독기(phi35/mistral)는 n2에서 멈춤 —
  **양(정지 지점)이 대상별로 자동 적응**. 이것이 핵심: 언어 순서는 보편 고정, 양은 온라인 그리디.
- 정지 규칙 민감도: patience=1(원래)은 granite에서 노이즈 딥(k2)에 속아 조기 정지(그리디 0.612,
  regret 0.025). **"2스텝 기다렸다 멈춤"(patience=2)만으로 오라클 사실상 완전 회복**(regret 0.004).
  margin=0.08이면 5/6·regret 0.000(경미한 튜닝). 40문항 노이즈(±0.05) 때문이라 문항 늘리면 완화.

## 확정 알고리즘

1. (오프라인, 1회) 보편 언어 순서 확정 = 유효 패널 평균 단일언어 joint 내림차순.
2. (온라인, 대상별) EN에서 시작해 보편 순서대로 언어 1개씩 추가, 매 스텝 유해 joint 40배치 측정.
3. running best 대비 2스텝 연속 하락(>0.05) 또는 recon<0.4면 정지, best 구성 채택 (평균 6쿼리).

무해 프로파일 per-target 수집이 불필요(순서는 보편 고정). 유해 라이브 쿼리 ~6회로 오라클 근접.

## 코드/산출물

- `scripts/lang_rank_probe.py`, `scripts/run_lang_rank_panel.py` → `results/lang_rank_20260905/universal_order.json`.
- `scripts/sequential_add.py`, `scripts/run_sequential_panel.py`, `scripts/aggregate_sequential.py`
  → `results/sequential_add_20260905/{<tag>.json, aggregate.json}`.
- 관련: `docs/AMOUNT_SCALING_2026-09-05.md`, `docs/ADAPTIVE_LANGUAGE_C3_2026-09-05.md`.
