# exp06 — 17모델 확정(confirmatory) selector 실험 (L40S, 2026-09-14)

L40S 박스가 수집한 **17모델 / 7계열 / 160 arm / MultiJail+Lingua** 패널 위에서 돌린 확정 실험의
코드·집계·통계. 원본 작업트리는 L40S의 `/home/jklee/safe_mental-dpo/ICLR-POLY-latest/`이고,
전체 산출물은 프라이빗 버킷 `hf://buckets/jin-kwon/poly/PolyJigsaw/0913/L40S-only/`에 있다.

## 세 가지 확정 결과

1. **엄격 item-held-out 가산 전이 BO** (`tab_confirm_strict.tex`) — 항목을 source-train 50% /
   target-calibration 25% / target-test 25%로 쪼개고 계열 LOO를 겹쳐, 15개 독립 분할에서 평가.
   저예산(B=2)에서 기존 structured GP 대비 이득이 분명하고, 고예산 이득은 불확실.
2. **동일 item-cost 다중충실도 배분** (`tab_confirm_mfid.tex`) — **음성 결과**. 보정 항목이
   10~16개뿐인 현 프로토콜에서는 얕은 부분측정보다 전량측정이 낫다.
3. **오프라인/온라인 학습 selector** (`tab_confirm_selector.tex`) — PPO 정책은 cold start와
   문맥·피드백 절제에서 이득이 있으나 random/GP를 일관되게 이기지는 못한다.

추가로 `tab_confirm_endtoend.tex`는 탐색 전략(퍼저·트리·QD) 이산 적응판과의 **집계 replay** 비교로,
엄격 결과와 분리해 상위자원 진단으로만 보고한다.

## 레이아웃

- `code/` — 버킷에 올라온 러너. `pilot_advanced_additive_bo.py`(가산 전이 GP 본체),
  `pilot_itemheldout_advanced_bo.py`, `pilot_multifidelity_itemcost_bo.py`,
  `train_current_asr_selector_gpu.py`, `analyze_confirmatory_statistics.py` 등.
  **주의**: 이들이 import 하는 `pilot_family_loo_asr_selector.py`는 아직 버킷에 없다 →
  [`docs/L40S_CODE_GAP_2026-09-15.md`](../../docs/L40S_CODE_GAP_2026-09-15.md).
- `results/` — 1MB 미만 집계·통계만 커밋. `*_summary.json`은
  `scripts/distill_confirmatory_summaries.py`가 17~52MB 원본에서 protocol+summary만 뽑은 것.
  `confirmatory_stats_final_20260914.json`이 계열 클러스터 부트스트랩 CI와 부호뒤집기 p값.
- `results/robustness/` — CVaR-25·minimax 선택 규칙 두 가지만 보관(나머지 민감도는 버킷).

## 표 재생성

```bash
python3 scripts/distill_confirmatory_summaries.py   # 버킷 원본이 있을 때만 필요
python3 scripts/make_confirmatory_tables.py         # paper/tab_confirm_*.tex + confirmatory_numbers.tex
```

## 정직성 경계

- 계열이 7개뿐이라 부호뒤집기 검정의 검정력이 낮다. p값은 다중비교 보정을 하지 않았다.
- 오라클은 사후 최대값이라 달성 목표가 아니라 상한이다.
- 집계 replay(`tab_confirm_endtoend`)는 공격자 측 생성 비용을 반영하지 않으며, 탐색 baseline은
  원 논문 구현의 재현이 아니라 같은 arm 공간 위의 이산 적응판이다.
