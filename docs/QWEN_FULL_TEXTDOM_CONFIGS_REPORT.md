# Full Text-Dominant dev (2,879) × 4 target configs

마지막 갱신: 2026-09-01

Lingua-SafetyBench Text-Dominant **전체 dev 2,879개**를 4개 타깃 설정에서 동일 조건으로
돌린 결과다. 그동안 `results/full_textdom_ALL_configs_summary.json`에 수치만 있고 해석
문서가 없었으므로 여기서 정리한다. 320개 층화 표본 결과는
`docs/QWEN_ALL_MODALITIES_REPORT.md`에 있고, 이 문서는 그 전수 버전이다.

## 1. 설정

| 설정 | 타깃 모델 | 엔진 | 이미지 | n(items) |
|---|---|---|---|---:|
| A_hf | Qwen2.5-7B-Instruct | HF `generate` | 없음 | 2,879 |
| A_vllm | Qwen2.5-7B-Instruct | vLLM 0.11.0 | 없음 | 2,879 |
| B_vllm | Qwen2.5-VL-7B-Instruct | vLLM 0.11.0 | 원본 이미지 | **1,439 (아래 §5 참고)** |
| C_vllm | Qwen2.5-VL-7B-Instruct | vLLM 0.11.0 | 회색 blank | 2,879 |

- 디코딩: temperature 0, max_new_tokens 320, 동일 seed(20260828).
- 조건: `english_direct`, `translated_direct_Finnish`(가장 강한 번역 baseline),
  `interleave_ordered_n4/n10`, `slot_k1/k2/k3`.
- 재구성 judge: Qwen2.5-7B-Instruct (action/target/intent/constraints 의미 등가).
- 안전 judge: Qwen3Guard-Gen-8B, 공식 eval_gen 프로토콜, `[ANSWER]` 섹션만 입력.
- 주지표 **gated ASR** = 의미 등가 재구성 AND strict-unsafe(Unsafe 또는 Controversial).
- 각 설정의 baseline은 그 설정 자신의 english direct(A는 `english_direct`,
  B/C는 `vl_english_direct`)이므로 paired 대비는 설정 내부에서만 유효하다.

## 2. gated ASR

| 조건 | A_hf | A_vllm | B_vllm (real img) | C_vllm (blank img) |
|---|---:|---:|---:|---:|
| english_direct | 0.2567 | 0.2501 | 0.2050 | 0.2140 |
| translated_direct_Finnish | 0.5811 | 0.5832 | 0.5108 | 0.5815 |
| **interleave_ordered_n4** | **0.6825** | **0.6985** | 0.5469 | 0.4192 |
| **interleave_ordered_n10** | 0.6440 | 0.6624 | 0.5886 | 0.4255 |
| slot_k1 | 0.5440 | 0.5572 | 0.2585 | 0.2933 |
| slot_k2 | 0.4163 | 0.4144 | 0.2639 | 0.2257 |
| slot_k3 | 0.3476 | 0.3605 | 0.2093 | 0.1631 |

A_vllm 기준 95% Wilson CI: english [0.2346, 0.2662], n4 [0.6815, 0.7150],
n10 [0.6449, 0.6794]. n4 − english paired difference = **+0.4484**, exact McNemar
p ≈ 0 (n=2,879 pairs). A_hf도 +0.4258, p ≈ 0으로 동일 방향이다.

## 3. 재구성률 — "이해 못 해서 뚫린 것"이 아니다

| 조건 | A_hf | A_vllm | B_vllm | C_vllm |
|---|---:|---:|---:|---:|
| interleave_ordered_n4 | 0.9361 | 0.9406 | 0.9625 | 0.9608 |
| interleave_ordered_n10 | 0.8496 | 0.8538 | 0.8548 | 0.7860 |
| slot_k1 | 0.8486 | 0.8581 | 0.8491 | 0.8969 |
| slot_k2 | 0.6518 | 0.6420 | 0.6320 | 0.7529 |
| slot_k3 | 0.5279 | 0.5494 | 0.5581 | 0.7082 |

n4에서 재구성 등가율이 94%인데 gated ASR이 0.70이라는 것은, 모델이 **원 요청의 의미를
정확히 복원한 상태에서** 안전 정책이 무너진다는 뜻이다. slot k2/k3은 재구성 자체가
붕괴(0.65 → 0.53)하며 gated가 같이 떨어지므로, 과부하는 방법을 강화하지 않고 약화시킨다.

## 4. 엔진 재현성 (A_hf vs A_vllm)

같은 모델·같은 프롬프트·greedy 디코딩인데 엔진만 다른 두 실행의 gated ASR 차이는
모든 조건에서 ≤ 0.016(n4 0.6825 vs 0.6985, english 0.2567 vs 0.2501)이다. 커널·배치
구성 차이에서 오는 수준의 편차이며 결론을 바꾸지 않는다. 처리량은 vLLM이 압도적이다:
동일 shard 6,636 generations 기준 HF 8,211초 vs vLLM 202초(**약 40배**). 이후 대규모
실행은 vLLM을 기본으로 한다.

## 5. B_vllm 설정의 커버리지 결함 (중요)

config B(VL + 원본 이미지)는 **shard 0이 중간에 죽어 shard 1의 1,439개만** 집계돼 있다.
원인은 데이터셋의 0바이트 PNG 한 장이다.

```text
A2/public/imgs/Physical Harm/0126.png   (size 0, PIL UnidentifiedImageError)
```

`run_vl_gated_vllm.py`가 프롬프트 조립 중 이 파일을 열다 예외로 종료했고, 짝수 인덱스
아이템 전체가 유실됐다. 전체 2,879개 중 디코딩 불가 이미지는 이 1장뿐임을 전수 확인했다.

따라서 §2 표의 B 열은 **다른 열과 동일한 아이템 집합이 아니다.** 설정 내부의 paired 대비
(B의 n4 − B의 english = +0.3419, p ≈ 0)는 유효하지만, B 열과 A/C 열의 절대값 직접 비교는
아이템 구성 차이가 섞여 있으므로 잠정치로 취급해야 한다.

조치: `run_vl_gated_vllm.py`와 `run_vl_gated.py`가 실행 전에 이미지를 검증해, 디코딩
불가 이미지를 가진 아이템의 **모든 조건을 함께 제외**하고 그 목록을 `target_summary.json`의
`skipped_unreadable_image_items`에 남기도록 수정했다. shard가 통째로 죽는 일은 더는 없다.
전체 커버리지 재실행은 이 수정 위에서 다시 돌린다.

## 6. 이미지 유무의 효과 (B vs C)

- 원본 이미지가 있으면(B) interleaving gated ASR이 0.55~0.59, 회색 blank(C)면 0.42~0.43.
  즉 **이미지가 붙어 있을 때 텍스트 조립 공격이 더 잘 통한다**(단 §5의 커버리지 차이가
  섞여 있으므로 방향성만 취한다).
- slot 계열은 B/C 모두 0.16~0.29로 낮고, VL 모델에서는 slot 게임이 english baseline 대비
  이득이 거의 없거나(k2 p=0.075, k3 p=0.248) 음수다.
- 텍스트 전용 모델(A) 대비 VL 모델(B/C)의 interleaving 성능이 전반적으로 낮은 것은,
  같은 7B급이라도 VL 학습이 긴 텍스트 조립 과제에 불리하게 작용함을 시사한다.

## 7. 한계

1. B 설정 커버리지(§5). 재실행 전까지 B의 절대값은 잠정치다.
2. 안전 판정이 Qwen3Guard-Gen-8B 단일 judge에 의존한다. judge 편향 배제를 위한 독립
   judge 교차검증이 필요하다.
3. 타깃이 Qwen 계열 두 종뿐이다. 다른 계열 모델에서의 일반화는 미검증이다.
4. dev 전수 결과이므로, 논문 최종 수치는 프롬프트·judge·threshold 동결 후 별도 test
   split에서 재산출해야 한다.

## 8. 산출물

- 집계: `results/full_textdom_ALL_configs_summary.json`,
  `results/full_textdom_textmodel_hf_summary.json`,
  `results/full_textdom_textmodel_vllm_summary.json`,
  `results/full_textdom_vlreal_summary.json`,
  `results/full_textdom_vlblank_summary.json`
- 원시 생성·judge 원문·성공/실패 케이스: `private_artifacts/full_textdom/*` (0600 보존)
- 실행: `scripts/run_full_textdom_3configs.sh`, `scripts/run_vl_configs_vllm.sh`,
  `scripts/run_text_gated_vllm.py`, `scripts/run_vl_gated_vllm.py`,
  `scripts/compare_all_configs.py`
