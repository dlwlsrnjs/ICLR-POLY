# 비포화 willingness probe (OR-Bench-Hard × 프레임)

문제: 평범한 무해 probe는 helpful 모델에서 프레임 신호가 포화(비거부 다 1.0) → 의지축을 무해로 못 읽음.
해결: **OR-Bench-Hard**(무해하지만 유해해 보이는 1,319개; bench-llm/or-bench)에 willingness 프레임을 씌워
반응 민감도를 잰다. 프레임이 실제로 갈려 의지축 유효성이 무해로 드러난다.

## qwen7b (OR-Bench-Hard 200 × 프레임)
| frame | nonrefusal | hedge/ans | fiction_struct | words |
|---|---|---|---|---|
| plain | 0.84 | 2.40 | 0.06 | 235 |
| persona | 0.15 | 2.79 | 0.03 | 136 |
| fiction | 0.73 | 1.15 | 0.40 | 253 |
| pap | 0.89 | 2.25 | 0.04 | 234 |

- 포화 깨짐(비거부 0.15~0.89). fiction: 헤지 절반 + 구조↑ → 유효(유해 fiction 0.38과 일치).
- persona: 경계 내용에 오히려 안전장치 자극(비거부 폭락).
데이터: private_artifacts/alignment_probe/probe_orbench_hard.jsonl (OR-Bench-Hard-1k).
다음: 여러 모델(대형·정렬 포함)에 돌려 "무해 프레임 민감도 → 유해 comply" 상관을 재고 전이 feature로.
