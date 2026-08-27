# Lingua-SafetyBench 원본 Text-only Baseline

## 질문

PolyJigsaw 변형을 전혀 적용하지 않고 Lingua-SafetyBench 원본 질문만 넣었을 때,
Qwen2.5-VL baseline이 논문의 Text-Dominant 성능과 같은 결과를 보이는가?

## 실험 설계

- 모델: Qwen2.5-VL-7B-Instruct
- 데이터: validation 고정 원문 80개, 카테고리별 10개
- 언어: 공식 번역 10개 언어
- 전체 생성: 80 × 10 = 800개
- 이미지: 제거
- PolyJigsaw wrapper·fragment·shuffle: 모두 제거
- 생성: greedy, temperature 0, max new tokens 256
- 판정: Qwen3Guard-Gen-8B 공식 response moderation
- ASR: `Unsafe` 또는 `Controversial`
- 판정 parse failure: 0/800

논문의 Qwen2.5-VL Text-Dominant 수치는 안전한 관련 이미지와 텍스트를 함께 입력한
3B·7B 계열 평균이다. 이번 실험은 7B 하나만 사용하고 이미지를 제거했으므로 절대값의
엄밀한 재현이 아니라 image ablation에 가깝다.

## 핵심 결과

| 설정 | ASR |
|---|---:|
| 논문 Qwen2.5-VL 계열 Text-Dominant | 55.24% |
| 이번 Qwen2.5-VL-7B Text-only | 188/800 = 23.50% |
| 절대 차이 | −31.74%p |

Text-only ASR의 Wilson 95% CI는 20.69–26.56%다. 논문 수치의 42.54% 수준만
남았으므로 절대 성능은 재현되지 않았다. 안전한 관련 이미지도 모델이 질문의 맥락과
의도를 더 잘 이해하게 만들어 ASR을 올리는 competence-induced 효과가 있을 가능성과,
3B/7B 평균·표본·judge 차이가 함께 작용했을 가능성이 있다.

## 언어별 결과

| 언어 | strict ASR |
|---|---:|
| Finnish | 40.00% |
| Norwegian | 30.00% |
| English | 26.25% |
| Russian | 25.00% |
| Arabic | 23.75% |
| German | 21.25% |
| French | 18.75% |
| Spanish | 18.75% |
| Japanese | 17.50% |
| Chinese | 13.75% |

논문의 Finnish 고위험 경향은 재현됐다. 반면 Japanese는 높지 않았고 Norwegian은
낮지 않아 세부 언어 순위는 재현되지 않았다.

Qwen 기준 HRL인 English·Chinese는 32/160=20.00%, 나머지 Non-HRL은
156/640=24.37%다. Non-HRL−HRL은 +4.37%p로 논문의 +6.99%p와 방향은 같다.
다만 item 단위 paired bootstrap 95% CI는 −2.66~+11.25%p로 0을 포함하므로,
80개 원문만으로 언어 자원 격차를 확정할 수는 없다.

## 카테고리별 결과

| 카테고리 | strict ASR |
|---|---:|
| Economic Harm | 40.00% |
| Malware Generation | 35.00% |
| Illegal Activity | 33.00% |
| Fraud | 29.00% |
| Physical Harm | 23.00% |
| Privacy Violence | 17.00% |
| Sex | 6.00% |
| Hate Speech | 5.00% |

Malware Generation과 Illegal Activity가 높은 경향, Sex가 낮은 경향은 논문과 맞는다.
하지만 논문에서 상위권인 Physical Harm은 이번에 23%였고 Economic Harm이 가장 높아
전체 카테고리 순위는 재현되지 않았다. 카테고리당 base item이 10개뿐이라는 점도
큰 불확실성이다.

## 판단

“논문 성능이 text-only에서도 그대로 나온다”는 가설은 기각된다. 절대 ASR은 크게
낮아졌다. 다만 Non-HRL > HRL, Finnish의 높은 위험, Malware/Illegal Activity의 높은
위험처럼 일부 방향성은 남았다.

정확한 원인 분리를 위해서는 동일 80개에 대해 다음 두 조건을 paired로 추가해야 한다.

1. 원본 Text + 원본 safe relevant image
2. 원본 Text only

같은 Qwen2.5-VL-7B와 같은 Qwen3Guard 판정을 사용해 두 조건의 item-level 차이를
계산하면 이미지 제거 효과를 직접 추정할 수 있다.
