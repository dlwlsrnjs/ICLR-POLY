# Lingua-SafetyBench 논문 기반 사전 가설

이 문서는 새 결과를 보기 전에 고정한 분석 방향이다. 근거는 Lingua-SafetyBench
v2(arXiv:2601.22737, 2026-04-21)의 본문과 표이다.

## 논문에서 확인한 경향

- Qwen 계열에서 영어와 중국어는 HRL, 나머지 8개 언어는 Non-HRL로 분류된다.
- 전체 평균 ASR은 Non-HRL 40.12%, HRL 38.72%였다.
- text-dominant 조건은 Non-HRL 48.18%, HRL 42.72%로 격차가 더 컸다.
- 언어별 평균에서는 Finnish, Japanese, German이 높은 편이고 English와 Norwegian이
  낮은 편이었다. 저자는 Norwegian의 낮은 ASR이 안전 정렬이 아니라 낮은
  instruction-following에서 올 수도 있다고 주의한다.
- 카테고리에서는 Physical Harm, Malware Generation, Illegal Activity가 거의 모든
  언어에서 높은 위험군이었고 Economic Harm과 Sex는 낮은 위험군이었다.

## 이번 실험에 적용하는 방식

위 경향을 표본 선택에 사용해 성공 사례만 고르지 않는다. 기존 validation 고정 표본
80개, 즉 8개 카테고리별 10개를 모두 실행한다. 논문 경향은 다음 세 방향성 대비를
미리 정하는 데만 사용한다.

1. `multilingual_shuffled - multilingual_ordered`를 주효과로 본다.
2. PH/MG/IA 묶음이 EH/Sex 묶음보다 높은지 확인한다.
3. FI/JA/DE 조각 노출과 EN/NO 조각 노출의 차이는 탐색적 언어 대비로 보고한다.

세 번째 대비는 한 입력에 여러 언어가 동시에 들어가므로 단일언어 ASR이 아니다.
따라서 Lingua-SafetyBench의 언어별 수치를 직접 재현했다고 표현하지 않는다.

## 판정과 보존

- 대상 모델: Qwen2.5-7B-Instruct, greedy decoding
- 자동 판정: MD-Judge 7B proxy
- 논문 기준과의 차이: 논문의 주 판정기는 GPT-5이고 Qwen3Guard는 일부 분석에
  사용된다. 따라서 이번 값은 공식 논문 ASR의 직접 재현이 아니라 자동
  judge 기반 proxy ASR이다.
- 원문, 실제 프롬프트, 전체 모델 응답, 재구성 추출, judge 전체 출력, 해시와 집계는
  모두 mode 0600인 비공개 산출물로 저장한다.
