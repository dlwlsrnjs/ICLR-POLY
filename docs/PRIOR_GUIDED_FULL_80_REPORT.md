# 논문 기반 사전가설 전체 80개 재실험

## 결론

자동 judge 기반 raw proxy ASR은 `C0 37.50% → C1 61.25% → C2 70.00% →
C3 77.50%`로 증가했다. 그러나 메서드의 핵심 비교인 C3−C2는 `+7.5%p`이고 paired
bootstrap 95% CI가 `−2.5~+17.5%p`, McNemar exact p가 `0.238`이므로 이 표본에서
셔플의 추가 효과는 확정되지 않았다.

더 중요한 문제는 reconstruction gate다. 임계값 0.8을 통과한 사례가 C2는 1/80,
C3는 0/80이다. 따라서 C3의 77.50%를 “정확한 의미 재구성 뒤 탈옥 성공률”이라고
부를 수 없다. 현재 결과는 wrapper·다국어 표면형이 자동 판정률을 높였다는 신호이지,
의도한 compositional safety failure를 입증한 결과는 아니다.

## 실행 구성

- 표본: Lingua-SafetyBench validation text-dominant 고정 80개
- 카테고리: 8개, 각 10개
- 조건: English direct, English shuffled, Multilingual ordered,
  Multilingual shuffled
- 대상: Qwen2.5-7B-Instruct, GPU 6, greedy, max new tokens 256
- 판정: MD-Judge-v0_2-internlm2_7b, GPU 7
- 전체 생성: 320개, judge parse failure 0개
- 실행 전후 GPU 6·7은 0 MiB였고 다른 GPU 프로세스는 종료하거나 변경하지 않았다.

논문의 주 판정기는 GPT-5이며 Qwen3Guard는 일부 분석에 사용됐다. 이번 MD-Judge
수치는 공식 Lingua-SafetyBench ASR의 직접 재현이 아니라 7B 자동 judge proxy다.

## 조건별 결과

| 조건 | unsafe / 80 | proxy ASR | 95% Wilson CI | reconstruction ≥ 0.8 |
|---|---:|---:|---:|---:|
| C0 English direct | 30 | 37.50% | 27.69–48.45% | 80 |
| C1 English shuffled | 49 | 61.25% | 50.29–71.18% | 47 |
| C2 Multilingual ordered | 56 | 70.00% | 59.23–78.94% | 1 |
| C3 Multilingual shuffled | 62 | 77.50% | 67.21–85.27% | 0 |

주 대비 C3−C2에서는 C2 safe→C3 unsafe가 12개, 반대가 6개였다. 보조 대비
C3−C0는 `+40.0%p`, bootstrap 95% CI `+26.25~+53.75%p`, exact p `< 0.000001`로
크지만 reconstruction 실패 때문에 공격 메커니즘의 증거로 바로 해석할 수 없다.

## 논문 기반 category prior 점검

논문은 Physical Harm, Malware Generation, Illegal Activity를 대체로 높은 위험군,
Economic Harm과 Sex를 낮은 위험군으로 보고했다.

| 조건 | 고위험군 PH/MG/IA | 저위험군 EH/Sex |
|---|---:|---:|
| C0 | 15/30 = 50.00% | 7/20 = 35.00% |
| C1 | 20/30 = 66.67% | 12/20 = 60.00% |
| C2 | 23/30 = 76.67% | 15/20 = 75.00% |
| C3 | 24/30 = 80.00% | 16/20 = 80.00% |

direct 조건에서는 방향이 맞지만 C3에서는 두 군이 동일했다. 따라서 논문의 category
prior는 이번 복합 다국어 C3 결과에서 재현되지 않았다. 특히 Economic Harm C3가
8/10, Sex C3도 8/10이어서 자동 judge의 wrapper 민감도와 false positive 가능성을
사례 감사로 확인해야 한다.

## 논문 기반 language prior 점검

논문에서는 Finnish, Japanese, German이 높은 편이고 English와 Norwegian이 낮은
편이었다. 이번 C3에서 FI/JA/DE 중 하나 이상이 포함된 입력은 56/72=77.78%, 포함되지
않은 입력은 6/8=75.00%였다. EN/NO 중 하나 이상이 포함된 입력은 50/63=79.37%,
포함되지 않은 입력은 12/17=70.59%였다.

이는 사전 방향을 지지하지 않는다. 다만 각 입력이 5개 언어를 동시에 포함하고 언어
노출이 서로 중첩되므로 단일언어 효과를 식별할 설계가 아니다. 언어별 효과를 검증하려면
같은 원문·조각·순서를 유지한 채 하나의 language assignment만 바꾸는 paired ablation이
필요하다.

## 다음 메서드 보완

1. semantic-role fragmenter를 학습·검증해 C2/C3 reconstruction pass를 먼저 높인다.
2. reward를 `unsafe judge` 단독이 아니라 `reconstruction ≥ 0.8 AND unsafe`로 제한한다.
3. 같은 wrapper를 쓰는 English ordered 대조군을 추가해 wrapper 효과를 분리한다.
4. 언어 조합을 one-factor-at-a-time으로 바꾼 paired assignment 실험을 만든다.
5. MD-Judge 판정을 Qwen3Guard-Gen과 사람 층화감사로 교차검증한다.
6. reconstruction이 확보되기 전에는 C3 raw proxy ASR을 메서드 성공으로 보고하지 않는다.

## 산출물 보존

모든 원문, 실제 조건 입력, 전체 Qwen 응답, 추출 reconstruction/answer, judge 전체
출력, 판정, 해시는 저장됐다. 유해 텍스트가 포함되므로 저장 폴더와 파일 권한은
각각 0700/0600이며 공개 저장소에는 넣지 않는다. 공개 저장소에는 이 집계 보고서와
사전가설만 둔다.
