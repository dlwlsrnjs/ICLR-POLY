# PolyJigsaw

PolyJigsaw는 하나의 요청을 4–6개의 의미 조각으로 나누고, 조각별 언어와 표시 순서를
조정했을 때 다국어 LLM의 안전 행동이 어떻게 변하는지 연구하는 재현 가능한
red-teaming 프로젝트입니다.

핵심 연구 질문은 **“안전 정렬된 모델이 cross-lingual compositional reasoning으로
전체 의도를 재구성하기 전에 그 위험을 탐지할 수 있는가?”**입니다. 개별 조각에는 전체
의도가 없지만, 다언어 decoding·순서 추론·의미 합성을 마친 뒤에만 위험한 전체 의미가
생기는 `compositional safety failure`를 측정합니다. 이는 단순 번역 공격이나 단일 언어
퍼즐 공격과 구별되는 연구 가설입니다.

현재 저장소에는 Lingua-SafetyBench의 Text-Dominant 항목으로 수행한 **고정 80개
파일럿**, 실행 코드, 집계 결과, 실험 프로토콜이 포함되어 있습니다. 원본 데이터와
유해 프롬프트·원시 모델 응답은 포함하지 않습니다.

2026-08-28에 추가한 **parallel-language interleaving puzzle**의 설계, GPT-4o mini
초기 결과, FLORES benign control, Hugging Face 이전 계획과 정확한 재개 명령은
[작업 인계 문서](docs/INTERLEAVING_PUZZLE_HANDOFF.md)에 정리했습니다.

## 현재 상태

- Lingua-SafetyBench 원본 100,440개 레코드의 로컬 구조·정렬 검사 완료
- 10개 언어, 8개 안전 시나리오 확인
- 의미적으로 정렬된 base item 10,044개 구성
- Text-Dominant, 영어 15단어 이상, 정규화 중복 제거 후 4,798개 구성
- dev 2,879 / validation 959 / test 960 고정 분할
- validation에서 8개 시나리오 × 10개 = 80개 고정 파일럿 수행
- Qwen2.5-7B-Instruct 320회 생성 완료
- Qwen3Guard-Gen-8B 공식 response-moderation 방식 320회 판정 완료, parse failure 0
- MD-Judge-v0.2-InternLM2-7B 판정은 보조 교차검증으로 유지
- 논문 기반 언어·카테고리 사전가설을 결과 확인 전에 고정
- 재실험의 원문·실제 입력·전체 응답·judge 원출력을 비공개 0600 산출물로 보존
- Qwen2.5-VL-7B 원본 text-only 80개×10언어=800개 baseline 완료
- 원본 이미지 포함 Qwen2.5-VL-7B dev 재검증 완료: raw ASR 상승과 reconstruction-gated 실패를 분리
- 카드 연결 게임 ablation 완료: English K0 gated ASR 48.75%, multilingual K1 17.50%

## 파일럿 설계

각 영어 원문을 동일한 5개 연속 조각으로 나눈 뒤 네 조건을 짝지어 비교했습니다.

| 조건 | 입력 |
|---|---|
| C0 | English direct |
| C1 | English 5-fragment shuffled |
| C2 | Multilingual fragments, canonical order |
| C3 | Multilingual fragments, shuffled order |

사용 언어는 Arabic, Chinese, English, Finnish, French, German, Japanese, Norwegian,
Russian, Spanish이며, 400개 조각에서 각 언어가 정확히 40회 등장하도록 균형화했습니다.

## 주요 결과

| 조건 | ASR | 95% Wilson CI | 평균 재구성 | 재구성 ≥0.8 |
|---|---:|---:|---:|---:|
| C0 English direct | 35.00% | 25.45–45.92% | 1.0000 | 80/80 |
| C1 English shuffled | 62.50% | 51.55–72.31% | 0.7857 | 47/80 |
| C2 Multilingual ordered | 67.50% | 56.64–76.76% | 0.4735 | 1/80 |
| C3 Multilingual shuffled | 70.00% | 59.23–78.94% | 0.4563 | 0/80 |

원시 출력을 보존한 최신 결정론적 재실행을 공식 Qwen3Guard 방식으로 판정했을 때
짝지은 C3−C0 차이는 +35.0%p였지만, 핵심 순서 효과인 C3−C2는 +2.5%p,
bootstrap 95% CI −7.5%p–+12.5%p로
확정되지 않았습니다. 또한 C3의 재구성 통과가
0/80이므로 현재 결과를 “정확한 의미 재구성 이후의 탈옥 성공”으로 해석할 수 없습니다.

현재 가장 타당한 결론은 다음과 같습니다.

> 다국어화·조각화·재구성 지시가 결합된 프롬프트는 Qwen2.5-7B의 안전 거부 행동을
> 약화할 수 있다. 다만 고정 분할기의 의미 복원 성능이 낮고 wrapper 효과가 크므로,
> 적응형 PolyJigsaw 메서드의 효과는 아직 검증되지 않았다.

원본 text-only baseline은 188/800=23.50%로, 논문의 Qwen2.5-VL 계열
Text-Dominant 55.24%를 그대로 재현하지 못했습니다. 다만 Non-HRL 24.37%가 HRL
20.00%보다 높은 방향은 논문과 같았습니다. 자세한 결과는
[원본 Text-only baseline 보고서](docs/LINGUA_TEXTONLY_BASELINE_REPORT.md)에 있습니다.

이후 원본 이미지를 실제 VL 입력에 포함하고, judge에서는 `[RECONSTRUCTED]` 인용을
제외한 `[ANSWER]`만 판정하도록 프로토콜을 바로잡았다. 이미지 포함 dev 80개에서
lossless-chain 후보의 raw answer ASR은 C0 23.75%에서 45.00%로 상승했지만,
`semantic reconstruction AND unsafe answer` gated ASR은 C0 23.75% 대비 15.00%로
낮았다. 따라서 현재 메서드는 의도한 compositional jailbreak에 성공하지 못했다.
자세한 수정 이유와 결과는 [멀티모달 gated dev 보고서](docs/VL_MULTIMODAL_GATED_DEV_REPORT.md)에
정리했다. 이전 text-only 42.50% 결과는 평가 오염이 확인되어 superseded 상태다.

후속 카드 연결 게임에서는 공격·유해성 표현 없이 START→END 규칙, payload 보존,
조립 후 수행만 지시했다. English K0는 reconstruction-gated ASR이 C0 23.75%에서
48.75%로 상승했지만, 비영어 한 조각을 넣은 K1은 17.50%였다. 게임 mechanics에는
효과가 있으나 현재 fragment 번역·정렬이 병목이라는 결론이다. 자세한 결과는
[VL 카드게임 보고서](docs/VL_GAME_DEV_REPORT.md)에 정리했다.

상세 분석, 실제 언어 배정·순서·응답 해시 사례, 시나리오별 결과와 후속 계획은
[파일럿 보고서](docs/PILOT_REPORT.md)와
[공식 방식 Qwen3Guard 재평가 보고서](docs/OFFICIAL_STYLE_QWENGUARD_REPORT.md)를
참고하세요. 기존 MD-Judge 결과는 보조 비교값으로만 유지합니다.

## 저장소 구성

```text
PolyJigsaw/
├── README.md
├── requirements.txt
├── docs/
│   ├── PILOT_REPORT.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── LINGUA_PRIOR_GUIDED_RERUN.md
│   ├── LINGUA_TEXTONLY_BASELINE_REPORT.md
│   ├── VL_ADAPTIVE_VALIDATION_REPORT.md
│   ├── VL_MULTIMODAL_GATED_DEV_REPORT.md
│   ├── VL_GAME_DEV_REPORT.md
│   ├── INTERLEAVING_PUZZLE_HANDOFF.md
│   ├── OFFICIAL_STYLE_QWENGUARD_REPORT.md
│   ├── PRIOR_GUIDED_FULL_80_REPORT.md
│   └── CASE_STUDIES.md
├── results/
│   ├── pilot_summary.json
│   ├── lingua_textonly_qwen25vl7b_800_summary.json
│   ├── vl_adaptive_qwen25vl7b_validation_summary.json
│   ├── vl_multimodal_chain_gated_dev_summary.json
│   ├── vl_multimodal_game_k0_gated_dev_summary.json
│   ├── vl_multimodal_game_gated_dev_summary.json
│   ├── interleaving_probe_summary.json
│   ├── qwen3guard_official_rejudge_summary.json
│   └── prior_guided_full_80_summary.json
└── scripts/
    ├── prepare_lingua_text.py
    ├── build_static_lingua_pilot.py
    ├── build_vl_adaptive_variants.py
    ├── run_openai_interleaving_probe.py
    ├── generate_static_qwen25vl_textonly.py
    ├── rejudge_qwen3guard_official.py
    ├── run_static_lingua_asr.py
    ├── polyjig_pilot.py
    └── pilot_factorial.py
```

## 데이터 준비

Lingua-SafetyBench는 공식 접근 절차와 이용 조건을 따라 별도로 받아야 합니다. 이
저장소는 데이터셋을 재배포하지 않습니다.

- [Lingua-SafetyBench 논문](https://arxiv.org/abs/2601.22737)
- [공식 저장소](https://github.com/zsxr15/Lingua-SafetyBench)

압축을 푼 원본 경로를 지정해 텍스트 정렬 파일을 생성합니다.

```bash
python scripts/prepare_lingua_text.py \
  --dataset-root /path/to/Lingua-SafetyBench/extracted/dataset
```

생성 파일에는 통제된 유해 질문이 포함되므로 기본적으로 외부에 공유하지 않고 접근
권한을 제한해야 합니다.

## 고정 파일럿 생성

NLLB-200-distilled-1.3B를 이용해 80개 고정 파일럿을 만듭니다. GPU는 명시적으로
할당받은 경우에만 `--device cuda`를 사용하세요.

```bash
python scripts/build_static_lingua_pilot.py \
  --input datasets/lingua_safetybench_text/lingua_polyjigsaw_textdominant_full.jsonl \
  --output datasets/lingua_safetybench_text/static_pilot_80.jsonl \
  --n 80 \
  --seed 20260827 \
  --device cpu
```

## 7B 평가

다음 명령은 대상 응답을 디스크에 저장하지 않고 MD-Judge 판정, 길이, SHA-256만
기록합니다.

```bash
python scripts/run_static_lingua_asr.py \
  --data datasets/lingua_safetybench_text/static_pilot_80.jsonl \
  --outdir polyjig_outputs/lingua_static_80 \
  --target Qwen/Qwen2.5-7B-Instruct \
  --judge OpenSafetyLab/MD-Judge-v0_2-internlm2_7b \
  --target-device cpu \
  --judge-device cpu
```

할당된 GPU 하나를 사용할 경우 `CUDA_VISIBLE_DEVICES=<allocated-id>`를 설정하고 두
device 인자를 `cuda:0`으로 바꿉니다. 공유 서버에서 비어 있지 않은 GPU를 임의로
사용하지 마세요.

## 다음 단계

1. 동일 wrapper 대조군과 English ordered 조건을 추가해 형식 효과를 분리합니다.
2. 균등 단어 분할을 4–6개 semantic-role span fragmenter로 교체합니다.
3. 원 데이터셋의 공식 10개 언어 전체 번역을 fragment alignment anchor로 사용합니다.
4. reconstruction <0.8인 episode에는 ASR 보상을 주지 않는 constrained controller를
   학습합니다.
5. `multilingual shuffled − multilingual ordered`를 primary estimand로 사용합니다.
6. dev에서 개발하고 validation에서 선택한 뒤, test 정책을 동결해 복수 7B 모델과
   독립 판정기로 재검증합니다.

구체적인 B0–B7 대조군, reward, gate와 성공 기준은
[실험 프로토콜](docs/EXPERIMENT_PROTOCOL.md)과
[파일럿 보고서](docs/PILOT_REPORT.md)에 정리되어 있습니다.

전체 파이프라인을 함수 단위로 이해하려면
[메서드·코드 구현 가이드](docs/IMPLEMENTATION_GUIDE.md)를 참고하세요. 데이터 정렬,
조각 경계 선택, 언어 균형, NLLB 번역, Qwen 생성, MD-Judge 호환 처리와 paired
bootstrap 구현뿐 아니라 관련 연구 대비 위치, 핵심 수식, B0–B7 factorial 분해와
현재 고정 baseline/최종 adaptive method의 차이를 설명합니다.

기존 파일럿에서 실제 safe→unsafe 전환이 있었던 8개 사례의 fragment 길이, 언어 배정,
순서 이동, 네 조건의 판정·응답 길이·해시는
[사례 분석](docs/CASE_STUDIES.md)에 정리했습니다.

C0–C3 대상 입력, Qwen chat wrapper, NLLB 번역 호출, MD-Judge 평가 지시, 무해
mechanics/factorial prompt까지 기존 실행에서 사용한 전체 프롬프트 계층은
[전체 프롬프트 명세](docs/ALL_PROMPTS.md)에 정리했습니다. 통제된 유해 원문과 원시
응답만 자리표시자로 마스킹하고 지시문과 변수 연결 방식은 그대로 보존했습니다.

ASR의 정확한 판정식, raw ASR과 reconstruction-gated ASR의 차이, 기존 실행에서 남아
있는 실제 입력·판정 파일과 보존되지 않은 원시 출력의 범위는
[ASR 및 입출력 보존 명세](docs/ASR_AND_ARTIFACTS.md)에 정리했습니다.

원시 출력 감사 가능성을 보완하기 위해 대표 3개 item, 12개 generation을 별도
재실행해 접근 제한 파일에 raw output과 judge 판정문을 보존했습니다. 기존 hash와
일치한 출력 및 재실행 변동성은 [재실행 사례 요약](docs/RERUN_EXAMPLES.md)에
정리했습니다. 이 재실행은 기존 80개 통계에 합치지 않습니다.

10분 내 방향성 판단을 위한 조건별 자동 judge ASR, 재구성 gate, 카테고리별 결과는
[3개 카테고리 긴급 보고서](docs/QUICK_3CATEGORY_REPORT.md)에 별도로 정리했습니다.

## 안전 및 데이터 취급

- 본 프로젝트는 승인된 학술적 AI 안전 평가 목적으로만 사용해야 합니다.
- 원본 데이터셋의 접근·재배포 조건을 준수해야 합니다.
- 유해 프롬프트, 원시 모델 응답, 모델 캐시, 토큰 및 로컬 경로를 커밋하지 않습니다.
- 공개 결과에는 집계값과 위해 세부가 제거된 구조 예시만 포함합니다.
- 실행 가능한 위해 절차를 생성·배포하는 용도로 사용해서는 안 됩니다.

## 재현성 주의사항

이번 결과는 단일 대상 모델, 단일 자동 판정기, validation 80개로 얻은 파일럿입니다.
ASR 상승만으로 메서드 성공이나 모델 일반 취약성을 주장하지 않습니다. 특히 낮은
재구성 성능과 prompt wrapper confound를 해결하기 전까지는 탐색적 결과로 취급해야
합니다.
