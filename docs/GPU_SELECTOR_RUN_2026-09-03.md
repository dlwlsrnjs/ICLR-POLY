# GPU에서 실행하는 다중 모델 선택기

> 2026-09-03 목적 정정 후 **중지됨**: 이 실행은 무해 재구성 전용이며 ASR 공동 목적을 학습하지 않는다. 기존 관측은 보존했다. `paused_objective_review` 상태와 `docs/JOINT_EVALUATION_OBJECTIVE_2026-09-03.md`를 먼저 확인한다. 아래 단계 표와 재개 명령은 이전 설계 기록이다.

사용자 요청: 가능한 한 많은 모델·설정 관측을 확보하고 GPU에서 강화학습을 실행한다.

## 규모와 실제 상태

기존 캐시 12개에 공개 체크포인트 18개를 추가 확보했다. 추가 가중치 다운로드는
약 153GB이며 `private_artifacts/selector_gpu_20260903/downloads.json`에 고정 revision,
로컬 경로와 완료 여부가 있다. 모델 파일 확보와 모델 실행 성공은 별도 상태다.

| 단계 | 대상 | 원문 | 구성 | 계획된 고유 대상 호출 |
|---|---:|---:|---:|---:|
| 로컬 모델 첫 GPU 학습 | 12 | 128 | 64 | 98,304 |
| 추가 모델 포함 GPU 학습 | 30 | 128 | 64 | 245,760 |
| 전체 원문으로 확장 | 30 | 997 | 64 | 1,914,240 |

각 단계는 앞 단계의 관측을 재사용한다. 세 숫자를 합쳐 새로운 관측 수로 보고하지
않는다. 실패 후 실제로 다시 생성한 호출이 있다면 비용에는 따로 반영해야 한다.
처음 두 모델의 1,024개 처리량 측정도 같은 관측 저장소에 보존해 재사용했다.

현재 진행 상태의 기준은 `expanded_pipeline_status.json`, 대상별 `progress.json`,
훈련 결과 디렉터리의 `progress.json`이다. 이 문서의 단계 표는 완료 보고가 아니다.

## 대상과 데이터

- 기존 12개: Qwen2.5 1.5/3/7/14/32B, Qwen3-8B, Phi-3.5-mini, Mistral-7B-v0.3,
  Zephyr-7B-beta, OLMo2-7B, InternLM2.5-7B, Falcon3-7B.
- 추가 18개: Qwen2.5-0.5B, Qwen3 0.6/1.7/4/14B, SmolLM2 135M/360M/1.7B,
  SmolLM3-3B, Granite3.3 2/8B, OLMo2-1B, StableLM2-Zephyr-1.6B,
  Yi1.5 6/9B, DeepSeek-V2-Lite-Chat, Falcon3 1/3B.
- 보수적인 분할 그룹은 Qwen, Phi, Mistral/Zephyr, OLMo, InternLM, Falcon,
  SmolLM, Granite, StableLM, Yi, DeepSeek의 11개다. 이름이나 크기만 달리해 독립
  계열 수를 늘리지 않는다. 모든 대상이 학습에 들어가는 것은 아니며 각 fold에서
  한 계열은 검증, 다른 한 계열은 최종 평가로 제외한다.
- 로컬 FLORES 파일의 997문항 모두 10개 언어가 채워져 있다. 원문 정규화 해시로
  기존 5모델 파일럿과 비교한 중복은 0개였다. 이는 과거 프로젝트 전체에서 미사용인
  원문이라는 뜻은 아니며, 후속 panel_v2의 모든 원시 자료는 접근 점검하지 못했다.
- 전체 원문 분할은 train 598 / validation 149 / test 250. 첫 단계는 이 분할 안에서
  train 64 / validation 32 / test 32를 사전 선택한다.
- 64개 구성은 언어 조합, 언어 수 2~10, 언어별 조각 수 3/5/8/12,
  interleaved/shuffled/reversed/blocks 배열을 조합한다. 모든 63,808개 문항·구성
  입력의 해시가 서로 달랐으며, 모델별로 동일한 입력을 사용한다.

추가 대상은 각 공식 배포 저장소의 파일 목록·revision과 접근 여부를 확인했다.
예: [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct),
[Granite](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct),
[Yi](https://huggingface.co/01-ai/Yi-1.5-6B-Chat),
[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat).
DeepSeek-LLM-7B 후보는 이번 safetensors 수집 방식에 맞지 않아 실제 다운로드 전에
DeepSeek-V2-Lite-Chat으로 교체했다. 다운로드한 모델의 라이선스·모델 카드도 캐시에 보존한다.

## 학습과 평가

대상 LLM은 고정된 환경이다. 학습 대상은 구조 특징과 관측 이력으로 다음 평가 구성,
마지막 추천 구성을 선택하는 공유 인코더 정책이다. 구성 ID 전용 출력층 대신
후보별 특징을 같은 인코더로 처리하고, 관측된 후보의 정보를 모아 대상 특징으로 쓴다.

CUDA에서 지도학습으로 초기화하고 PPO를 적용한다. 학습 episode마다 대상 계열을
균등하게 고른 후 해당 계열의 체크포인트를 선택한다. 같은 학습 원문 집합에서
8문항을 calibration으로, 다른 32문항을 마지막 추천의 보상 계산에 사용한다.
보류 모델 및 최종 평가 문항의 결과는 업데이트에 쓰지 않는다.

첫 단계는 fold당 지도학습 500 update + PPO 300 update × 4 epoch, seed 7로 실행한다.
전체 단계는 지도학습 1,000 update + PPO 1,000 update × 4 epoch,
seed 7/17/27, 모든 계열을 보류하는 fold로 실행한다. batch size는 128이다.
학습 episode와 PPO epoch 재사용 횟수를 고유 관측 수로 세지 않는다.

비교군은 공통 고정 설정, 무작위 탐색, 고정 구조 커널 GP-UCB, 한 번의 지도학습
선택, 순차 지도학습 선택, 지도학습 추가 학습, PPO, 추론 중 피드백 제거다.
추가 지도학습은 PPO의 optimizer update 수를 맞추며, 상태 예제 수와 GPU 시간까지
같은 비교군은 아니다. 해당 차이를 config와 곡선에 기록한다.

재구성 보상은 `0.7 × token-F1 + 0.3 × 정규화 문자 시퀀스 유사도`다. **무해 재구성의
어휘 일치도 대리 지표이며, 의미 동등성이나 reconstruction-gated ASR이 아니다.**
이 단계는 선택기·전이 학습 기반을 구축하는 실험이다. 논문의 안전성 목적에서 좋은
설정을 고른다는 주장은 별도 목표/평가 연결 없이는 하지 않는다.

## 코드와 재개

- 데이터: `scripts/build_selector_gpu_data.py`
- 대상 실행: `scripts/collect_selector_gpu.py`, `scripts/run_selector_gpu_queue.py`
- 추가 모델 확보: `scripts/download_selector_targets.py`
- CUDA 지도학습/PPO: `scripts/train_selector_gpu.py`
- 전체 단계 실행: `scripts/run_selector_gpu_expanded_pipeline.py`
- 완료 결과 집계: `scripts/report_selector_gpu.py`
- 원문/입력/출력/진행 로그: `private_artifacts/selector_gpu_20260903/`
- checkpoint/학습 곡선/검증/최종 평가: `results/selector_gpu_20260903/`

출력 chunk는 임시 파일 작성 후 교체하므로 완료 파일만 재사용한다. 재개 시 구성과
프롬프트 해시를 검사한다. 수집된 데이터의 누락을 실패 점수나 예측값으로 채우지
않고, 요구한 대상 수와 문항·구성의 완전한 관측표를 확인한 뒤 학습한다.

```bash
/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python \
  scripts/run_selector_gpu_expanded_pipeline.py \
  --run-root private_artifacts/selector_gpu_20260903 \
  --results results/selector_gpu_20260903 \
  --start-stage collect128_12
```

이미 실행 중인 supervisor가 있으면 같은 명령을 중복 시작하지 않는다. 상태 파일의
PID/단계를 확인한 뒤 중단된 단계에서만 재개한다. 활성 프로세스와 실행 로그를 근거로
진행을 판단하며 파일이나 폴더가 존재한다는 이유만으로 완료를 주장하지 않는다.

## 점검 및 수정

- 정책에서 아직 평가하지 않은 점수를 바꿔도 출력이 달라지지 않는지 확인했다.
- 보류 모델·test 문항의 값을 바꿔도 지도학습/PPO 결과가 동일한지 확인했다.
- PPO가 초기 모델을 훼손하지 않고 별도 정책의 가중치를 갱신하는지 확인했다.
- 후보 배열 순서를 바꾸면 예측도 같은 순열을 따르는지 확인했다.
- OLMo의 실제 문맥 한도 4,096에 고정 8,192를 요청해 시작 실패했다. 모델별 원래
  한도와 8,192 중 작은 값을 사용하도록 고쳤다. 강제 문맥 확장으로 우회하지 않는다.

검사: `tests/test_selector_gpu.py` 네 가지 통과. 추가 모델의 tokenizer/config
사전 검사는 `additional_model_preflight.json`에 기록한다. 추론 성공은 실제 수집
로그에서 별도로 확인한다.
