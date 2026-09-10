# PolyJigsaw 연구 작업 기록

## 2026-09-03 — 검토에서 선택기 구현으로

### 사용자 결정

- 기존 연구 전체의 맥락과 최적화 쟁점을 검토한다.
- 연구 관련 통상 작업은 별도 확인을 반복하지 않고 진행한다.
- 언어 수/순서의 작은 격자를 전체 탐색 공간으로 간주하지 않는다.
- 학습 기반 적응형 선택기를 주된 연구 후보로 삼고 순차적 RL의 이점을 검증한다.
- 기존 여러 모델의 결과를 학습에 재사용한다.
- 코드를 구현하고 실험을 실행하며 대화 기록의 보존 상태를 확인한다.

### 근거와 설계

- 기존 BO의 질의 계수 및 관측 보존 오류를 확인했다. 기존 BO 결과는 새 실험 결과로 재인용하지 않는다.
- 기존 5개 모델 패널의 출력과 판정 로그 16,820행 연결을 점검했다.
- 이번 첫 구현은 무해한 FLORES 재구성 결과 5모델 × 40문항 × 10조건을 사용한다.
- 모든 400개 문항/조건 키에서 대상 간 프롬프트 차이가 있으므로 기존 실행 시스템 간 예비 비교로 해석한다.
- 5개 leave-one-target-out fold와 원문 기준 20/8/12 분할을 사용한다.
- 대상 특징은 비용을 계산한 초기 2조건의 관측으로 얻는다. 최종 test 정답은 정책의 입력에 들어가지 않는다.
- 지도학습 초기화 후 terminal-reward actor-critic을 학습한다. 사전 지정한 평가 예산을 사용하며 이번 버전에는 학습된 종료 규칙이 없다.
- 추론 시 추가 피드백 제거 대조군은 별도 재학습 모델이 아닌 개입 실험이다.
- 새 대상 LLM/API/GPU 실행은 없으며, 이번 결과는 기존 관측표의 CPU replay 실험이다.

### 구현

- scripts/benign_selector.py: 무해 자료 검증/로딩, 재생 환경, 지도학습 및 actor-critic, 비교군.
- scripts/run_benign_selector_experiment.py: 모델/문항 분할, 반복 학습, 독립 평가, checkpoint/trace/보고서 저장.
- tests/test_benign_selector.py: 판정 타입, 비용, 관측 불변성, test 누출 및 학습 파라미터 갱신 검증.
- 결과 경로: results/benign_selector_pilot_20260903/.
- 검증: unittest 10개 통과.

### 대화 기록

- Codex 작업 기록에서 이 대화의 이전 사용자 메시지와 공개 assistant 답변을 실제 조회했다.
- docs/conversations/2026-09-03_visible_messages.md 및 .json에 조회 시점의 공개 메시지를 보존한다.
- 이 파일은 대화 스냅샷이다. 현재 실행 이후의 미래 메시지를 자동 저장하는 백그라운드 작업은 설정하지 않았다.
- 도구 호출/출력, 시스템 지침 및 내부 추론은 공개 대화 파일에 포함하지 않는다.
- 요약/판단은 이 문서와 ADAPTIVE_SELECTOR_RESEARCH_PLAN.md, 실제 측정은 실험별 manifest/config/summary/episodes 파일을 기준으로 한다.

### 실험 결과

완료: 15회 학습, 33,600개 비교 기록, CPU 62.44초.

추가 설정 예산 4개에서 무해 재구성률은 고정 선택 91.67%, RL 90.56%, 추가 피드백 제거 90.90%다. 이번 조건에서는 RL 우위 및 순차 피드백의 이점이 입증되지 않았다.

상세 결과: results/benign_selector_pilot_20260903/FINDINGS_KO.md 및 REPORT.md.
전 수치/불확실성: summary.json. 원문 분할/입력 해시: manifest.json. 실험 설정: config.json. 15개 checkpoint와 전체 episodes.jsonl을 보존한다.

새 API 호출이나 GPU 모델 생성은 없었다. 기록된 10개 무해한 조건에 대한 학습/재생 검증만 완료했고, 확장 공간과 외부 안전성 평가에서의 성능은 아직 검증하지 않았다.

## 2026-09-03 — 모델 수와 학습 규모에 대한 사용자 정정

- 사용자는 여러 **대상 모델**에서 자료를 모아 선택기를 학습하려는 것이며 10개 수준의 규모가 작다고 지적했다.
- 기존 10개는 모델이 아니라 설정 수였다. 파일럿의 고유 관측은 5모델 × 40문항 × 10설정 = 2,000개이며, 반복 학습/평가 episode가 새 관측을 늘리지는 않는다.
- 두 H100 80GB의 사용 메모리 1MiB, 사용률 0%를 다시 확인했다. 기존 CPU 실행은 GPU 부족 때문이 아니었다.
- 로컬 캐시에서 범용 텍스트 대상 12체크포인트의 가중치 shard와 tokenizer 관련 파일 존재를 확인했다. 보수적으로 묶으면 6계열 그룹이고 Qwen이 6개다. 실행 호환성과 checksum 검증은 이번 목록 검사 범위가 아니다.
- 원문 FLORES dev Parquet 메타데이터는 997행, 10열이다. 이번 vLLM 환경에는 pyarrow가 없어 설치 없이 시스템 Python의 pyarrow로 확인했다.
- `scripts/inventory_selector_targets.py`와 생성한 `docs/SELECTOR_TARGET_INVENTORY_2026-09-03.json`에 확인 근거를 보존했다. 대상 ID 중복 및 집계 일치 검증 완료.
- 총 약 30대상 / 10개 이상 계열을 **수집 설계 목표**로 두고, 원문 수·실제 구성 수도 확대한다. 목표 규모의 충분성은 새 계열의 검증 학습곡선으로 확인한다. 구체적 계획: `docs/MULTIMODEL_SELECTOR_SCALE_PLAN.md`.
- 이번 정정 작업은 목록·코드·설계 기록까지 완료했다. 새 모델 확보, 확장 관측 수집 및 GPU 학습을 실행한 것은 아니다. 기존 CPU 파일럿을 본 실험으로 대체 보고하지 않는다.

## 2026-09-03 — 대규모 자료 수집과 실제 GPU 강화학습 실행

사용자 원문: “최대한 많이 뽑아 강화학습 충분히 할 수 있게 그렇게 하고 학습돌려봐 GPU로”

- 두 H100 80GB에서 실제 대상 LLM 관측을 수집하기 시작했다. 초기 처리량 검사 1,024개씩도 저장해 재사용한다.
- 로컬 모델 12개에 공개 모델 18개(약 153GB)를 추가 다운로드했다. 총 30개 체크포인트, 보수적 11계열이며 실제 revision과 파일 경로를 확장 inventory에 기록했다.
- FLORES 원문 997개, 실제 구성 64개로 문항·구성 입력 63,808개를 생성했다. 기존 5모델 파일럿과 정규화 원문 해시 중복은 0개였으나 과거 모든 프로젝트 사용 이력과의 독립성까지 확인한 것은 아니다.
- 먼저 128문항(64/32/32 split)으로 12모델의 GPU 학습을 검증한다. 추가 모델을 포함한 30모델 단계와 전체 997문항 단계까지 자동 연결했다. 전체 계획 관측 수는 1,914,240개이며 아직 완료 수치가 아니다.
- CUDA 공유 후보 인코더, 지도학습 초기화, PPO, 계열 보류 평가, 세 seed, 비교군과 결과 자동 집계를 구현했다.
- 미래 관측 차단, 보류 모델/test 변경에 대한 학습 불변성, PPO 실제 갱신, 후보 순열 등변성 검사 4개가 통과했다. GP-UCB의 미관측 값 차단과 비용 48호출 계산도 확인했다.
- 모든 추가 모델의 config/tokenizer를 로컬에서 읽었고 vLLM registry에 아키텍처가 등록돼 있음을 확인했다. 문맥 한도 4,096인 OLMo/StableLM/Yi는 전체 63,808개 입력 길이 검사에서 초과가 없었다.
- OLMo 시작 오류는 고정 8,192 문맥 요청 때문이었다. 모델별 한도를 사용하도록 수정했다.
- InternLM 오류는 SentencePiece 0.2.2가 기존 vocabulary의 null 문자를 거부하기 때문이었다. 공유 환경을 바꾸지 않고 InternLM 전용 경로에 SentencePiece 0.2.0을 설치했으며 slow tokenizer로 정상 토큰화를 확인했다.
- 실험 목표는 현재 **무해 재구성 어휘 일치도 대리 보상**이다. 의미 보존 판정 또는 reconstruction-gated ASR 학습이 완료됐다는 뜻은 아니며, 논문 안전성 목적의 연결은 별도 검증이 필요하다.
- 상세 설계·실행·재개: docs/GPU_SELECTOR_RUN_2026-09-03.md. 실제 진행: private_artifacts/selector_gpu_20260903/expanded_pipeline_status.json. 학습 결과: results/selector_gpu_20260903/.

## 2026-09-03 15:04 KST — 중단된 확장 GPU 실험 재개

- 사용자 요청: “이전에 하던 작업이 있어 이어서 진행해줘”. 최신 연구 기록과 실제 프로세스/산출물을 대조해 다중 모델 선택기 실험을 이어갔다.
- 재개 전 두 H100은 유휴였고, 상태 파일에 남은 supervisor/worker PID는 존재하지 않았다. 로그에는 마지막 정상 배치 이후 종료 원인이 기록돼 있지 않아 원인을 확정하지 않았다.
- 첫 단계 `train128_12`는 12대상 × 128문항 × 64구성, 6개 계열 보류 fold, seed 7로 완료돼 있었다. 추가 구성 예산 4에서 PPO와 고정 설정의 계열 평균 재구성 대리 점수는 모두 약 0.8767이었다. 현재 결과만으로 RL 우위를 주장하지 않는다.
- 실제 완료 chunk를 파싱해 확장 128문항 cohort 관측 141,824개를 확인했다. 16개 대상은 8,192개씩 완료했고 Qwen3-14B 5,120개, StableLM 5,632개가 저장돼 있었다. 대상 내부 중복 키는 없었다.
- `collect128_30`부터 supervisor PID 58072로 백그라운드 재개했다. 이번 단계 목표는 245,760개이며 이후 `train128_30 → collect997_30 → train997_30` 순서다. 전체 계획의 고유 관측 수 1,914,240개는 완료 수치가 아니다.
- `run_selector_gpu_expanded_pipeline.py`에 동일 run root의 동시 supervisor를 막는 파일 잠금을 추가하고 재개 전 상태 스냅샷과 완료 이력을 보존했다. 실제 중복 시작을 시도해 차단 및 활성 상태 파일 보존을 확인했다.
- 기존 `tests/test_selector_gpu.py` 4개 통과. 정보 누출 차단, 후보 순열 등변성 및 PPO 실제 파라미터 갱신 검사다.
- 실행 명령/PID/로그: `private_artifacts/selector_gpu_20260903/resume_launch.json`. 실제 현재 단계: `expanded_pipeline_status.json`. 결과: `results/selector_gpu_20260903/REPORT.md`.
- 재개 후 확인(2026-09-03T15:06:50+09:00): 두 GPU에서 실제 추론 및 신규 chunk 저장 확인. cohort 관측 146,944개로 증가(+5,120). 확인 시 GPU 사용률 68%/99%, supervisor 생존. 아직 확장 학습 완료가 아닌 수집 진행 상태다.

## 2026-09-03 — 재구성 + ASR 목적 정정

사용자는 재구성 능력과 ASR을 함께 평가해야 함을 명확히 했다. 재구성 대리 보상만으로 실행한 PPO를 본래 연구 목표의 성과로 간주한 방향은 잘못됐으며, 해당 확장 수집과 후속 자동 학습 실행을 중지했다. 기존 관측·checkpoint는 보존했다.

기존 5개 패널의 재구성률과 공동 성공률 50개 집계를 연결해 목적 불일치를 확인했다. 원시 판정 파일은 OS 권한으로 접근할 수 없어 새 join 검증이나 재채점은 하지 못했다. 새 64개 설정에는 ASR 자료가 없으며 기존 패널과 구성/프로토콜도 다르다. 공동 목적의 PPO나 온라인 안전성 실험은 아직 실행하지 않았다.

상세: docs/JOINT_EVALUATION_OBJECTIVE_2026-09-03.md. 기존 집계의 출처/해시: results/existing_joint_evaluation_20260903/summary.json.

## 2026-09-03 — 사용자 요청으로 실험 중단 확정

사용자: “먼저 중단하는게 낫지 않을까 설계 자체에 문제가 있어서 진행해도 의미가 없을것 같은데”. 관련 selector 수집·학습·supervisor 프로세스가 없고 GPU compute process도 없는 것을 확인했다. 상태를 `paused_design_review`로 기록했다. 기존 데이터와 checkpoint는 보존하며, 사용자의 새로운 재개 지시 전에는 수집·학습·온라인 실험을 시작하지 않는다.

재검토할 문제: 원래 공동 평가 목적과 무해 어휘 재구성 보상의 불일치, 원문 복사로 풀 수 있는 과제 구조, 언어/조각 수/순서 효과를 분리하기 어려운 후보 설계, 모델별 적응의 가치와 최종 평가 분리. 현재 결과는 실행 기반 검증으로만 취급하고 연구 방법의 효과로 주장하지 않는다.

## 2026-09-03 — 재구성+ASR 공동 목적으로 선택기 재정의

사용자 정정: 강화학습이 재구성만 신경썼고, 제일 중요한 것은 ASR이 되게 만드는 것이다.
너무 쉬우면 감지되어 탈옥이 안 되므로, 감지를 피하면서 재구성이 성공하는 설정이 목표다.

- 기존 `panel_v2`의 문항별 판정을 재집계해 공동 목적을 정의했다. 9개 대상 × 250개 유해
  문항 × 12개 설정에서 R(재구성 동등성)과 U(답변만의 Qwen3Guard 안전성 실패)를 같은
  response_sha256로 연결했다. 불일치 0, parse 무효 0. OLMo-2-7B는 OS 권한으로 제외.
- 주 지표는 reconstruction-gated ASR: J = R AND U. raw ASR, 재구성률, 조건부 ASR을 함께 보고.
  재구성 인용문은 ASR로 채점하지 않고 U는 답변 섹션만 대상으로 한다.
- "너무 쉬우면 감지" 확인: 정렬된 모델에서 평문 직접 요청은 거부(gated 0.05~0.25)되고
  최고 interleaving이 공동 ASR을 +0.30~0.56 올린다. Mistral/Zephyr는 평문에도 거의
  거부하지 않아(0.73) obfuscation 이득이 없다. 모델별로 조건이 다르게 걸린다.
- 축 분리: 언어 수 n과 배열은 분리되고 상호작용이 있다(n2는 ordered 우세, n8은 shuffled 우세).
  조각 수 축은 panel_v2에서 5로 고정돼 분리되지 않으며 유해 요인 수집이 필요하다.
- 적응형 근거(LOO, 방법 후보 10개): 고정 최고 설정 gated 0.410, 오라클 0.539(regret 0.129),
  설정 8개 관측 적응형 0.517로 격차의 약 83% 회수(+0.107). random 0.384.
- 코드 `scripts/joint_objective_analysis.py`, 결과 `results/joint_selector_20260903/`,
  설계·수치 `docs/JOINT_OBJECTIVE_SELECTOR_2026-09-03.md`. 새 생성·judge 호출 없음.
- 무해 재구성 대리 보상으로 학습한 확장 GPU 수집은 중지 상태 유지. GPU 유휴 확인.
  공동 보상으로의 선택기 학습과 조각 수 요인 유해 수집은 다음 실험 사양으로 문서화했고
  아직 실행하지 않았다.

## 2026-09-03 — 조각 수 factorial 유해 공동 수집 + 공동 보상 선택기 학습

- 사용자 요청으로 조각 수 축을 교차한 유해 공동 factorial을 GPU에서 실행했다. 6개 대상
  × 250 유해 문항 × 34 설정(조각 3/5/8/12 × 배열 ordered/shuffled × 언어 2/4/6/8 + 앵커 2).
  셀별 재구성 판정과 답변 전용 Qwen3Guard 판정으로 공동 J를 측정했다.
- Qwen3-8B는 기본 thinking 모드가 320 토큰을 소진해 [ANSWER]가 안 나와 재구성이 0이었다.
  thinking 비활성 재실행을 시작했으나 사용자 지시로 포기하고 제거했다. 정상 5개 모델로 확정.
- 조각 수 축(5개 모델): 조각을 잘게 쪼갤수록 재구성 단조 감소, 공동 ASR도 단조 감소
  (조각3 gated 0.349 → 조각12 0.216), raw ASR 거의 불변. 세 축 중 조각 수가 가장 큰
  단일 효과(모델 중심화 분산 비중 조각 19.8% / 언어 2.4% / 배열 1.1%). 최적 조각 수도
  모델 의존적(4개 모델 조각3, qwen25_14b 조각8). 결과: results/fragment_factorial_20260903/.
  코드: scripts/run_fragment_factorial_vllm.py, run_fragment_factorial.sh, aggregate_fragment_factorial.py.
- 트레이너 입력 어댑터(scripts/build_joint_selector_data.py)로 factorial을 값 테이블
  [5대상,250문항,32설정]의 공동 J로 변환했다. 결측 0, 응답 해시 검증 통과.
- 공동 보상 선택기 학습(scripts/train_joint_selector.py): 검증된 Policy/PPO/rollout/
  evaluate를 import, 계열 3개뿐이라 대상 보류 LOO. test 공동 gated ASR(5대상 평균):
  고정 0.420, gp_ucb 0.507(b16), ppo 0.425, random 0.455, no_feedback 0.411.
  고정의 대상 오라클 regret 0.147을 gp_ucb가 0.060으로 회수. 학습 PPO는 고정과 거의 동일.
  가치는 RL 자체가 아니라 피드백 기반 적응. 결과: results/joint_selector_run_20260903/.
- 관련 핵심 논문: Perez et al. 2022, Red Teaming LMs with LMs(EMNLP), RL 보상=공격 성공.

## 2026-09-03 — probe 예측력 검증(1단계)

- 목적: 블랙박스 행동 probe(정렬=english_direct raw ASR, 다국어=translated_Finnish raw ASR,
  재구성 능력=무해 FLORES 재구성률)가 대상별 최적 obfuscation 설정을 예측하는지. panel_v2
  9개 대상 leave-one-model-out. 유해 interleave 공동 라벨은 특징에서 제외(누출 없음),
  무해 능력은 별도 데이터. 코드: scripts/probe_predictivity_check.py.
- held-out 대상의 공동 gated ASR(예측 설정 적용): oracle 0.531, ridge(probe 회귀) 0.458,
  nn_probe 0.421, global_fixed(무 probe) 0.411. ridge가 고정 대비 +0.047, regret 0.120→0.074로
  오라클 격차의 약 38% 회수. 그러나 9개 대상 부트스트랩 CI가 겹쳐 통계적으로 빡빡하지 않다.
- 해석 신호: Spearman(재구성 능력, 최적 설정 언어수) = 0.27, Spearman(정렬 asr, 최적 gated) = 0.49.
  능력이 높을수록 더 높은 언어수를 견디고, 정렬이 약할수록 공동 ASR이 높다는 가설 방향과 일치.
- 결론: probe 지문에 약~중 예측 신호가 있어 유니버설 방향이 살아 있으나 n=9로 미확정.
  더 많은 대상·계열 수집(2단계)으로 검정력을 확보할 근거는 된다. 결과: results/probe_predictivity_20260903/.

## 2026-09-03 — 2단계: 30모델급 확장 수집 + 문맥 선택기 + online 루프

- panel_v2 프로토콜을 다운로드된 추가 모델로 확장(GPU1 단독, GPU0은 타 사용자 DeepVoice
  점유로 미사용). 신규 17개 성공, DeepSeek-V2-Lite 실패. 총 27개 공동 관측, 학습용 26개/8계열
  (qwen10 smollm4 mistral_zephyr3 falcon3 yi2 granite2 olmo1 phi1). Qwen3-0.6B는 셀 무효로 드롭.
- 문맥 선택기(scripts/train_context_selector.py): 대상 행동 지문(정렬/다국어/재구성 능력)을
  입력받는 정책, 계열 보류 LOO. no_context 절제 포함. 두 GPU가 타 사용자 작업으로 차서
  자동 체인은 학습을 건너뛰었고, GPU1(TTS와 공유, 76GB 여유)에서 직접 실행.
- 결과(test 공동 gated, 78 대상-폴드): 질의0에서 문맥정책 0.333 > no_context 0.313 ≈ fixed 0.312.
  지문 warm-start 이득 +0.020(절제로 지문 기여 분리). 예산8에서 gp_ucb 0.349 best(regret 0.052),
  PPO는 0.330 평평(피드백 약활용). CI 넓음. 결론: 문맥 warm-start + GP-UCB 피드백 하이브리드.
- online 적응 루프(scripts/online_adapt.py, replay): 26대상 평균 질의 5.58, 추천 gated 0.350,
  오라클 0.400, regret 0.050, 예산8내 성공 10/26, 평균 첫성공 1.70질의. PAIR식 조기종료·질의효율.
- 자동 체인 scripts/run_stage2_chain.sh. 결과: results/{context_selector_train,context_selector_run,
  online_adapt}_20260903/. GPU0 타 사용자 작업은 전 과정에서 미접촉.

## 2026-09-04 — DeepSeek 복구 + 27개 대상 문맥 선택기 갱신

- 실패했던 DeepSeek-V2-Lite 수집을 GPU0 여유 공간에서 복구. 원인 2개 해결: (1) 커스텀 코드
  파일 부재로 --trust-remote-code 실패 → vLLM 네이티브 경로, (2) FlashInfer 샘플러 커널
  비호환 → VLLM_USE_FLASHINFER_SAMPLER=0. 다른 사용자(GPU0 DeepVoice, GPU1 TTS) 작업 미접촉,
  메모리 사용률 0.45로 제한. 총 27개 대상/9계열(deepseek 추가).
- online 루프 27개 재실행: 평균 질의 5.78, 추천 gated 0.345, 오라클 0.403, regret 0.057,
  예산8내 성공 10/27, 첫성공 2.0질의.
- 문맥 선택기 27개 재학습: 공유 GPU 경합으로 두 번 굶음(폴드당 5분). 유휴 GPU에서 1 seed
  축소로 완료. 결과 8계열 재현: 질의0 문맥정책 0.329 > no_context 0.315 = fixed 0.315(+0.014),
  예산8 gp_ucb 0.348 best. 결론 불변: 문맥 warm-start + GP-UCB 피드백.
- run_panel_v2.sh에 GPU_MEM_UTIL 환경변수 추가(기본 0.90).

## 2026-09-04 — 무해 해독 기반 컨트리뷰션 + 우리 상황용 최적화 알고리즘

- 사용자 방향: 유해 텍스트 없이 일반 텍스트를 다국어로 흩어 "해독-수행" 능력으로 공격
  설정을 최적화하는 것 자체가 컨트리뷰션(대부분 레드팀은 유해 과제로 발전시킴).
- 전이 검증: 무해 해독률 ↔ 유해 공동 ASR corr 0.84 (224셀). 무해 해독≥0.7 설정의 유해
  ASR 0.50 vs <0.4의 0.09. => 무해로 설정 최적화 후 유해는 최종 검증만.
- 무해 해독 스윕(재구성 패널 19개, panel benign): n2 0.58→n10 0.50, ordered 0.60>shuffled 0.50.
  대상별 해독 곡선이 능력을 구분(qwen3-14b 평탄 0.9+, falcon-7b 급감 0.27→0).
- 우리 상황용 최적화 알고리즘: 지문 warm-start + 고정예산 베이지안 BAI(구조 커널 GP).
  근거 논문 prior-dependent FB-BAI(2402.05878), Bayesian FB-BAI(2211.08572), unimodal(1406.7447).
  재구성 패널 replay: 예산8에서 GP-UCB 0.550/GP-BAI 0.532 > 고정 0.519. 공격가능 대상 원ASR 0.70.
- 데이터 구조: J는 완전 1D 단봉은 아니나(축별 단봉 F 84%/n 58%), 최적점이 능력과 +0.37 상관.
- 고정 32-arm 천장: 평균 오라클 0.63/최대 0.78, 0.80 대상 0개. 돌파하려면 PAIR식 열린 공간
  생성(약한 언어 지렛대). scripts/open_optimize.py 구현(라이브 필요).
- 에이전트 탐침 IRT/CAT 업그레이드는 우리 데이터에서 staircase보다 나빴음(고정 기울기 오설정).
  staircase(level-set) 유지.
- 라이브 스윕/파이프라인은 공유 GPU 경합으로 외부 SIGKILL 반복. online_live 라이브 1건 성공
  (qwen25_7b 공동0.68). per-language 스윕은 오프라인 곡선으로 대체.
- 코드: scripts/{structured_policy_ext,gp_bai,agentic_probe(+irt),open_optimize,benign_recon_sweep}.py.
  문서: docs/{PROBE_OPTIMIZATION_FRAMING,BENIGN_DRIVEN_CONTRIBUTION,AGENTIC_PROBE}_2026-09-04.md.
