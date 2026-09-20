# 현재 역할: 재구성만 이 서버, 수행 판정은 외부 서버

사용자 지시에 따라 모든 남은 수행 판정을 외부 서버로 넘겼습니다. 이전 run_assigned.py와 이전 실험의 자동 큐는 계속 SIGSTOP 상태로 보존되어 있으며 자동으로 재개하지 않습니다. 현재 실행 진입점은 run_reconstruction_only.py입니다. judge_one_stage.py --stage reconstruction만 실행합니다. 완료된 Gemma/Mistral24 재구성은 건너뛰며 나머지 모델의 재구성만 진행합니다.

외부 서버에는 fulfillment_external/package의 입력·현재 결과와 judge_one_stage.py --stage fulfillment 실행법을 전달합니다. 다른 서버가 수행 판정을 실행 중이라는 의미는 아니며, 다운로드 후 그 서버에서 시작해야 합니다. 현재 인계본의 미완료 재구성 파일은 이 서버의 향후 최종 재구성 결과로 갱신해야 합니다. 아래 기록은 시간순 변경 이력이므로 최신 역할 분리를 우선합니다.

# 2026-09-21 외부 서버 Qwen32 판정 분담

원본 서버 담당자가 지정한 9개 모델만 이 서버에서 처리합니다. 원본 서버의 나머지 8개 모델은 처리하지 않습니다. 각 GPU가 Qwen2.5-32B-Instruct 한 개를 로드해 배정된 모델 응답의 재구성 및 실제 수행을 판정합니다. 새 target 응답 생성 작업이 아닙니다.

- HF: `jin-kwon/poly-qwen32-judge-resume-20260920` (private)
- HF revision: `1168515989bc8ca9da902dc340670086d4eab39e`
- 번들 SHA-256: `b28d1cf8ce5afa17a03ac4d004e029873d1cf0f55027fdcd167ac2a94ed9820e` (213298006 bytes, 검증 완료)
- 코드: GitHub `work/0920-prior-slurm`, `bb69581`과 byte 단위 일치 확인. 수정하지 않음.
- Qwen32 revision: `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
- 입력: 모델당 10592개, 전체 17개 입력 건수 검증. 이 서버 담당 9개 입력 키 중복 및 기존 판정의 source response SHA 검증.
- 담당: gemma2_27b, mistral24b, mistral7b, phi3_medium_14b, qwen25_14b, qwen25_32b, qwen25_3b, qwen25_7b, phi35_mini.
- GPU 0 batch=2, GPU 1 batch=16. 원본 스크립트의 Transformers BF16/SDPA backend 사용. reconstruction 최대 96, fulfillment 최대 256 생성 토큰. rubric/온도/revision 변경 없음.
- Python: 기존 `.venv_prior085` (torch 2.6.0+cu124, transformers 4.57.6). 해당 환경을 수정하지 않고 `runtime_deps`에 accelerate 1.10.1만 추가, 자식의 PYTHONPATH로 주입.
- 기존 protocol.json의 checkpoint 경로만 이전. 나머지 모든 필드 일치 검증. 원본은 `original_protocols/`, 변경 기록은 `migration_audit.json`에 보관. 번들 원본도 그대로 보존.

## 실행과 상태

`run_assigned.py`는 작업 목록을 두 GPU가 공유하도록 배분합니다. 동일 모델을 두 번 동시에 실행하지 않으며 기존 valid 기록을 건너뜁니다. 각 모델은 최대 3회 실행 후 입력 키·응답 해시·stage별 유효 라벨을 검사합니다. 실패/미완료를 완료로 표시하지 않습니다. 상태 확인:

```bash
cat process.json queue_status.json gpu0.json gpu1.json
```

`logs/`에 모델별 진행/오류, `results/`에 모델별 검증 결과가 저장됩니다. 실제 출력은 `download/qwen32_resume_bundle/models/<model>/all/judge_qwen32/{reconstruction,fulfillment}.jsonl`입니다. 기존 입력, manifest, protocol과 함께 회수해야 합니다. 원본 서버로 결과 반환은 아직 수행하지 않았습니다.

기존 우리 GPU0 adaptive worker와 GPU1 fulfillment supervisor는 PID/명령을 확인하고 SIGSTOP했습니다. 상세는 `paused_previous_queues.json`. 이번 큐 종료의 finally에서 PID 시작 시각을 다시 확인한 뒤 SIGCONT합니다. 다른 사용자의 프로세스는 변경하지 않았습니다. `previous_queues_restored.json`은 이전 실패 실행의 기록일 수도 있으므로 시각과 실제 ps 상태를 함께 확인해야 합니다.

백그라운드 세션은 채팅 종료 후에도 유지됩니다. 재부팅 또는 SIGKILL 시 자동 복구는 보장하지 않습니다. 정상 정지는 process.json의 큐 PID에 SIGTERM을 보내면 자식 프로세스를 종료한 뒤 기존 큐를 복원합니다. GPU 메모리 여유는 공유 서버에서 변동되며, 부족하면 기다립니다. 같은 폴더에서 큐를 중복 실행하면 flock가 차단합니다.

배치 확대 요청을 반영했습니다. OOM 발생 시 저장 결과를 유지하고 해당 작업 배치를 절반으로 내려 재시도합니다.

## GPU0 배치 4 시험

사용자 요청으로 GPU0만 batch 4로 재개합니다. `gpu0_batch_trial.py`가 원본 판정 코드를 별도 프로세스로 실행하며 CUDA OOM이면 `gpu0_batch_override.json`에 batch 2를 기록하고 저장된 valid 판정부터 재개합니다. 이후 모델에도 fallback batch 2를 유지합니다. GPU1 batch 16 및 실행 중 프로세스는 변경하지 않았습니다. run_one_gpu 원본은 `run_one_gpu.original.sh`에 보존했습니다. 판정 Python 코드/rubric/토큰 한도는 변경하지 않았습니다.

사용자의 추가 요청으로 GPU0 batch 8을 시험합니다. CUDA OOM이면 8 → 4 → 2 순서로 낮추고 해당 값을 이후 모델에도 유지합니다. GPU1 batch 16은 계속 실행합니다.

## Gemma2-27B 우선 병렬 처리

사용자 요청으로 Gemma의 남은 수행 판정을 bb69581의 기본 index-modulo 방식 8개 shard로 분할했습니다. 기존 canonical 결과는 gemma_dual_gpu/*before_split.jsonl에 보존하고 유효 결과를 해당 shard에 미리 채웠습니다. 재구성은 전부 재사용합니다. gemma_dual_gpu.py가 GPU 0(batch 2), GPU 1(batch 16)에 다음 shard를 동적으로 배정합니다. GPU1에서 시작됐던 Mistral7B는 저장 결과를 유지한 채 중지했습니다. 기존 run_assigned 부모는 SIGSTOP 상태이며, Gemma shard 처리·검증·canonical 병합 후 SIGCONT하여 Mistral7B와 나머지 큐를 재개합니다. 기존 GPU 대기 큐 복원은 원래 run_assigned가 담당합니다.

진행 상태: gemma_dual_gpu/status.json 및 shard별 JSON/로그. 합쳐진 결과는 기존 all/judge_qwen32/{reconstruction,fulfillment}.jsonl입니다. 실패 후에도 shard 파일과 원본 백업은 남습니다. supervisor를 강제로 중단했다면 기존 shard 파일 때문에 그대로 재시작하지 말고 상태를 확인해 수동 재개해야 합니다.
