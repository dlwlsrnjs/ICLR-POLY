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
