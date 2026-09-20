# 수행 판정 전체 외부 서버 인계

HF: [jin-kwon/poly-qwen32-judge-resume-20260920](https://huggingface.co/datasets/jin-kwon/poly-qwen32-judge-resume-20260920/tree/0ac734dc3da1d08c97d7a824aed5d79f41ffad3c/handoff/fulfillment_external_20260921) (private)
고정 revision: `0ac734dc3da1d08c97d7a824aed5d79f41ffad3c`
경로: `handoff/fulfillment_external_20260921`

이 서버는 재구성만 실행합니다. 외부 서버는 gemma2_27b, mistral7b, phi3_medium_14b, qwen25_14b, qwen25_32b, qwen25_3b, qwen25_7b, phi35_mini의 수행 판정만 실행합니다. Mistral24B는 이미 두 단계 모두 완료입니다. Gemma는 재구성 10592개 완료, 수행 1266개 완료/9326개 남음 상태입니다.

```bash
hf download jin-kwon/poly-qwen32-judge-resume-20260920 --repo-type dataset --revision 0ac734dc3da1d08c97d7a824aed5d79f41ffad3c --include 'handoff/fulfillment_external_20260921/*' --local-dir ./received
cd received/handoff/fulfillment_external_20260921
export QWEN32_MODEL=/path/to/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
python judge_one_stage.py --run models/gemma2_27b/all --model "$QWEN32_MODEL" --stage fulfillment --gpu 0 --batch 16
```

나머지 모델은 태그를 바꾸고, 서로 다른 모델을 GPU별로 배정하세요. 같은 모델을 두 프로세스로 동시에 실행하지 마세요. 이미 유효한 판정은 건너뜁니다. 원본 bb69581 판정 코드/rubric/토큰 한도를 유지합니다. protocol의 체크포인트 경로는 이전값을 백업한 후 현지 경로로 갱신합니다. 실행 환경: torch 2.6.0+cu124, transformers 4.57.6, accelerate 1.10.1을 사용했습니다.

모든 입력·판정 스냅샷·설정·코드와 SHA256SUMS.json이 포함됩니다. 수행 판정에는 재구성 완료를 기다릴 필요가 없습니다. 최종 분석 시 이 서버의 최종 reconstruction.jsonl과 외부 서버의 fulfillment.jsonl을 key 및 source_response_sha256으로 결합하세요. 전달본의 미완료 reconstruction 스냅샷을 최종 결과로 간주하지 마세요.

외부 서버 작업 시작은 해당 서버에서 해야 합니다. 기존 이 서버 run_assigned 큐와 이전 자동 큐는 정지 상태로 보존했고, 현재는 run_reconstruction_only.py만 새로 실행했습니다.
