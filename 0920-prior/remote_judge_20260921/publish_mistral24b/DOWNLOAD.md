# Mistral24B 완료 후 자동 업로드

비공개 데이터셋: `jin-kwon/poly-qwen32-judge-resume-20260920`

업로드 위치: `results/gpu_server_20260921/mistral24b/`

`publish_mistral24b.py`가 30초 간격으로 판정 큐의 모델 완료/최종 재시도 결과를 확인합니다. 완료 시 입력 10592건, manifest, protocol, 양쪽 판정 JSONL, 고정 판정 코드, coverage 요약과 SHA-256 목록을 복사해서 업로드합니다. 오류가 최종 재시도 후 남으면 NEEDS_REVIEW로 표시합니다. 업로드 후 고정 commit에서 다시 내려받아 전 파일 해시를 검증하고 `receipt.json`에 revision을 기록합니다. 원본 데이터셋의 기존 번들은 변경하지 않습니다.

아직 업로드 완료 전에는 아래 경로에 파일이 없을 수 있습니다. 완료 후 저장소 읽기 인증이 있는 다른 서버에서:

```bash
hf download jin-kwon/poly-qwen32-judge-resume-20260920 \
  --repo-type dataset \
  --include 'results/gpu_server_20260921/mistral24b/*' \
  --local-dir ./mistral24b_received
```

재현 가능한 다운로드에는 receipt.json의 revision을 `--revision`으로 지정하세요. 모든 stage가 complete인지 SUMMARY.json에서 확인하고 SHA256SUMS.json으로 파일 해시를 검증하세요. 재시도 이력에는 같은 key가 여러 번 나타날 수 있습니다. valid이면서 해당 stage의 스키마가 올바르고 응답 해시가 일치하는 결과로 key당 한 건씩 집계해야 합니다.
