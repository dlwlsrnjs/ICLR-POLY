# 확인된 수집 환경

2026-09-20 사용자 확인: 기존 331개 prior의 실제 환경을 지금까지의 수집 기준으로 사용했다.
`willingness_overrefusal331_17models_20260918/models/*/metadata.json` 17개를
대조한 결과도 모두 다음 값과 일치했다.

| 패키지 | 고정 버전 |
|---|---|
| vLLM | 0.8.5 |
| Transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| PyTorch | 2.6.0 |

전체 코어 패키지는 루트 `requirements-collection.lock.txt`에 고정한다.
추가 패키지 버전은 실제 `.venv_prior085` 환경에서 확인했다.
`accelerate`는 이 환경에 설치되어 있지 않으므로 필수 코어 목록에서 제거했다.
이전에 저장소에 쓰인 다른 고정값은 잘못된 선언이었으며, 별도의 MJ/LG 실제
수집 환경을 증명하지 않는다. 수정된 검사기는 위 실제 환경을 기준으로 검사한다.

## 토크나이저

모델별로 해당 모델의 고정 revision에 포함된 tokenizer와 chat template를 사용한다.
모든 모델에 Qwen tokenizer를 사용하는 것이 아니다. Qwen2.5-7B의 기준 revision은
`a09a35458c702b33eeacc393d103063234e8bc28`이며 모델과 tokenizer가 동일 snapshot을 사용한다.
그 snapshot의 chat template는 user-only 메시지를 받으면 기본 Qwen system 문장을
자동 삽입한다. 임의로 template나 special tokens를 교체하지 않는다.

## 환경과 실험 조건의 구분

패키지 버전 정정은 프롬프트·언어 구성·출력 길이·seed를 변경하지 않는다.
기존 331개 prior는 1024 출력 토큰/8192 문맥 길이/seed 20260918,
새 영어+아랍어 실험은 320 출력 토큰/4096 문맥 길이/생성 seed 0이다.
이 차이는 환경 버전 오류와 별개로 계속 기록해야 한다.
GPU 메모리를 위한 eager, 배치, CPU offload 변경도 기록한다.
환경 검사가 통과했다는 사실은 GPU 메모리 충분성이나 전체 모델 실행 성공을 뜻하지 않는다.

```bash
/path/to/.venv_prior085/bin/python scripts/verify_repro_env.py
```

`--strict-reference`는 환경 외에 기존 benchmark 데이터·모델·환경변수 조건까지
검사한다. 신규 영어+아랍어 prior를 기본 MJ/LG와 같은 실험으로 인증하는 옵션이 아니다.
