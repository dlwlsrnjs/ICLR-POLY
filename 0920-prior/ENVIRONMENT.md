# 실제 환경과 고정값

## 기존 환경을 우선 사용

```bash
cd /home/ubuntu/342/jinkwon/poly
source target_prior_only_20260916/.venv_prior085/bin/activate
```

| 항목 | 확인 값 |
|---|---|
| Python | 3.10.12 |
| torch | 2.6.0 |
| vLLM | 0.8.5 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| GPU | NVIDIA H100 80GB HBM3 두 장, 다른 사용자와 공유 |
| 실행 코드 checkout | `/tmp/poly-model-anchors-20260920` |
| target 모델 캐시 | `/home/ubuntu/342/jinkwon/hf_cache/hub` |
| 번역·보조 모델 캐시 | `/home/ubuntu/342/jinkwon/poly/target_prior_only_20260916/model_cache` |

버전만 설치한다고 동일한 CUDA/커널/모델 파일까지 재현되는 것은 아닙니다. `packages.observed.txt`는 현재 설치 목록이며, 새 환경을 만들어 검증한 재설치 절차는 아닙니다. 기존 환경을 업그레이드하지 않는 것이 우선입니다. 재구성 판정은 accelerate 의존성을 피하도록 CPU 로드 후 `.to(cuda)`를 사용합니다.

## Target 응답 생성: 전 모델 동일

| 설정 | 값 |
|---|---|
| dtype / tensor parallel | BF16 / 1 |
| temperature / top_p | 0 / 1 |
| max_tokens | **1024** |
| seed / repetition_penalty | 0 / 1 |
| max_model_len | 4096 |
| batch / max_num_seqs | 16 / 16 |
| system_message | null |
| strong_reconstruction_prompt | false |
| enforce_eager | true |
| VLLM_USE_V1 | 0 (target 수집) |
| HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE | 1 / 1 |

이 설정은 기존 331 prior 환경에 맞춘 것입니다. “MJ/LG 저장소 전체의 보편적인 고정값”이라는 뜻이 아닙니다. 모든 target의 모델명/정확한 revision은 `configs.snapshot.json`과 기존 `aligned/shared_en_ar/configs/*.json`을 확인합니다.

## 토크나이저와 프롬프트

모델과 토크나이저는 같은 pinned snapshot을 사용합니다. 로컬 snapshot 경로를 넘기면 basename이 설정의 target_revision과 일치해야 합니다. `AutoTokenizer.from_pretrained(..., local_files_only=True)`와 모델 고유 chat template을 사용하고 template 해시를 `collection_metadata.json`에 기록합니다. 모델마다 native template은 다르므로 직렬화된 토큰열까지 모두 동일하다는 뜻은 아닙니다.

`collect.py`는 재개 시 config/jobs/template/runner 해시와 패키지 버전을 검사합니다. 입력 토큰 + 1024가 context 4096을 넘으면 에러를 내고, 원문을 임의로 자르지 않습니다. 응답 재사용은 원문·messages·prompt·조각·정답 순서·샘플링·revision 등의 일치 확인 후에만 허용됩니다.

Mistral 계열에서는 현재 transformers와 tokenizer regex 경고가 있었고, Mistral7B는 `head_dim=None` 초기화 오류가 실제 발생했습니다. 검증 없이 옵션을 바꾸거나 기존 결과와 섞지 말고 별도 compatibility 수정과 pilot 기록이 필요합니다.

## 보조 모델의 역할과 별도 설정

| 역할 | 모델 / revision | 실행 설정 |
|---|---|---|
| 번역·역번역 | MiLMMT-46-12B-v1.0 / `a27dbbb37142ff076990820a1c9f0827beb5d6ea` | 기존 언어와 추가 번역의 token budget이 다름; 번역 metadata 확인 |
| 번역 QA, 기존 행동 판정 | Qwen2.5-32B / `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` | 역할별 prompt/output budget이 서로 다름 |
| 재구성 의미 동등성 | Qwen2.5-7B / `a09a35458c702b33eeacc393d103063234e8bc28` | transformers BF16, batch16, greedy, 기본 새 토큰80 |
| 거절 판정 | WildGuard / `cbba4823f3e8020e5a74a5e29bf85072def6f2ff` | vLLM V1, context8192, temperature0, 기본 출력64, seed20260916 |
| **현재 수행 여부 판정** | **Qwen2.5-32B**, 위 revision | **vLLM V0, BF16, context8192, temperature0, top_p1, seed20260920, 출력384/오류 재시도768** |

수행 판정의 스케줄러 batch는 32, 엔진 max_num_seqs는 8, max_num_batched_tokens는 8192입니다. 384/768은 **판정 모델 출력 한도**이며 target의 1024 설정을 바꾼 것이 아닙니다.

GPU memory utilization은 과학적 조건으로 고정한 값이 아니라 공유 GPU 가용 공간에 맞춘 실행값입니다. 실제 값은 `gpu_allocations.jsonl` 또는 판정 런 `allocation.json`에 기록됩니다. 프로세스 시작 뒤 다른 사용자가 메모리를 확보하면 여전히 OOM이 날 수 있습니다. 다른 사용자 프로세스를 종료하면 안 됩니다.
