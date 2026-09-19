# FalseReject 394 / MiLMMT prior 수집 코드

**MJ·LG에 맞춘 새 수집 설정은 [aligned/README.md](aligned/README.md)에 있습니다.** 무해 이해 prior로 `g3_ordered_n2`를 고정하고 기존 grid의 정확한 5프레임을 사용합니다. 아래 원본 보관본은 변경하지 않았습니다.

Qwen2.5-7B-Instruct 대상으로 실행했던 `willingness394_milmmt_v3` 수집 소스의 독립 보관 폴더입니다. 원본 Python·shell 코드, 설정과 프레임 템플릿 27개를 변경 없이 보존했습니다. `SOURCE_MANIFEST.json`의 SHA-256으로 원본 일치를 검증할 수 있습니다.

394개 원문, 번역, 조각 정렬, 생성 응답, 모델 가중치, 캐시와 기존 prior 데이터는 포함하지 않습니다. `COMPATIBILITY_AUDIT.json`은 GPU 호출 없는 코드 호환성 검사 결과입니다.

## 코드 위치

- [`runs/willingness394_milmmt_v3/code/scenarios.py`](runs/willingness394_milmmt_v3/code/scenarios.py): 조각 정렬 요청, 111개 상황, 퍼즐·대조군 입력 생성.
- [`runs/willingness394_milmmt_v3/code/run.py`](runs/willingness394_milmmt_v3/code/run.py): 정렬 검증 및 대상 모델 응답 수집.
- [`runs/willingness394_milmmt_v3/code/collect_pipeline.py`](runs/willingness394_milmmt_v3/code/collect_pipeline.py): 이미 준비된 입력에 대해 수집 → WildGuard → 행동 판정 → 집계 → 내보내기.
- [`runs/willingness394_milmmt_v3/code/translate.py`](runs/willingness394_milmmt_v3/code/translate.py), `translation_qa.py`, `accept_translations.py`: MiLMMT 번역·역번역 및 Qwen32 검증.
- [`runs/willingness394_milmmt_v3/code/summarize.py`](runs/willingness394_milmmt_v3/code/summarize.py), `export_prior.py`: 상황별 관측과 판정 집계.
- [`runs/willingness394_milmmt_v3/configs/legacy_394.json`](runs/willingness394_milmmt_v3/configs/legacy_394.json): 실제 111상황 설정. `expanded_394.json`은 확장 계획이며 전체 실행 완료를 의미하지 않습니다.
- [`runs/willingness394_milmmt_v3/inputs/frames.json`](runs/willingness394_milmmt_v3/inputs/frames.json): 11프레임의 정확한 문구.
- [`prototypes/game_frames_v1/build_examples.py`](prototypes/game_frames_v1/build_examples.py): 원본 공통 게임 지시와 출력 계약. 코드에 포함된 예시 문항은 합성 예시입니다.
- [`runs/wildguard_labeling_v1/code/`](runs/wildguard_labeling_v1/code): 기존 판정기 구현. `prepare_judges.py`가 실행 폴더로 복사합니다.

## 보관된 실험의 범위

대상 응답 모델은 Qwen2.5-7B-Instruct 하나입니다. MiLMMT-46-12B는 번역, Qwen2.5-32B는 번역·정렬·행동 판정, WildGuard는 거부·유해성 판정에 사용됐습니다.

111상황 = 영어 원문 1 + 4언어 전체문장 × 11프레임 44 + 2/4언어 혼합 × 배열 3종 × 11프레임 66. 퍼즐은 전체 5조각이며 배열은 ordered/reverse/shuffled입니다. 2026-09-20 원본 출력 확인 시 준비된 28,565개 응답이 모두 저장돼 있었고, WildGuard 유효 판정 28,565개, 행동 유효 판정 12,370개였습니다. 이는 인간 정답 라벨이나 160조합 전체 관측을 뜻하지 않습니다.

## 실행 의존성

이 폴더는 **원본 소스 보존본**입니다. 데이터 없이 바로 추론할 수 있는 배포 패키지가 아니며, 기존 경로와 GPU1/vLLM 0.8.5 검사를 그대로 유지합니다. 새 환경 이식은 아직 수행하지 않았습니다. 원래 프로젝트의 `target_prior_only_20260916`에 대응하는 구조를 유지했으므로 원래 환경에서는 아래 파일을 복원한 후 기존 실행 흐름을 사용할 수 있습니다.

1. 수집 폴더의 `inputs/items.json`에 기존 선별 394문항(`id`, `prompt`)을 복원합니다. `inputs/frames.json`은 포함되어 있습니다.
2. 준비된 입력을 재사용하려면 `inputs/accepted_translations.json`, `outputs/alignment.jsonl`, `outputs/alignment_qa.jsonl`이 필요합니다. 기존 jobs/build manifest를 이동하면 절대경로가 달라지므로 `manage.py build`로 다시 생성해야 합니다.
3. 원본 번역 비교·검증 코드에는 형제 폴더 `runs/willingness394_scenarios_v2/inputs/items.json`, `accepted_translations.json` 의존성이 있습니다. shell은 같은 형제 폴더의 `cache/gpu1.lock`을 공유합니다.
4. 패키지 기준 `.venv_prior085/bin/python`, `model_cache/`와 코드에 기록된 `/home/ubuntu/342/jinkwon/hf_cache/hub/` 모델 경로가 필요합니다. 원래 환경의 vLLM 0.8.5, torch, transformers 및 고정 revision을 사용합니다. 환경을 바꿀 경우 경로 수정과 별도 검증이 필요합니다.
5. `outputs/`, `logs/`, `logs/full_collection/`, 필요한 실행 출력 폴더를 준비합니다. 기존 메타데이터·해시는 소스나 입력이 바뀐 실행과 섞지 않습니다.

입력이 준비된 원래 환경에서 실행하는 명령:

```bash
cd runs/willingness394_milmmt_v3
../../.venv_prior085/bin/python code/manage.py build --config configs/legacy_394.json
../../.venv_prior085/bin/python code/collect_pipeline.py
```

번역부터의 원래 실행 순서는 `translate.sh` → `translation_qa.sh` → `run.sh align-qa --config configs/legacy_394.json` → 위 build/collection입니다. `prepare_inputs.sh`는 확장 설정의 정렬까지 요청하므로 legacy 수집만의 최소 준비 명령과 다릅니다. `collect_all.sh`는 summarize에서 끝나며 최종 export는 `collect_pipeline.py`에 포함됩니다.

## 32 이해 설정 × 5프레임 호환성

[호환성 검토](COMPATIBILITY_KO.md)를 먼저 확인하세요. MJ와 LG의 C=160 공간은 서로 같은 5프레임을 사용합니다. 이 보관본의 legacy 4프레임은 문구가 정확히 같지만 plain, 전체 wrapper, 조각 수의 의미가 다릅니다. 기존 응답을 동일한 160개 arm의 관측으로 재명명할 수 없습니다.

CPU만 사용하는 재검증:

```bash
python3 experiments_suite/exp08_willingness394_milmmt_v3/audit_compatibility.py
```

검사는 원본 소스 해시, 4개 프레임의 문자열 일치와 plain 불일치, 합성 입력의 총 조각 수 차이, Qwen7B의 MJ·LG별 32개 이해 셀과 C=160 결과 존재를 확인합니다. 모델 추론이나 원본 응답 정확도 검증은 수행하지 않습니다.
