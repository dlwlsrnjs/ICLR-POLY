# 작업 지시 — MultiJail 번역 baseline 재수집 (2026-09-15)

다른 GPU 박스에서 실행할 작업이다. **읽고 바로 실행할 수 있게** 필요한 것만 적는다.
분량은 생성 1,088건, GPU 반나절 이내다.

## 1. 왜 필요한가

MultiJail의 저자원 번역 baseline이 두 수집에서 **서로 다른 언어**로 만들어졌다.

| 수집 | 번역 언어 | 근거 |
|---|---|---|
| 논문 수집 (2026-09-08) | **Swahili** | `results/extra_arms_mj_20260908/*.json`의 `answer_lang` |
| 현재 17모델 패널 (L40S) | **Bengali** | `scripts/closed_compare.py`의 `COLLECTION_INPUTS["MultiJail"]["tlang"]` |

둘 다 MultiJail에 실제로 존재하는 저자원 언어다. `resource_order.json` 기준 1순위가 Bengali,
2순위가 Swahili라서, 규칙("가장 저자원 언어")을 엄격히 적용하다 언어가 바뀌었다. 버그가 아니라
설정 변경이다.

이 한 줄 때문에 성능이 크게 달라진다. 9모델 평균으로 논문 0.563 vs 현재 0.234이다.

**언어가 원인이라는 증거**: Lingua는 양쪽 모두 Norwegian을 썼고, 문항이 40개 중 5개만 겹치는데도
값이 거의 그대로다(Qwen2.5-7B 0.325/0.325, 14B 0.200/0.200, 32B 0.175/0.175, Llama-3.1-8B 0.100/0.100).
같은 언어면 재현되고 언어가 바뀐 쪽만 무너졌다.

## 2. 무엇을 뽑는가

**MultiJail 번역 arm을 Swahili로 추가 수집한다. 기존 Bengali 결과는 지우지 않는다.**
두 언어를 모두 가지면 "저자원 언어 선택에 대한 민감도"를 보고할 수 있고, 논문 수치와도 이어진다.

- 대상: 아래 17개 모델
- 문항: 64개 (시드 `20260909`로 harm_grid 315개에서 고정 추출되므로 별도 지정 불필요)
- arm: 번역 하나만. 격자 arm과 다른 baseline은 건드리지 않는다.
- 출력 라벨: `translated` (기존 `m_translated`와 파일명이 달라 충돌하지 않는다)

## 3. 실행 명령

정본 코드는 **이 저장소**다. 버킷의 `scripts/closed_compare.py` 사본은 구버전이라
`build_arms(order)` 시그니처에 Norwegian이 하드코딩돼 있다. 그것으로 돌리면 MultiJail에
Norwegian이 없어 **조용히 영어로 폴백**한다. 반드시 저장소 최신 코드를 쓸 것.

```bash
git clone https://github.com/dlwlsrnjs/ICLR-POLY.git && cd ICLR-POLY
# 데이터셋 입력은 버킷에서: private_artifacts/multijail_v1/{harm_grid.jsonl,resource_order.json}
```

모델 하나당 아래를 실행한다. `<HF_ID>`와 `<TAG>`는 4절 표에서 가져온다.

```bash
python3 scripts/closed_compare.py attack \
  --backend vllm --model <HF_ID> --tag <TAG>_mj \
  --collection MultiJail \
  --root experiments_suite/exp02_panel_collect/results \
  --n-items 64 --methods translated --tlang Swahili \
  --judge-device cuda:1 --util 0.45 --max-tokens 320
```

- `--methods translated`가 번역 arm만 돌린다. `--all-arms`를 **주지 말 것**. 주면 164개 전체가 다시 돈다.
- 27B/32B/24B는 `--tensor-parallel 2`를 추가한다.
- 이미 결과가 있으면 자동으로 건너뛴다. 덮어쓰려면 `--force`.
- 판정기는 재구성 Qwen2.5-7B, 안전 Qwen3Guard-Gen-8B로 자동 로드된다. 번역 arm은 게이트 면제라
  재구성 판정은 형식상만 붙는다.

## 4. 모델 목록과 고정 리비전

| 태그 | HF id | 리비전 |
|---|---|---|
| qwen25_3b | Qwen/Qwen2.5-3B-Instruct | aa8e72537993ba99e69dfaafa59ed015b17504d1 |
| qwen25_7b | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| qwen25_14b | Qwen/Qwen2.5-14B-Instruct | (기록 없음, 최신 태그 고정 후 기록할 것) |
| qwen25_32b | Qwen/Qwen2.5-32B-Instruct | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd |
| llama32_3b_it | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4f764b7a12671c53f0838cd831a0843b95 |
| llama31_8b_it | meta-llama/Llama-3.1-8B-Instruct | (기록 없음) |
| gemma2_2b_it | google/gemma-2-2b-it | (기록 없음) |
| gemma2_9b_it | google/gemma-2-9b-it | (기록 없음) |
| gemma2_27b | google/gemma-2-27b-it | aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0 |
| mistral7b | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 |
| mistral24b | mistralai/Mistral-Small-24B-Instruct-2501 | (기록 없음) |
| phi35_mini | microsoft/Phi-3.5-mini-instruct | 2fe192450127e6a83f7441aef6e3ca586c338b77 |
| phi3_medium_14b | microsoft/Phi-3-medium-4k-instruct | 48d87cd5e0523b430d77e93becd3655cd6897230 |
| falcon3_3b | tiiuae/Falcon3-3B-Instruct | 411bb94318f94f7a5735b77109f456b1e74b42a1 |
| falcon3_7b | tiiuae/Falcon3-7B-Instruct | (기록 없음) |
| falcon3_10b | tiiuae/Falcon3-10B-Instruct | 8799bc6aec0152757221dc6b272d824642db6202 |
| glm4_9b | THUDM/glm-4-9b-chat-hf | 8599336fc6c125203efb2360bfaf4c80eef1d1bf |

리비전이 비어 있는 모델은 받은 리비전을 status 파일에 기록해 둘 것. 기존 패널과 같은 가중치여야
비교가 성립한다.

## 5. 끝나고 확인할 것

```bash
ls experiments_suite/exp02_panel_collect/results/attack/*__translated.json | wc -l   # 17 이어야 함
python3 - <<'PY'
import json,glob
for p in sorted(glob.glob('experiments_suite/exp02_panel_collect/results/attack/*_mj__translated.json')):
    d=json.load(open(p)); print(d['tag'], 'unsafe', d['unsafe'], 'n', d['n'])
PY
```

응답 언어가 스와힐리인지도 한 파일만 눈으로 확인한다
(`attack/_raw/qwen25_7b_mj__translated.jsonl`의 `raw_output`).

기대치: 논문 9모델의 값이 논문 수집(0.563 부근)에 가까워야 한다. 여전히 0.2대라면 언어가
적용되지 않은 것이므로 `--tlang`이 먹었는지부터 볼 것.

## 6. 결과 올리기

버킷에 올린다. **쓰기 권한 토큰이 필요하다.** 현재 공유 토큰은 `repo.content.read`만 있어 403이 난다.

```bash
hf sync ./experiments_suite/exp02_panel_collect/results/attack \
        hf://buckets/jin-kwon/poly/PolyJigsaw/0913/L40S-only/experiments_suite/exp02_panel_collect/results/attack
```

유해 원시 출력(`_raw/*.jsonl`)은 버킷에만 둔다. 공개 저장소에 커밋하지 않는다.

## 7. 하지 않아도 되는 것

- **격자 arm 재수집 불필요.** 164개 전체는 이미 17모델 전부 채워져 있다.
- **문항 표본을 논문과 맞출 필요 없음.** 논문은 앞 64개, 현재는 시드 추출로 10개만 겹치지만,
  같은 언어인 Lingua에서 문항만 바뀌었을 때 변동이 평균 0.061로 작았다. 언어가 주원인이다.
- **무해 probe 재수집은 보류.** MultiJail에서 무해 prior가 실성능과 음의 상관(-0.106)이라는 별도
  문제가 있으나, 번역 arm 언어와는 독립이다. 이 작업이 끝난 뒤 따로 판단한다.

## 8. 배경

이 작업이 왜 지금 필요한지는
[HANDOFF_2026-09-15.md](HANDOFF_2026-09-15.md)와
[FINDINGS_17MODEL.md](../experiments_suite/exp07_panel17_selector_replay/FINDINGS_17MODEL.md)에 있다.
