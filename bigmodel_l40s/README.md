# 🟦 L40S 전용 — 대형/중형 패널 수집 (PolyJigsaw / ICLR-POLY)

> **이 폴더는 L40S 박스에서만 실행합니다.** 공유 A100 박스에서는 소형만 돌리고 있고, 아래 6개
> 모델은 여기(L40S)에서만 수집합니다. 헷갈리지 않도록 이 폴더 밖의 것은 건드리지 마세요.

담당 모델 6개 (그 외 아무것도 여기서 안 돌림):

| 모델 | 태그 | TP | util | 필요 GPU(타깃+판정기) |
|---|---|---|---|---|
| google/gemma-2-2b-it | `gemma2_2b_it` | 1 | 0.30 | 2 |
| tiiuae/Falcon3-7B-Instruct | `falcon3_7b` | 1 | 0.45 | 2 |
| meta-llama/Llama-3.1-8B-Instruct | `llama31_8b_it` | 1 | 0.50 | 2 |
| google/gemma-2-9b-it | `gemma2_9b_it` | 1 | 0.55 | 2 |
| Qwen/Qwen2.5-14B-Instruct | `qwen25_14b` | 1 | 0.85 | 2 |
| mistralai/Mistral-Small-24B-Instruct-2501 | `mistral24b` | **2** | 0.85 | **3** |

- `gemma-2-2b-it`은 원래 held-in인데 공유 박스에서 OOM으로 빠졌던 모델 → 여기서 수집.
- **mistral24b만 TP=2**(48GB L40S 한 장에 안 들어감). 판정기까지 상주시키려면 **L40S 3장** 필요.
  2장뿐이면 AWQ 양자화본으로 대체(아래 §7).
- gemma-2-27b, Qwen2.5-32B는 이번 배치에 **없음**. 원하면 `models.txt`에 주석 해제해서 추가.

수집 대상 = **C=160 arm 공간**(comprehension 32셀 × willingness {plain,persona,fiction,pap,persona+fiction}
= 160) **+ baseline 4** = 164 arm/모델, **데이터셋 2개(MultiJail·Lingua) 각각**. 성공 판정은
재구성-게이트 **verified**(재구성 판정 Qwen2.5-7B **AND** 안전 판정 Qwen3Guard-8B unsafe).

---

## 0. 왜 L40S를 분리했나 / 저장이 어떻게 합쳐지나 (한눈에)

```
[L40S 박스]  git clone → fetch_data.sh(데이터) → 모델 weight 다운 → verify_env.py → run_l40s.sh
                                                                          │
                          결과가 bigmodel_l40s/results/ 에만 쌓임 (공유박스와 분리, 안 헷갈림)
                                                                          │
                                                    upload_results.sh → 프라이빗 버킷 jin-kwon/poly
                                                                          │
[공유 박스]  버킷에서 pull → rsync 로 패널 results/ 에 병합 → repair-manifest → 표 자동 재생성
```

핵심: **L40S 결과는 `bigmodel_l40s/results/`에만 저장**됩니다(공유 박스 결과와 물리적으로 분리).
나중에 버킷 경유로 공유 박스 패널에 **rsync 병합**하면 태그 이름이 그대로라 자동으로 합쳐집니다.

---

## 1. 코드 받기

```bash
git clone https://github.com/dlwlsrnjs/ICLR-POLY.git
cd ICLR-POLY
```

코드는 이 한 번의 clone으로 끝. L40S 전용 실행물은 전부 `bigmodel_l40s/` 안에 있습니다
(`run_l40s.sh`, `models.txt`, `fetch_data.sh`, `verify_env.py`, `upload_results.sh`).

---

## 2. 환경 세팅

파이썬 3.10 기준. vLLM은 L40S(Ada, CUDA 8.9)와 설치한 CUDA/torch가 맞아야 합니다.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -r requirements.txt          # transformers, huggingface_hub 등
pip install "vllm>=0.6.3"                 # torch는 vllm이 맞는 버전을 끌고 옴 (L40S = sm_89)
# 활성화 안 하고 절대경로 파이썬을 쓸 거면 VP 로 지정: export VP=/path/.venv/bin/python

# HF 캐시 위치 (weight/판정기 저장소). 디스크 여유 넉넉한 곳으로.
export HF_HOME=/data/hf_cache            # 예시. 본인 경로로.
```

**판정기 2개 + 타깃 weight 미리 받기** (오프라인 실행을 위해 캐시에 내려둠):

```bash
# 판정기: 재구성 = Qwen2.5-7B-Instruct, 안전 = Qwen3Guard-Gen-8B
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-7B-Instruct'); d('Qwen/Qwen3Guard-Gen-8B')"
# 타깃 6개
for m in google/gemma-2-2b-it tiiuae/Falcon3-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \
         google/gemma-2-9b-it Qwen/Qwen2.5-14B-Instruct mistralai/Mistral-Small-24B-Instruct-2501; do
  huggingface-cli download "$m";
done
```

> gemma / llama 등 gated 모델은 HF에서 라이선스 동의 + `huggingface-cli login` 필요.

---

## 3. 데이터셋 받기 (어디서 / 어떻게)

유해 텍스트는 **정책상 git에 없습니다.** 두 가지 경로:

**(A) 프라이빗 버킷 (권장, 협업자가 접근권 있으면 제일 빠름)** — `jin-kwon/poly`

```bash
export HF_TOKEN=<jin-kwon/poly 접근 권한이 있는 토큰>
bash bigmodel_l40s/fetch_data.sh        # 아래 6개 파일만 내려받고 존재 검증까지 함
```

내려받는 파일(수집이 읽는 전부):

| 파일 | 내용 |
|---|---|
| `private_artifacts/multijail_v1/harm_grid.jsonl` | MultiJail 유해 그리드 (제한) |
| `private_artifacts/multijail_v1/benign_probe.jsonl` | MultiJail 무해 probe (FLORES) |
| `private_artifacts/multijail_v1/resource_order.json` | MultiJail 언어 순서(저자원 우선, tlang=Bengali) |
| `private_artifacts/panel_v2/harm_grid.jsonl` | Lingua 유해 그리드 (제한) |
| `private_artifacts/panel_v2/benign_probe.jsonl` | Lingua 무해 probe |
| `results/lang_rank_20260905/resource_order.json` | Lingua 언어 순서(tlang=Norwegian) |

**(B) 버킷 접근권이 없으면 — 공식 데이터셋에서 재구성**

- **MultiJail** (Deng et al.) — 라이선스에 따른 공식 소스에서 취득
- **Lingua-SafetyBench** (Text-Dominant 파티션) — 공식 소스에서 취득

그런 다음 로컬 파일을 재생성: `python scripts/prepare_lingua_text.py --dataset-root <압축푼경로>` 후
`docs/REPRODUCE.md`의 파일럿 빌더 실행. 최종적으로 위 6개 파일이 있으면 됩니다.

---

## 4. 프리플라이트 (한 번에 점검)

```bash
python bigmodel_l40s/verify_env.py
```

GPU 수, HF_HOME, 판정기 캐시, 타깃 weight 캐시, 데이터셋 6파일, TP 배선까지 전부 체크하고
`ALL GOOD` 또는 고쳐야 할 항목을 알려줍니다. 통과해야 다음으로.

---

## 5. 실행

```bash
# GPU 자동감지(전부 사용). 특정 카드만 쓰려면 CUDA_VISIBLE_DEVICES 로 순서 지정.
bash bigmodel_l40s/run_l40s.sh 2>&1 | tee bigmodel_l40s/run.log
```

동작:
- `models.txt`의 6개 모델을 순서대로, **각 모델마다 MultiJail → Lingua**를,
  `both`(무해 probe + 유해 full-matrix) 로 수집.
- 타깃은 `cuda:0..TP-1`, **판정기는 `cuda:TP`** (타깃 샤드 바로 다음 카드)에 상주.
- 결과는 `bigmodel_l40s/results/` 에만 쌓임.
- **중간에 끊겨도 안전**: 다시 실행하면 `MANIFEST.json` 기준으로 **끝난 arm은 건너뛰고 이어서** 함(resume).

옵션(환경변수):
```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash bigmodel_l40s/run_l40s.sh   # 카드 3장 지정(24B TP=2 + 판정기)
MJ_ITEMS=64 LG_ITEMS=40 JB=8 bash bigmodel_l40s/run_l40s.sh # 문항수/판정 배치 조정
VP=/path/.venv/bin/python bash bigmodel_l40s/run_l40s.sh    # venv 미활성화 시
```

**(선택) willingness 축 prior도 함께 수집** — 이 6개 모델의 오프라인 willingness 지문(전이 기법의
prior)을 같이 쌓으려면 모델당 아래를 실행(무해·over-refusal 데이터, 안전판정 없음).
스키마·의미는 `experiments_suite/exp05_willingness_probe/WILLINGNESS_PRIOR.md` 참고:
```bash
python experiments_suite/exp05_willingness_probe/willingness_prior.py \
    --model Qwen/Qwen2.5-14B-Instruct --tag qwen25_14b --dataset falsereject --util 0.85
# 출력: experiments_suite/exp05_willingness_probe/results/willingness_prior_falsereject_<tag>.json
# 이 폴더도 upload_results.sh 대신 버킷/rsync로 함께 올려 병합
```

한 모델만 따로 돌리고 싶으면 (예: qwen14b):
```bash
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python experiments_suite/exp02_panel_collect/collect_mj.py both --models Qwen/Qwen2.5-14B-Instruct \
   --arm-space C --n-items 64 --root bigmodel_l40s/results \
   --tensor-parallel 1 --util 0.85 --judge-device cuda:1 --judge-batch-size 8
# lg 도 동일하게 collect_lg.py 로 한 번 더
```

---

## 6. 진행 상황 보는 법

```bash
# 지금까지 모델별로 몇 arm 모였나 (목표 164/모델·데이터셋)
python - <<'PY'
import glob,os,collections
c=collections.Counter()
for p in glob.glob("bigmodel_l40s/results/attack/*.json"):
    b=os.path.basename(p)[:-5]
    if "__" in b: c[b.split("__",1)[0]]+=1
for k in sorted(c): print(f"{k:22} {c[k]:>4}/164")
PY

tail -f bigmodel_l40s/run.log          # 실시간 로그 (SKIP/에러 여기서 보임)
nvidia-smi                             # GPU 점유 확인
```

`[l40s ...] SKIP <model>` 가 보이면 그 모델은 건너뛴 것 — 로그 위쪽에서 원인 확인 후
GPU 비우고 재실행하면 이어서 채웁니다.

---

## 7. ⚠️ 주의사항 (꼭 읽기)

- **L40S 전용.** 공유 박스와 결과 폴더가 다릅니다(`bigmodel_l40s/results/`). 절대 공유 박스에서
  이 스크립트를 돌리지 마세요(같은 태그로 덮어쓰기/충돌).
- **판정기 카드 분리 필수.** 타깃과 판정기(≈31GB)를 같은 카드에 올리면 OOM. `TP+1` 장이 필요.
  `verify_env.py`가 GPU 수 부족을 미리 잡아줍니다.
- **mistral24b (TP=2)**:
  - L40S **3장 이상**: 그대로 `run_l40s.sh` (타깃 cuda:0,1 / 판정기 cuda:2). 권장.
  - L40S **2장뿐**: `models.txt`에서 그 줄을 AWQ 양자화본으로 교체 —
    `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ  1  0.85` (타깃 한 장 + 판정기 한 장).
    이 경우 논문/기록에 **"mistral24b는 양자화 타깃"**이라고 반드시 표기.
- **OOM이 나면**: `--util` 낮추기, `JB`(판정 배치) 낮추기, `VLLM_ENFORCE_EAGER=1`(기본 켜짐) 유지,
  다른 사용자 프로세스가 카드를 점유 중인지 `nvidia-smi`로 확인.
- **유해 원본 출력**(`results/attack/_raw/*.jsonl`, 권한 0600)은 **제한 자료**. git 커밋 금지,
  버킷으로만 전송. `upload_results.sh`가 이 규칙대로 처리합니다.
- **오프라인 판정기**: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`이어야 판정기가 캐시에서 로드됩니다
  (안 그러면 네트워크 접근하다 실패). weight 최초 다운로드 때만 잠깐 `HF_HUB_OFFLINE=0`.
- **재현성**: 타깃 샘플링은 temperature=0(그리디). 판정기도 고정. 같은 arm 재실행 시 결과 일관.

---

## 8. 끝나면 — 결과 올리고 병합

```bash
export HF_TOKEN=<쓰기 권한 토큰>
bash bigmodel_l40s/upload_results.sh          # 기본 SINK=bucket (버킷으로)
# 또는 공유 박스로 직접 rsync:
# SINK=rsync RSYNC_DEST=user@shared:/.../experiments_suite/exp02_panel_collect/results/ bash bigmodel_l40s/upload_results.sh
```

그 뒤 **공유 박스에서** 병합:
```bash
hf sync hf://buckets/jin-kwon/poly/PolyJigsaw/bigmodel_l40s/results ./bigmodel_l40s/results
rsync -a bigmodel_l40s/results/ experiments_suite/exp02_panel_collect/results/
python scripts/closed_compare.py repair-manifest --root experiments_suite/exp02_panel_collect/results
# 이후 표 재생성 파이프라인(space_ab_compare / compare_strategies / ... / make_arm_space_table)이 이 데이터를 사용
```

---

## 9. 파일 맵 (이 폴더)

| 파일 | 역할 |
|---|---|
| `README.md` | 이 문서 |
| `models.txt` | 담당 6모델 + TP + util (편집해서 대상 조정) |
| `fetch_data.sh` | 프라이빗 버킷에서 데이터 6파일 받기 + 검증 |
| `verify_env.py` | 실행 전 프리플라이트(GPU/캐시/데이터/배선) |
| `run_l40s.sh` | 수집 실행(모델별 MJ+LG, 판정기 카드 자동 배치, resume) |
| `upload_results.sh` | 결과를 버킷/rsync로 반환(유해 원본은 버킷만) |
| `results/` | **여기에만** L40S 결과가 쌓임(실행 후 생성) |

문의/맥락: 리포 루트 `GITHUB_README.md`, `experiments_suite/INVENTORY.md`,
`experiments_suite/BUCKET.md`, `paper/REVISION_PLAN_2026-09-10.md`.
