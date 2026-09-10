# 다른 곳에서 이어서 작업하기 — HF 프라이빗 버킷

코드와 데이터셋을 프라이빗 버킷 `hf://buckets/jin-kwon/poly`에 두고, 세션을 옮겨도 받아서 이어서
작업한다. 버킷은 **프라이빗**(소유자 전용)이며 접근 제한 연구 자산을 담는다. 공개하지 말 것.

> **CLI 요구사항**: 버킷 명령(`hf sync`, `hf buckets`)은 최신 `huggingface_hub`(≥1.30)에만 있다.
> 이 서버 vLLM 환경의 hf(0.36)에는 없으니, 업로드/다운로드용으로는 최신 hf를 별도로 설치한다:
> `pip install -U "huggingface_hub[cli]"` (그러면 `hf sync`/`hf buckets`가 생긴다).

## 올릴 때 (현재 머신에서, 소유자 인증 필요)

```bash
pip install -U "huggingface_hub[cli]"   # 버킷 지원 hf
hf auth login                           # HF 토큰 입력 (본인만 가능)
bash scripts/bucket_sync.sh             # 시크릿/캐시/모델가중치 제외하고 업로드
```

`bucket_sync.sh`가 자동 제외: API 키/시크릿, 토큰/HF 캐시, 파이썬 캐시, `.git`, 가상환경,
**모델 가중치**(`private_artifacts/**/*.safetensors` 등 수백 GB, 재다운로드 가능). 포함: 코드
(`scripts/`, `experiments_suite/`), `paper/`, `docs/`, `results/`(집계+`_raw`), 그리고 **작은 데이터셋
입력**(`private_artifacts/{multijail_v1,panel_v2,paper_main}/`의 harm_grid·benign_probe·order).
업로드 전 시크릿 스캔을 돌리고, 걸리면 중단한다. (현재 버킷 크기 ≈ 173MB / 1869파일.)

## 새 머신에서 받을 때 (검증된 명령)

```bash
# 1) 버킷 지원 hf 설치 + 로그인
pip install -U "huggingface_hub[cli]"
hf auth login

# 2) 버킷 목록 확인 / 전체 작업트리 내려받기
hf buckets ls -R jin-kwon/poly/PolyJigsaw          # 무엇이 있는지
hf sync hf://buckets/jin-kwon/poly/PolyJigsaw ./PolyJigsaw   # 전체 다운로드(bucket -> local)
#   일부만:  hf sync hf://buckets/jin-kwon/poly/PolyJigsaw/scripts ./PolyJigsaw/scripts
#   대안:    hf buckets cp -R hf://buckets/jin-kwon/poly/PolyJigsaw ./PolyJigsaw

# 3) 파이썬 환경 (vLLM + transformers) 재구성
cd PolyJigsaw
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # + vllm, torch (GPU에 맞춰)

# 4) 판정기 + 대상 모델 (버킷엔 가중치 없음 → HF에서 재다운로드)
#   판정: Qwen/Qwen2.5-7B-Instruct (재구성), Qwen/Qwen3Guard-Gen-8B (안전)
export HF_HOME=<원하는 캐시경로>
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-7B-Instruct'); d('Qwen/Qwen3Guard-Gen-8B')"
```

## 무엇이 어디에

| 경로 | 내용 |
|---|---|
| `scripts/` | 수집·판정·분석 코드(공유 엔진 포함) |
| `experiments_suite/` | 실험별 폴더(데이터셋 분리 드라이버), `README.md`·`INVENTORY.md` |
| `private_artifacts/multijail_v1/` | MultiJail: `harm_grid.jsonl`(유해, 접근제한), `benign_probe.jsonl`(무해 FLORES), `resource_order.json` |
| `private_artifacts/panel_v2/` | Lingua: 같은 구성 |
| `results/lang_rank_20260905/resource_order.json` | Lingua 언어 순서 |
| `results/` | 집계 JSON + `_raw*`(유해 원시출력, 접근제한) |
| `paper/` | tex, 표(`tab_*.tex`), 매크로(`*_numbers.tex`) |

## 접근·재배포 주의

- `harm_grid.jsonl`과 `_raw*`는 Lingua-SafetyBench / MultiJail에서 파생된 **유해·접근제한** 자료다.
  버킷을 프라이빗으로 유지하고 제3자에게 공유하지 말 것. 원본 데이터셋은 각 공식 경로의 이용
  조건을 따라 별도로 취득한다.
- `benign_probe.jsonl`은 FLORES 기반 무해 뉴스 문장이라 제약이 없다.
- 공개 GitHub에는 유해 자료를 커밋하지 않는다(README 참조). 버킷(프라이빗)만 사용.
