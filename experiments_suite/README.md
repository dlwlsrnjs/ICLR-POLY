# PolyJigsaw 실험 스위트

실험별 폴더 하나씩. **베이스라인과 우리 제안 기법이 한 실행에서 같이 돌고 한 곳에 수집**된다.
두 데이터셋은 언어 집합·순서·번역언어(tlang)가 다르므로 **드라이버 파일을 데이터셋별로 분리**했다
(`mj.py` = MultiJail, `lg.py` = Lingua-SafetyBench). 공유 로직은 `common/engine.py`가 담당하고,
데이터셋 고유 세팅만 각 드라이버가 명시한다.

## 데이터셋 세팅 (반드시 분리해야 하는 이유)

| | MultiJail | Lingua-SafetyBench |
|---|---|---|
| 언어(10) | English + Bengali·Swahili·Javanese·Arabic·Korean·Thai·Chinese·Italian·Vietnamese | English + Norwegian·Finnish·Arabic·Russian·German·Japanese·Chinese·French·Spanish |
| order | `private_artifacts/multijail_v1/resource_order.json` (저자원 우선) | `results/lang_rank_20260905/resource_order.json` |
| benign probe | `private_artifacts/multijail_v1/benign_probe.jsonl` | `private_artifacts/panel_v2/benign_probe.jsonl` |
| harm | `private_artifacts/multijail_v1/harm_grid.jsonl` (315) | `private_artifacts/panel_v2/harm_grid.jsonl` (250) |
| 번역 baseline tlang | **Bengali** (최저자원, 존재) | **Norwegian** (존재) |

과거에 실제로 났던/막은 버그: (1) Lingua order를 MultiJail에 쓰면 KeyError, (2) 무해 probe 파일을
안 맞추면 엉뚱한 언어로 재구성 측정, (3) tlang이 없으면 translated baseline이 **조용히 영어로 폴백**.
엔진이 이 셋을 컬렉션 자동 해석 + tlang 존재 검증으로 막는다.

## Arm 공간 = comprehension × (스택 가능) willingness 교차곱

- 재구성 셀 = frag{3,5,8,12} × {ordered,shuffled} × n{2,4,6,8} = 32 (n은 데이터셋 언어 수로 자동 상한)
- willingness = {persona, fiction, pap}의 멱집합(plain 포함 8) + role = 셀당 9
- 32 × 9 = 288 + 단일벡터 baseline 4 = **292 arm**. 두 데이터셋 다 292(언어 수가 둘 다 10이라 동일;
  언어가 적은 데이터셋이면 자동으로 줄어든다).

## 공통 프로토콜 (모든 실험)

- **phase 1 probe (무해)**: 평문 퍼즐 재구성(셀당 1회) + 무해 프레임 순응 신호 → warm-start prior →
  선택 arm + shortlist. 유해 요청 없음, 안전 판정기 미적재. Claude가 실행 가능.
- **phase 2 attack (유해)**: shortlist(우리 확정 풀이) + baseline 전부를 유해셋에 적용, verified
  (재구성 AND unsafe) 채점, 문항별 raw를 0600으로 보존. 연구자가 실행.
- **audit (오프라인)**: 전 arm 빌드·중복 점검. API/GPU 불필요. 안전한 파일럿.

## 실험 목록

| 폴더 | 실험 | 상태 |
|---|---|---|
| `exp01_closed_compare/` | 폐쇄모델(GPT-4o 등)에서 우리 기법 vs baseline (verified) | **완료·파일럿 통과** (audit + MJ probe live) |
| `exp02_panel_collect/` | 패널 16모델 292-arm 전수 수집 | 진행 중 (Qwen2.5-7B부터 수집) |
| `exp03_heterogeneity/` | 모델별 최선 arm / 고정 최적 불가 분석 | 수집 완료 후 실행 |
| `exp04_heldout_transfer/` | held-out 전이 | 예정 |
| `exp05_mechanism_ppl/` | 이해×순응 인수분해, perplexity 스텔스 | 예정 (대체로 공간 무관) |

패널 실험(exp02-04)을 292 공간으로 돌리려면 16모델 × 2데이터셋 유해 재수집이 필요하다(공유 GPU
수일, 논문 수치 전면 갱신). 폐쇄모델(exp01)부터 끝내고 GPU 여유를 봐 착수 권장.

## 실행

```bash
export HF_HOME=/data1/users/ljk98/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/data1/users/ljk98/runtime_cache/vllm
export FLASHINFER_WORKSPACE_BASE=/data1/users/ljk98/runtime_cache/flashinfer
# 필요한 폐쇄모델 키는 셸 환경변수로 별도 주입한다. 키를 저장소나 명령 기록에 넣지 않는다.
VP=/data1/users/ljk98/envs/VLLM-VL-LABEL/bin/python

cd experiments_suite/exp01_closed_compare
$VP mj.py audit                 # 오프라인 점검
$VP mj.py probe --judge-device cuda:0   # 무해 세팅선택 (MultiJail)
$VP lg.py probe --judge-device cuda:0   # 무해 세팅선택 (Lingua)
$VP mj.py attack --judge-device cuda:0  # 유해 비교 (연구자 실행)
$VP lg.py attack --judge-device cuda:0
```

패널 수집기는 resident judge를 기본 16개씩 micro-batch한다. 메모리가 빠듯하면
`--judge-batch-size 8`로 낮춘다. 중단 후 같은 `matrix` 명령을 다시 실행하면 aggregate+raw가 모두
있는 arm은 건너뛰고 MANIFEST를 복구한 뒤 남은 arm만 실행한다.

수집물: 각 실험 폴더 `results/`에 `MANIFEST.json`, `benign/<tag>.json`(선택·shortlist),
`attack/<tag>__<method>.json`(verified/recon/unsafe), `attack/_raw/*.jsonl`(0600). 분석은 MANIFEST만
읽으면 되고 raw(유해)는 열 필요 없다.
