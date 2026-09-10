# PolyJigsaw: 무엇이 어디에 있고, 어떻게 다시 돌리나

이 문서 하나로 논문의 모든 수치를 다시 만들 수 있다. 코드와 집계 결과는 `scripts/`와 `results/`에
있고, 접근 제한 원본(유해 문항, 원시 생성물)은 복사하지 않되 해시로 대조할 수 있게 해 두었다.

전제: 파이썬 환경은 `/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python`, 작업 디렉터리는
`/home/ubuntu/342/jinkwon/poly/PolyJigsaw`. 결과 파일은 `jinkwon` 소유이므로 읽고 쓰려면
`sudo -n -u jinkwon` 를 앞에 붙인다.

---

## 1. 지금 상태에서 논문 수치만 다시 만들기 (GPU 불필요, 수 분)

생성물은 이미 `results/`에 있다. 아래는 그것들로부터 표·그림·매크로를 다시 만든다.

```bash
sudo -n -u jinkwon $VP/python scripts/audit_panel_integrity.py     # 90개 입력 파일 불변식 검사
sudo -n -u jinkwon $VP/python scripts/benign_prior_selection.py    # 무해 프로브 선정(LOO) → tab_sel_prior
sudo -n -u jinkwon $VP/python scripts/bandit_bootstrap_ci.py       # 밴딧 + CI (밴딧 모듈을 import 하며 함께 갱신)
sudo -n -u jinkwon $VP/python scripts/query_efficiency.py          # 쿼리 효율 곡선 + fig6
sudo -n -u jinkwon $VP/python scripts/probe_batch_size.py          # 프로브 배치 크기 민감도
sudo -n -u jinkwon $VP/python scripts/robustness_checks.py         # 저렴한 베이스라인·승자의 저주·계열 제거
sudo -n -u jinkwon $VP/python scripts/domain_analysis.py           # 도메인별 집계
sudo -n -u jinkwon $VP/python scripts/domain_permutation.py        # 도메인 순열 검정(음성 결과)
sudo -n -u jinkwon $VP/python scripts/domain_selector.py           # 도메인 적응 비용 비교
sudo -n -u jinkwon $VP/python scripts/make_paper_figs.py           # fig1..fig5
sudo -n -u jinkwon $VP/python scripts/make_selector_tables.py      # tab_sel_*.tex + selector_numbers.tex
```

그다음 빌드:

```bash
sudo -n -u jinkwon /tmp/polyjigsaw-typesetting/tectonic --untrusted \
  --outdir paper/build paper/polyjigsaw_iclr2026.tex
```

**순서가 중요하다.** `benign_prior_selection.py`가 프로브를 고르고, 밴딧이 그 선택을 읽는다.
프로브 선정을 건너뛰면 밴딧은 기본값(무해 재구성)을 쓴다.

## 2. 처음부터 다시 돌리기 (GPU 필요, 며칠)

### 2.1 유해 평가 — 구성 공간 23개 × 9모델 × 2데이터셋

```bash
python scripts/run_mj_gpu.py --gpu 0        # 7개 소형 모델, 단일 단계
bash   scripts/big_models_run.sh            # 32B·27B, 2단계(생성/판정 분리)
bash   scripts/lingua_triple_run.sh         # 역할분리 arm의 Lingua 판
```

산출: `results/{mj_sequential,mj_disorder,mj_combo,mj_method,mj_triple}_20260906/`,
`results/{sequential_resource,disorder_sweep,combo,method_baselines_v2}_20260906/`,
`results/lingua_triple_20260907/` — 모델당 하나의 JSON, 구성별 `gated`/`recon`/`unsafe` 평균.

### 2.2 무해 프로브 — warm start의 유일한 재료

```bash
$VP/python scripts/build_mj_benign_probe.py   # MultiJail 언어로 무해 문항 생성(NLLB)
bash scripts/benign_arms_run.sh               # Lingua 재구성 프로브 (소형 7)
bash scripts/benign_arms_big.sh               # Lingua 재구성 프로브 (32B·27B)
bash scripts/benign_arms_mj_run.sh            # MultiJail 재구성 프로브 (9)
bash scripts/benign_signals_run.sh            # 답변 프레임 신호, 평범한 무해 요청
bash scripts/benign_borderline_run.sh         # 같은 신호, 안전 경계에 가까운 무해 요청
```

산출: `results/benign_arms_{,mj_}20260907/`, `results/benign_signals_{,mj_}20260907/`,
`results/benign_borderline_{,mj_}20260907/`.

### 2.3 도메인 분해 — 문항별 판정 저장

```bash
bash scripts/domain_breakdown_run.sh       # Lingua, 9모델 × 5구성 × 250문항
bash scripts/domain_breakdown_mj_run.sh    # MultiJail 같은 구성
```

### 2.4 판정기 검증

```bash
bash scripts/judge_crosscheck_run.sh        # Qwen3Guard vs MD-Judge, 저장된 생성물 재판정
bash scripts/judge_crosscheck_rows_run.sh   # 번역 후 판정, 문항별 라벨
$VP/python scripts/mdjudge_template_check.py 64   # 템플릿 앞 개행 버그 재현
```

모든 러너는 idempotent다. 이미 있는 산출물은 건너뛰므로 중단 후 다시 실행하면 빈 곳만 채운다.

---

## 3. 코드 지도 — 무엇이 어떤 수치를 만드나

| 스크립트 | 만드는 것 | 논문 위치 |
|---|---|---|
| `arm_scoring.py` | 게이트 규칙. 평문 요청 arm은 면제, 숨긴 arm은 엄격 | §3 |
| `benign_prior.py` | warm-start prior. **무해 프로브만** 읽고, 없으면 예외 | §5 |
| `benign_prior_selection.py` | 프로브 후보 비교와 LOO 선정 | 부록 프로브 선정 |
| `mj_bandit_full.py` / `lingua_bandit_full.py` | arm 행렬, 오라클, 최선 고정, 예산 곡선 | 표 4 |
| `gp_bai.py` | 고정예산 best-arm identification | §5 |
| `bandit_bootstrap_ci.py` | 패널 부트스트랩 CI, paired Δ | 표 5 |
| `query_efficiency.py` | 탐색 전략별 쿼리 효율, fig6 | §6 베이스라인 |
| `robustness_checks.py` | 저렴한 적응 베이스라인, 승자의 저주, 계열 제거 | 표 robust |
| `probe_batch_size.py` | 프로브가 보는 문항 수 민감도 | 부록 |
| `domain_analysis.py` / `domain_permutation.py` / `domain_selector.py` | 도메인 집계, 순열 검정(음성), 적응 비용 | §6 도메인 |
| `make_selector_tables.py` | `paper/tab_sel_*.tex` 와 `selector_numbers.tex` | 전 표 |
| `make_paper_figs.py` | `results/figs/fig1..5` | 부록 그림 |
| `audit_panel_integrity.py` | 90개 입력 파일 불변식 + 표 수치 재도출 | 부록 감사 |

**본문이 인용하는 모든 수치는 매크로다.** `paper/selector_numbers.tex`, `query_numbers.tex`,
`domain_numbers.tex`, `robust_numbers.tex`가 생성기에서 나오고 본문은 그것만 참조한다. 손으로
숫자를 적지 말 것. 표와 본문이 어긋날 수 없게 하려는 장치다.

## 4. 반드시 지켜야 할 규칙 (과거에 어겼다가 고친 것들)

1. **prior는 무해 프로브에서만 온다.** 유해 실행의 `recon` 컬럼을 prior로 쓰면 안 된다. 검증 ASR이
   `recon AND unsafe`이므로 그 컬럼은 목표의 인수다. `benign_prior.py`가 유일한 진입점이고, 프로브가
   없으면 조용히 대체하지 않고 예외를 던진다.
2. **게이트 규칙은 구성에 따라 정한다.** 요청을 영어 평문으로 주는 구성(AIM, DeepInception, PAP,
   fiction-only)은 재구성 게이트를 일률 면제하고, 요청을 숨기는 구성은 엄격 적용한다. 출력에 태그가
   있었는지로 정하면 안 된다.
3. **MD-Judge 템플릿은 `.strip()` 해서 넣는다.** 앞 개행 하나면 판정이 무너진다(unsafe 0.03 대 0.56).
4. **저자원 답변은 번역 후 판정한다.** 직접 판정하면 같은 행에 거부와 유해가 동시에 붙는다.
5. **예산 곡선은 한 번만 계산해 출력과 저장에 같이 쓴다.** 예전에는 두 번 호출해 로그와 JSON이 달랐다.
6. **새 arm을 넣으면** `arm_scoring.CLEARTEXT_ARMS` 분류부터 하고, 1절의 재생성 순서를 그대로 다시 돈다.

## 5. 접근 제한 자료

복사하지 않는다. `PROVENANCE.txt`에 sha256을 남겨 두었으니 원본 보유자가 대조할 수 있다.

- `private_artifacts/multijail_v1/harm_grid.jsonl` (315행), `private_artifacts/panel_v2/harm_grid.jsonl` (250행)
- `private_artifacts/*/benign_probe.jsonl` — 무해 FLORES 문항. MultiJail 판은 NLLB로 만든 번역이고
  파일 안에 `translation_source` 필드로 기록돼 있다.
- `results/*/_raw*.json` — 원시 생성물. 판정 재현에는 필요하지만 유해 텍스트를 포함한다.

## 6. 보관본 만들기

```bash
bash scripts/make_release_bundle.sh /path/to/release_YYYYMMDD
```

코드, 집계 결과, 표, 그림, 논문, 그리고 제한 자료의 해시를 한 디렉터리로 모은다. 그 디렉터리에서
1절의 명령을 그대로 돌리면 GPU 없이 모든 표가 다시 만들어진다.
