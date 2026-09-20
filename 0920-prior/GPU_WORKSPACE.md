# ETRI Slurm GPU 단일 작업공간

기록 시각: 2026-09-20 UTC. 현재 서버의 혼선을 막기 위해 원본을 이동하거나 복제하지 않고 아래 단일 인덱스에서 경로를 고정했다.

## 시작 위치

```bash
cd /home/ljk98/POLY/workspaces/prior_axes_mj_lg_20260920
source paths.env
./precheck.sh
```

작업공간 README:

`/home/ljk98/POLY/workspaces/prior_axes_mj_lg_20260920/README.md`

이 경로는 서버 로컬 경로라 GitHub에 결과 파일 자체를 올리지 않는다. 이 저장소에는 경로, revision, receipt 기반 상태만 기록한다. 대용량 원본은 symlink로 연결되어 있어 이동·중복·덮어쓰기가 없다.

## 네 가지를 구분하는 법

| 구분 | 의미 | 저장 위치 |
|---|---|---|
| 새 331 이해축 prior | 무해331에서 모델별 최초 재구성/과잉안전 지점 탐색 | `.../01_harmless331/runs/understanding_axis/` |
| 새 331 의지축 prior | 검증된 이해축 anchor에서 5개 frame 비교 | `.../01_harmless331/runs/willingness_axis/` |
| 기존 MJ/LG 이해축 view | 기존 harmful full grid의 `frame=plain`, 32조건 | `.../02_completed_mj_lg/views/by_axis/understanding_prior/` |
| 기존 MJ/LG 의지축 view | 같은 grid에서 anchor 고정 후 5 frame 비교 | `.../02_completed_mj_lg/views/by_axis/willingness_prior/` |

기존 MJ/LG의 이해축과 의지축은 별도 결과 복사본이 아니다. 하나의 160-arm grid를 서로 다른 관점으로 읽는다. 새 무해331 실험과도 섞지 않는다.

## 고정 입력과 완료 결과

### 무해331 번역 입력

- 로컬: `/home/ljk98/POLY/prior331_runs/translation_repair_20260920/hf_release_all2317_accepted_v3`
- Hub: `jin-kwon/0920-prior-overrefusal331-multilingual`
- revision: `8899f2bde0ae466dc9c0106cba8d3ae3abcb7722`
- 영어331개, 7언어2,317쌍, Qwen2.5-32B 의미 QA 2,317/2,317 통과, 재검토0
- 원격 재다운로드 후 SHA256 105개 항목 일치, 영어 원문 불변 확인

### 이해 조건 고정 의지 prior — canonical `92daeab`

- 로컬 canonical link: `/home/ljk98/POLY/workspaces/prior_axes_mj_lg_20260920/01_harmless331/willingness_prior_92daeab`
- 실제 로컬 디렉터리: `/home/ljk98/POLY/workspaces/prior_axes_mj_lg_20260920/01_harmless331/willingness_prior_fixed_g3_ordered_n2_20260920T123409Z`
- Hugging Face: `hf://buckets/jin-kwon/poly/PolyJigsaw/exp08_runs/willingness331_shared_en_ar_1024_run1/checkpoints/20260920T123409Z`
- 고정 이해 조건: `g3_ordered_n2` — English+Arabic, 언어당3조각, 언어 내 ordered
- 변화 의지 프레임: plain/persona/fiction/pap/persona+fiction
- 완료: collection/WildGuard/reconstruction/prior summary 모두17/17
- 적격 문항284개, 모델당1,420 jobs
- manifest SHA256: `b5205df806c8a005f0c1083906504adcc289c48b4d429b2662a5154a4651c941`; 416개 항목 전부 크기·SHA256 검증 통과

`llama32_3b_it`는 prior 파일까지 완결됐지만 `insufficient_common_reconstruction` 상태다. `92daeab` 문자열은 원격 경로나 manifest에 없으므로 사용자 지정 식별자로 기록하고, 실제 불변 식별에는 checkpoint timestamp와 manifest SHA256을 사용한다. 먼저 발견한 `20260920T095639Z`는 prior summary0/17인 중간본이라 사용하지 않는다.

### 기존 MJ/LG primary 5모델 완료본

- 불변 스냅샷: `/home/ljk98/POLY/snapshot_out/20260919_primary_large_complete_v1`
- 모델: Qwen2.5-14B/32B, Gemma2-27B, Mistral-Small-24B, Phi-3-medium-14B
- 생성·판정: MJ/LG 10쌍, raw452,000행, 1,600 arm, stale0, errors0
- 근거: `SNAPSHOT_MANIFEST.json`의 `run_complete: true`

### 기존 MJ/LG 분산 결과

- 로컬 shard/로그/receipt: `/home/ljk98/POLY/dist_full_grid`
- 원격 root: `hf://buckets/jin-kwon/poly/PolyJigsaw/full_grid_20260917_v1/`
- 업로드 및 readback 완료로 확정되는 receipt: 14개
- Qwen2.5-3B MJ는 50,400행 로컬 생성 및 snapshot manifest가 있으나 업로드가 98%에서 취소돼 receipt가 없다. 완료본으로 취급하지 않는다.
- 17모델×MJ/LG 34쌍 전체의 로컬 증거 상태는 [gpu-workspace.snapshot.json](gpu-workspace.snapshot.json)을 따른다.

이전 복원 체크포인트 `/home/ljk98/POLY/incoming-mj-lg/restored`는 참고·복원 전용이다. primary 5모델 최종 스냅샷 대신 사용하지 않는다.

## 현재 새 331 실험 상태

- 번역 입력: 완료 및 원격 검증 완료
- 이해축 prior: 실행 전
- 최종 8언어 기반 새 의지축 prior: 이해축 anchor 선택 전이므로 실행 전. 별도로 위의 기존 English+Arabic 고정-cell prior는17/17 완료본을 확보함
- GPU job: 이 문서 기록 시 사용자 job 0개, 전체 Slurm queue 0개
- 노드: h200-0, h200-1, rtx6000-0 모두 `idle`로 관측됨

GPU 추론은 로그인 노드에서 직접 실행하지 않는다. Slurm 파일은 작업공간 `03_gpu_runs/slurm/`, 로그는 `03_gpu_runs/logs/`, 새 출력은 `03_gpu_runs/outputs/`에 둔다. 토큰은 `/home/ljk98/POLY/.secrets/tokens.env`를 job 내부에서 읽고 코드·문서·로그에 값 자체를 남기지 않는다.

상태는 변할 수 있으므로 실행 직전에 `./precheck.sh`로 다시 확인한다.
