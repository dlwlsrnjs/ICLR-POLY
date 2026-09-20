# 현재 실행 큐: 다중 난이도 곡선 → 전환점 확인 → 의지축

`adaptive_queue.py`가 현재 실행기다. 모델별 두 조건만 비교하던 이전 큐를 대체한다. 기존 GPU 단계는 끝까지 실행하고, 새 큐는 그 프로세스가 종료되면 자동 시작한다.

모델마다 다음을 연속 수행한다.

1. 기존 후보의 언어 수 n을 유지한 채 g=12,8,5,3을 ordered/shuffled 각각 수집한다. 총 8개 조건이며 선택용 100문항 중 번역 QA를 통과한 동일 문항만 비교한다.
2. 재구성과 실제 응답 거절을 판정한다. 조건 간 기울기는 재구성률 차이를 g 차이로 나눈 값이다. 연속한 세 점에서 좌우 기울기 차이의 문항 단위 bootstrap 구간이 0을 벗어나는 후보를 찾는다.
3. 기존 재구성 증가·거절 기준도 통과한 후보 중 선택용 기울기 변화가 가장 큰 후보를 고른다. 이 단계에서는 확인용 결과를 사용하지 않는다.
4. 선택한 세 조건을 확인용 231문항의 번역 적격 문항으로 검증한다. 재구성·거절 기준, 유효 데이터 비율, 같은 방향의 기울기 변화까지 확인되어야 설정을 고정한다. 실패하면 같은 확인용 결과로 다른 후보를 다시 고르지 않는다.
5. 확인된 모델은 전환 구간의 도달 조건에 이해축을 고정하고 의지축 5개를 수집·판정·집계한다. 전체 은행 결과와 선택용 문항을 제외한 결과를 모두 남긴다.
6. 전환점 미확인 또는 실행 실패는 기록하고 다음 모델을 진행한다.

**분석 범위:** 네 개 g 값에 대한 거친 격자상의 기울기 변화이다. 연속적인 정확한 임계점 추정이 아니다. 모델별 n은 이전 후보를 계승하므로 모든 32개 이해축 조건에 대한 완전 탐색도 아니다. 동일 331개 은행과 이전에 확인한 분할을 사용하는 탐색 분석이며, 완전히 손대지 않은 최종 테스트로 해석하지 않는다.

## 재시작과 결과 보존

- 모델별 단계가 끝날 때마다 `completed_stages/`에 명령·실행 파일 해시·결과 해시를 저장한다. 재시작하면 검증된 완료 단계를 건너뛴다.
- 중간에 멈춘 생성은 기존 수집기의 키/입력/응답 해시 검증 후 남은 요청만 이어간다.
- 기존 응답은 모델 revision, sampling, context, renderer, 전체 프롬프트, 메시지, 조각, 정답 순서가 일치할 때만 재사용한다. 출처를 `reuse.json`과 각 응답에 남긴다. 새 난이도 분석의 판정은 새 실행에 연결해 생성한다.
- `queue.lock`으로 중복 실행을 막는다. `status.json`과 `results.json`은 임시 파일 작성 후 교체한다.
- `PAUSE` 파일은 단계 사이 대기, `STOP` 파일은 단계 사이 종료다. 진행 중인 GPU 명령을 갑자기 끊지 않는다. 파일을 제거하고 같은 명령으로 재개할 수 있다.
- 오류 모델은 `failed`로 기록하고 다음 모델로 넘어간다. 같은 큐를 재시작하면 실패 모델을 다시 시도한다. `complete`는 해당 모델 파이프라인 실행 완료이며, 의지축 표본 충분성은 별도의 `willingness_status`로 확인한다.

## 실행

```bash
PYTHON=/path/to/.venv_prior085/bin/python
REPO=/path/to/ICLR-POLY
CODE="$REPO/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific"
"$PYTHON" "$CODE/adaptive_queue.py" \
  --run /path/to/adaptive_run \
  --old-run /path/to/willingness331_transition_refusal_run \
  --repo "$REPO" \
  --cache /path/to/hf_cache/hub \
  --aux-cache /path/to/aux_model_cache
```

`--old-run`은 기존 `TRANSITION_PLAN.json`, `inputs/items.json`, `inputs/accepted_translations.json`, 기존 모델 응답을 가진다. 필요하면 `--wait-pid`로 현재 GPU 단계의 종료를 기다릴 수 있다. 타깃 출력 한도 1,024토큰과 기존 모델·라이브러리 고정값을 계승한다. GPU 0만 사용하며 다른 사용자의 프로세스는 종료하지 않는다.

새 실행 디렉터리의 `QUEUE_PLAN.json`, `models/<model>/selection/curve_analysis.json`, `selected_candidate.json`, `confirmation.json`, `frozen_configs/`, `willingness/`, `results.json`에서 전체 선택 근거와 결과를 확인할 수 있다.

## 검증

뚜렷한 기울기 변화, 선형 곡선의 전환점 오인 방지, 거절 없는 변화의 배제를 검증하는 테스트 3개를 통과했다. 실제 Qwen7B 선택용 문항으로 8조건 720개 입력을 준비했으며, 기존 응답 180개가 정확 일치 검사 후 재사용됐다.
