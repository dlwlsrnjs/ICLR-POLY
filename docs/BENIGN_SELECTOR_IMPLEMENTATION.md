# 무해한 선택기 구현 및 재현

이 구현은 승인된 연구 계획의 첫 번째 CPU pilot이다. 기록된 FLORES 재구성 결과만 로딩하며 API 호출이나 GPU 생성은 하지 않는다.

## 실행

저장소 상위 poly 폴더에서:

    .vllm_env/bin/python -m unittest discover -s PolyJigsaw/tests -p 'test_benign_selector.py' -v
    .vllm_env/bin/python PolyJigsaw/scripts/run_benign_selector_experiment.py --outdir PolyJigsaw/results/NEW_RUN --seeds 7 17 27 --pretrain-steps 200 --rl-steps 400 --batch-size 64 --episodes 64 --threads 2

NEW_RUN은 아직 존재하지 않는 디렉터리여야 한다. 기존 실험 결과를 덮어쓰지 않는다.
환경: Python 3.10.12, NumPy 2.2.6, PyTorch 2.8.0. scikit-learn이나 신규 설치가 필요하지 않다.

## 코드 구성

- scripts/benign_selector.py: 자료 검증, 관측표 재생, 지도학습, terminal-reward actor-critic, 고정/무작위/GP-UCB 및 피드백 대조군.
- scripts/run_benign_selector_experiment.py: 고정된 모델·문항 분할과 예산으로 반복 실험. checkpoint, trace, manifest 및 보고서 생성.
- tests/test_benign_selector.py: 데이터 누출과 과소 비용 계측을 방지하는 테스트.

## 평가 범위

초기 2조건 × 4문항 = 8회 재생 평가. 추가 예산 0/1/2/4/8조건. Full grid는 40회 재생 평가다. 최종 평가용 문항 점수는 정책 선택 후에만 읽는다.
학습은 각 fold의 4개 대상 및 20개 문항만 사용한다. 제외 대상의 20개 문항에서 얻는 초기/추가 관측은 적응용으로 허용하며 비용을 센다. Validation 8문항과 test 12문항은 모든 번역/구성에 걸쳐 train/calibration 문항과 분리한다.
고정 및 1회 선택기는 예산 상한을 다 쓰지 않을 수 있다. Replay 비용과 실제 새 API 호출은 다르며 후자는 0이다.

첫 실행 결과는 results/benign_selector_pilot_20260903/FINDINGS_KO.md에서 확인한다.
