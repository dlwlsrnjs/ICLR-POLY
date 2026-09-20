# 331개 다국어 번역 은행 — 여기서부터 이어서 작업

사용자 최신 지시: **기존331개 다국어 번역을 먼저 확보하고, 사용자와 필요한 기준을 합의한 뒤 이 자료부터 이어서 진행한다.** 처음부터 전체 은행을 재번역하거나 이전 n2/n6안을 자동 확정안으로 취급하지 않는다.

## Hugging Face 위치

- 데이터셋: [jin-kwon/0920-prior-overrefusal331-multilingual](https://huggingface.co/datasets/jin-kwon/0920-prior-overrefusal331-multilingual)
- 고정 revision: `6553ceb5e0db99849b4b6521f503afa57f991639`
- 공개 범위: **비공개(private)**. `jin-kwon` 계정 또는 권한을 받은 계정으로 로그인해야 한다.
- 업로드 범위: 영어 원문331개, 7언어 번역/역번역2317쌍 전체, QA원본·판정 사유·프롬프트·메타데이터와 무결성 해시. 총25개 파일, 약26.8MB.
- 이 저장소의 `huggingface-upload.json`에 업로드 커밋과 원격 검증 결과를 기록한다.

| 언어 | 전체 | QA 통과 | 재검토 |
|---|---:|---:|---:|
| 아랍어 |331|287|44|
| 중국어 |331|286|45|
| 노르웨이어 |331|306|25|
| 핀란드어 |331|276|55|
| 벵골어 |331|209|122|
| 태국어 |331|221|110|
| 한국어 |331|228|103|
| 합계 |2317|1813|504|

영어 원문은 그대로 보존했다. 자동 QA통과와 사람이 검증한 의미 동등성은 다르다. 504개 실패쌍도 삭제하거나 통과로 바꾸지 않고 원본과 사유를 함께 올렸다.

## 내려받기

`huggingface_hub`가 설치된 환경에서 접근 권한이 있는 계정으로 로그인한 뒤:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jin-kwon/0920-prior-overrefusal331-multilingual",
    repo_type="dataset",
    revision="6553ceb5e0db99849b4b6521f503afa57f991639",
    local_dir="./0920-prior-translations",
)
```

현재 서버의 실험 가상환경은 기존 `ENVIRONMENT.md`와 같다. 프로젝트 전용 인증은 `HF_HOME=/home/ubuntu/342/jinkwon/hf_cache`에 있으며, 토큰 값은 코드·문서·채팅에 복사하지 않는다. 다른 환경에서는 본인의 로그인 설정을 사용한다.

```python
import hashlib, json
from pathlib import Path
root = Path("./0920-prior-translations")
for name, expected in json.loads((root / "SHA256SUMS.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
```

## 무엇을 사용할 것인가

- `data/english_originals.jsonl`: 원문331개, 출처 및 선별 이력.
- `data/translations_all.jsonl`: **전체2317쌍**. `qa_pass`와 `qa_reason`을 반드시 확인한다.
- `data/translations_qa_accepted.jsonl`: 자동 QA통과1813쌍.
- `data/translations_needs_review.jsonl`: 재검토504쌍.
- `provenance/`: 저장돼 있던 원본 번역, QA전체 raw기록과 설정. 같은 데이터를 중복 수집할 필요가 없다.
- Hub의 `train`은 파일 열람용 split 이름이며 이해축 실험의 선정/검증 분할이 아니다.

## 다음 담당자가 먼저 합의할 사항

1. **공통 언어 구성:** 영어 원문은 고정. n2/n4/n6 또는 다른 공통 구성 중 무엇을 비교할지 합의한다.
2. **번역 품질 기준:** 자동 QA실패504쌍의 표본을 살펴 실제 의미 훼손인지 과도한 판정인지 구분하고, 재검토/재번역 기준을 합의한다.
3. **공통 문항 마스크:** 합의한 모든 언어에 대해 사용할 문항을 정하고 전 모델에 동일하게 적용한다. 모델마다 탈락 문항이 달라진 결과를 직접 비교하지 않는다.
4. **이해축 탐색과 실제 수행 판정:** 같은 조건의 R(재구성), F(거절), Y(수행)를 구분한다. 기존Qwen32는 target응답의 수행 여부를 판정하는 중이며 여기의 번역QA와 별개다.
5. 합의 결과를 짧은 설정 파일/문서로 먼저 남기고, **이 기존 은행에서 필요한 실패쌍·누락분만 보완하여 이어서 진행**한다. 새 번역이 생기면 이전 버전을 덮어쓰지 말고 새revision과 변경 사유를 남긴다.

기존 n2보충 계약·후속큐는 이전 진행의 기록이며 이 합의 절차를 생략할 근거가 아니다. 실제 런에는 판정 뒤 수집을 자동 재개하는 감독 프로세스가 남아 있다. 이번 업로드는 그 프로세스를 변경하지 않았으므로, 후속 담당자는 [HANDOFF.md](HANDOFF.md)의 실제 큐를 먼저 확인하고 합의한 범위와 맞춰야 한다.
