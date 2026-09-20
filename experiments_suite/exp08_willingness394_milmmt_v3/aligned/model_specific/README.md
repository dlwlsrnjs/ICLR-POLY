# 모델별 이해축 고정 후 331개 은행의 의지축 5개 수집

이 폴더는 기존 이해축 prior를 분석하고, 실제 수집 조건에서 모델별 후보를 재검증한 뒤 의지축 설정을 고정한다. **ASR이나 의지축 상관계수는 설정 선택에 사용하지 않는다.**

## 기존 결과에서 확인한 것

- FLORES 100문항 × 32조건 × 17모델의 재구성 집계 544행을 확인했다.
- MJ·LG 각각 24문항 × 32조건 × 17모델의 무해 prior도 확인했다. 빠져 있던 5모델은 기존 L40S 수집 디렉터리에서 복구했다. 출처와 SHA-256은 `inputs/provenance.json`에 있다.
- 두 데이터셋의 Wilson 95% 재구성률 하한 중 낮은 값을 최대화하고, 동률이면 조각 수가 적고 ordered인 셀을 선택한다. 이 기준으로 16모델은 `g3_ordered_n2`, Falcon3-7B는 `g3_ordered_n8`이 **후보**다. Falcon3-7B의 차이는 24문항 중 한 문항 수준이며 확정적인 우월성 근거가 아니다.
- `model_candidates.csv`는 모델별 재구성률·후보를, `analysis.json`은 선택 규칙·불확실성을 담는다. 모델마다 다른 값을 억지로 부여하지 않는다.
- plain 프레임에서 `g3_ordered_n2`와 `g12_shuffled_n8`을 비교하면, 전자의 ASR이 더 높은 경우가 MJ 16/17, LG 13/17이다. 이는 특정 두 셀의 기술적 비교이며 단조 관계나 인과관계 검정이 아니다. **쉬워지면 ASR이 급락한다는 공통 패턴은 확인되지 않았다.** 모든 32셀의 결과는 `asr_diagnostic.csv`에 있다.
- 기존 ASR은 유해 평가 세트의 재구성 게이트 성공률이다. 이 값만으로 이해 실패와 거부를 분리할 수 없다. 331개 은행에서는 무해 요청의 비거부율을 측정하며, 이를 ASR이라고 부르지 않는다.

## 실제로 고정하는 과정

1. 모델별 후보는 `g3_ordered_n2`, 기존 MJ/LG minimax 후보, FLORES 최고 재구성 후보의 합집합이다. 총 29개 모델×조건이며 모두 plain 프레임으로 100문항씩, 2,900응답을 생성한다.
2. n2는 **English + Arabic**이다. n4 이상은 공통 언어를 앞에 둔 `English, Arabic, Chinese, Norwegian, Finnish, Bengali, Thai, Korean`의 앞 n개 언어다. 이 새 구성에서는 n2/n4/n6/n8의 언어가 중첩된다. 과거 MJ·LG의 언어별 prefix와는 다르므로 재검증한다. MJ·LG 공통 언어는 3개뿐이므로 n4+를 두 데이터셋 공통 언어 조건이라고 주장하지 않는다.
3. MJ/LG의 고정된 프롬프트 렌더러를 사용한다. 원래 영어, 언어별 번역, 조각 ID/텍스트, 정답 순서, 전체 프롬프트, 응답, 종료 사유와 해시를 보존한다.
4. 고정된 해시 순서로 50문항을 선택용, 50문항을 검증용으로 나눈다. 선택용 재구성 성공 수가 최고에서 1개 이내인 셀 중 가장 단순한 것을 선택한다. 검증용 결과나 ASR을 보고 재선택하지 않는다.
5. 검증용 재구성률 ≥90%, Wilson 하한 ≥80%를 모두 충족하면 `validated_easy_anchor`로 표시한다. 미충족 모델도 최선의 가용 조건을 고정하되 `best_available_anchor_residual_comprehension_confound`로 표시한다. 판정 누락·파싱 오류가 있으면 고정하지 않는다.
6. 선택값을 `frozen_configs/<model>.json`에 남긴 뒤, 사용자 요청에 따라 331개 전부를 7개 외국어로 준비하며 MiLMMT로 번역·역번역하고 Qwen2.5-32B로 의미 보존을 판정한다.
7. 동일한 331개 원본 은행에서 번역 QA를 통과한 문항에 `plain, persona, fiction, pap, persona+fiction`의 다섯 프레임을 적용한다. 번역 탈락 문항은 5개 프레임 모두에서 제외하고 `blocked.json`에 남긴다. 원본 331개 전체는 삭제하지 않는다.
8. 모든 프레임에서 재구성에 성공하고, 정상 종료하고, 유효한 거부 판정이 있는 공통 문항으로 의지축을 추정한다. 최소 100문항 미만이면 준비된 prior로 표시하지 않는다.

**검증 범위:** 이 100개 영어 문항은 과거 prior에도 쓰였다. 따라서 50/50 분할은 이번 조건의 내부 검증이며 완전히 새로운 문항의 독립 검증은 아니다. 최종 331개에서도 프레임별 재구성률과 제외율을 반드시 함께 보고한다. 모델마다 언어·이해 셀·분석 문항이 달라질 수 있어 의지축 벡터의 모델 간 비교에는 이 차이가 남는다.

## 고정 실행 환경

기존 331개 prior에 실제 사용한 환경을 유지한다: vLLM 0.8.5, transformers 4.57.6, tokenizers 0.22.2, torch 2.6.0. 타깃 생성은 temperature 0, top_p 1, **max_tokens 1024**, seed 0, context 4096, BF16이며 모델 revision과 렌더러 해시는 기존 설정을 계승한다. 컨텍스트 초과 입력은 잘라서 실행하지 않고 실패 기록을 남긴다. 번역과 판정 모델의 출력 한도는 별도 설정이다.

로컬 실행 예:

```bash
PYTHON=/path/to/.venv_prior085/bin/python
REPO=/path/to/ICLR-POLY
CODE="$REPO/experiments_suite/exp08_willingness394_milmmt_v3/aligned/model_specific"
OUT=/path/to/new_run
"$PYTHON" "$CODE/analyze.py"
"$PYTHON" "$CODE/calibrate.py" prepare --run "$OUT" --repo "$REPO"
"$PYTHON" "$CODE/calibrate.py" run --run "$OUT" --repo "$REPO" --gpu 0 --cache /path/to/hf_cache/hub
```

`calibrate.py run`은 생성·재구성 판정·설정 고정까지 실행한다. `continue_collection.py`는 현재 로컬 작업의 `processes.json`을 읽어 번역 QA/재검증 종료를 기다린 뒤 추가 4개 언어 전체 번역, 의지축 수집, 판정, 집계를 이어가는 실행기다. 새 환경에서는 PID를 복사하지 말고 실제 실행 프로세스로 `processes.json`을 새로 만들어야 한다. `qa_existing.py`는 기존 Norwegian/Finnish/Arabic 993쌍 전용이고, `qa_languages.py`는 추가 언어의 전체 331문항 번역 전용이다.

## 남는 결과

실행 디렉터리에는 다음을 보존한다.

- `calibration/<model>/{manifest.json,jobs.jsonl,responses.jsonl,collection_metadata.json,collection.log}`
- `reconstruction_inputs.jsonl`, `reconstruction/restricted_reconstruction_audit.jsonl`, `anchor_decisions.json`, `frozen_configs/`
- `qa/`, `extra_translations/`, `extra_qa/`: 원문·번역·역번역·전체 QA 프롬프트·판정 원문·해시
- `willingness/<model>/`: 전체 5프레임 입출력, 차단 문항, 재구성 판정, WildGuard 판정, `willingness_prior.json`
- `panel_summary.json`: 모델별 문항 수준 5×5 공분산·상관계수와 모델 평균 벡터 간 5×5 공분산·상관계수. 분산이 0이면 상관계수는 null이다. 특정 상관계수(예: 0.6)를 목표로 데이터를 고르지 않는다.
- `status.json`, `continuation_status.json`, `collection_progress.json`: 현재 단계와 오류. 실행 중인 상태를 완료 결과로 해석하지 않는다.

기존 영어+아랍어 실행 결과와 새 모델별 설정 결과는 섞지 않는다. 이 폴더의 `RUN_SNAPSHOT.json`은 업로드 시점의 진행 기록이고 실시간 상태는 로컬 실행 디렉터리에서 확인한다.

## 검증

```bash
cd "$CODE"
python -m unittest -v test_protocol
cd ..
python -m unittest -v test_aligned
```

모델별 선택/검증 분리와 파싱 실패 차단을 포함한 신규 테스트 4개, 기존 렌더러·입력 보존·집계 테스트 17개가 통과했다. 준비한 2,900개 퍼즐은 모든 언어의 조각을 정답 순서로 합쳤을 때 원래 문장과 일치함을 확인했다(공백 정규화 기준).

## 번역 은행 확대

최신 요청에 따라 모델별 선택값과 무관하게 영어+7개 외국어(n8)까지 331개 전부를 준비한다. 기존 Norwegian/Finnish/Arabic 993쌍은 같은 MiLMMT revision의 결과를 재사용하며, Chinese/Bengali/Thai/Korean 1,324쌍을 추가 생성한다. 영어 원문은 번역 결과로 대체하지 않는다. 중국어는 번역 프롬프트에서 공식 명칭 `Chinese (Simplified)`를 쓰고 퍼즐 설정에서는 `Chinese`로 기록한다. 새 번역 출력 한도는 1,024토큰이다.

모델 선택 근거와 한계는 `TRANSLATION_MODEL_SELECTION.json`을 참조한다. [공식 모델 카드](https://huggingface.co/xiaomi-research/MiLMMT-46-12B-v1.0)는 모든 대상 언어를 지원하며, [개발진 논문](https://arxiv.org/abs/2608.10812)은 TranslateGemma 등을 포함한 비교 평가를 보고한다. 이 은행의 의미 보존은 별도의 Qwen32 QA로 확인하며 벤치마크 성능만으로 통과 처리하지 않는다.

최종 언어 구성은 `English, Arabic, Chinese, Norwegian, Finnish, Bengali, Thai, Korean`이다. 영어·아랍어·중국어는 두 데이터셋 공통이며, 전체 구성은 LG와 5개, MJ와 6개가 겹친다. Swahili는 MJ에 있지만 MiLMMT의 공식 지원 언어가 아니므로 Thai를 선택했다. `LANGUAGE_PROFILE.json`에 n별 실제 목록을 고정했다. 이전 LG 전용 후보 검증 결과는 별도 실행 디렉터리에 보존하며, 완전히 같은 영어+아랍어 입력 600개만 `reuse_exact_n2.py`의 모델·프롬프트 일치 검사를 거쳐 재사용했다. n4+ 응답은 새 언어 구성으로 다시 생성한다.

현재 머신에서는 다른 사용자의 GPU 1 서버가 재기동했으므로 MiLMMT는 GPU 1에서 batch/메모리 설정을 조정해 실행하고, 추가 번역 QA와 최종 의지축 판정은 재구성 검증 종료 후 GPU 0에서 수행한다. `judge_local.py`는 기존 판정기를 유지하면서 GPU 인덱스만 선택 가능하게 만들며, 복사한 판정 코드의 해시를 기록한다.
