# 394 prior와 최신 MJ·LG 160조합의 대응

검토 기준: GitHub `dlwlsrnjs/ICLR-POLY` main, 커밋 `2e61055ee40b7d5ef22a0cd27844c0f5a7319343` (2026-09-20 확인). 실행 가능한 검사와 결과는 `audit_compatibility.py`, `COMPATIBILITY_AUDIT.json`에 있습니다.

## 결론

32개 이해 설정과 5개 프레임의 **요인 분해 prior**로 결합하는 것은 가능합니다. 하지만 이 394 수집을 기존 MJ·LG 160개 조합과 동일한 프롬프트·동일한 퍼즐 조건의 관측이라고 취급할 수는 없습니다. 네 legacy 프레임의 문구만 일치합니다.

## 최신 GitHub의 공간

`experiments_suite/exp02_panel_collect/collect_mj.py`와 `collect_lg.py`는 `_common.py` → `experiments_suite/common/engine.py` → `scripts/closed_compare.py`를 공유합니다. C 공간 정의:

- 이해 설정 32개: `g{3,5,8,12}` × `{ordered,shuffled}` × `n{2,4,6,8}`.
- 프레임 5개: `plain`, `persona`, `fiction`, `pap`, `persona+fiction`.
- C 격자 160개에 별도 baseline 4개를 더하면 164개입니다.
- 저장소에는 더 큰 full 공간 결과도 있습니다. 현재 Qwen7B MJ·LG 파일은 각각 292개(288격자+4 baseline)이며, 그 안에 C의 160개가 모두 포함됩니다. 최신 결과 전체가 160개뿐이라는 뜻은 아닙니다.

동일 모델의 32개 이해 관측도 각각 존재합니다:

- `experiments_suite/exp02_panel_collect/results/benign/qwen25_7b_mj.json`
- `experiments_suite/exp02_panel_collect/results/benign/qwen25_7b_lg.json`

각 파일의 `benign_recon_by_cell`은 32개 셀을 갖습니다. 두 데이터셋은 언어가 다르므로 MJ의 R과 LG의 R을 서로 바꾸면 안 됩니다.

## 프레임 대응

| GitHub C 공간 | 394 MiLMMT 수집 | 프레임 문구 자체 |
|---|---|---|
| plain | plain | 다름 |
| persona | legacy_persona | 정확히 같음 |
| fiction | legacy_fiction | 정확히 같음 |
| pap | legacy_pap | 정확히 같음 |
| persona+fiction | legacy_persona_fiction | 정확히 같음 |

`persona_archivist`, `persona_explorer`, `scene_single`, `scene_nested3`, `pap_logical`, `pap_alliance`는 같은 프레임으로 매핑하면 안 됩니다. 예를 들어 `scene_nested3`는 3중 게임 장면이고 기존 `fiction`은 5층 설정입니다.

프레임 문구가 같아도 전체 입력은 다릅니다. v3는 `GAME`, `PRESENTATION MODE`, `RESPONSE CONTRACT`, `ROUND`, 별도 영어 출력 system 지시를 사용합니다. 기존 C 격자는 `_compose()`의 head/reassembly/answer 지시와 `MIXED PARAGRAPH`를 사용합니다. `POLY_STRONG_RECON` 옵션은 grid wrapper를 추가 변경하므로 실제 실행 설정이 있다면 함께 확인해야 합니다. 원격 공개 집계 파일은 전체 raw prompt를 담지 않아 모든 과거 호출의 byte 일치까지 증명하지는 못합니다.

## 가장 큰 차이: g5의 의미

v3 `scenarios.py`는 영어 요청을 전체 5조각으로 나누고 위치별로 대응 언어 조각 하나씩을 선택합니다. 2언어든 4언어든 최종 입력은 **총 5조각**입니다.

반면 기존 `closed_compare._compose` → `combo_eval.mixed` → `run_qwen_interleaving_probe.build_puzzle`은 **각 언어의 전체 요청을 g조각씩** 나누고 모든 언어의 조각을 모두 넣습니다. 각 언어에 충분한 토큰이 있을 때:

| 표기 | v3 총 조각 | 기존 grid 총 조각 |
|---|---:|---:|
| 5조각 / 2언어 | 5 | g5_n2 = 10 |
| 5조각 / 4언어 | 5 | g5_n4 = 20 |

짧은 원문은 grid에서 언어별 조각 수를 줄일 수 있습니다. 따라서 기존 데이터의 g5_n4와 v3의 n=5, 4언어를 같은 셀로 취급할 수 없습니다. ordered의 정의, 분할 경계, shuffle seed와 표시 방식도 다릅니다.

v3의 혼합 언어는 English+Norwegian 및 English+Norwegian+Finnish+Arabic입니다. LG의 낮은 언어 수 프로필과는 언어 집합이 겹치지만, MJ는 English+Bengali(+Swahili+Javanese)를 우선 사용합니다. v3는 g3/g8/g12나 6/8언어의 완료 관측을 제공하지 않습니다. reverse는 C 격자에 없습니다.

## 32 × 5로 결합하려면

1. **모델을 맞춥니다.** 현재 394 관측은 Qwen2.5-7B 전용입니다. 다른 모델의 willingness를 직접 관측한 것처럼 사용할 수 없습니다.
2. **서로 다른 두 벡터를 구성합니다.** `R_dataset(c)`는 해당 데이터셋의 32개 이해 셀에서 가져오고, `W(f)`는 위 5프레임 대응만으로 별도 집계합니다. 현재 `prior_observations.json`은 111상황별 집계이지 5차원 벡터가 아닙니다.
3. **W의 조건을 고정합니다.** 단순히 raw/intact/혼합 퍼즐과 언어를 모두 합치면 이해 난이도와 번역 누락이 섞입니다. 한 가지 선택은 공통 394문항의 English intact 5프레임으로 W를 구성하는 것입니다. 이는 v3 게임 wrapper 안의 전체문장 응답이며 영어 raw와도 다릅니다. 퍼즐 응답을 사용한다면 공통 문항·동일 조건으로 맞추고 이해 성공과의 관계를 처리해야 합니다.
4. `P0(c,f) = R_dataset(c) × W(f)`는 **분리 가능성과 조건 간 전이를 가정한 초기 점수**로 사용할 수 있습니다. 보정된 성공확률이라는 보장은 없습니다. 특히 이미 재구성 실패가 섞인 퍼즐 수행률에 R을 다시 곱하면 이해 실패를 이중 반영할 수 있습니다. 거부하지 않았다는 지표도 요청 수행 성공과 같지 않습니다.
5. 현재 코드가 이 결합을 자동으로 수행하지는 않습니다. 동일 조건의 prior가 필요하다면 C의 실제 renderer·5프레임·dataset별 언어 프로필을 사용한 별도 무해 관측이 필요합니다. 이번 업로드에서는 수집 방식을 변경하거나 추가 GPU 추론을 실행하지 않았습니다.

English intact의 대응 5프레임은 각각 394개 응답과 394개 유효 WildGuard 판정을 갖습니다. 행동 유효 판정은 plain 159, legacy_persona 53, legacy_fiction 128, legacy_pap 57, legacy_persona_fiction 142개로 서로 다르므로 이 유효 집합만 비교할 때도 표본 차이를 확인해야 합니다.

전체 행동 유효 판정은 28,565개 중 12,370개입니다. 나머지를 수행/실패로 임의 채우거나 부분 거부를 자동으로 0.5 보상으로 확정하지 않습니다. WildGuard 비거부율을 W로 쓸지 행동 수행률을 쓸지에 따라 분모와 의미가 달라집니다.

코드 확인 근거: `scripts/closed_compare.py`의 `C_WILL`, `GRID_*`, `FRAME_HEAD`, `_compose`, `build_arms`; `scripts/run_qwen_interleaving_probe.py`의 `balanced_chunks`, `build_puzzle`; 이 보관본의 `scenarios.py`, `inputs/frames.json`, `configs/legacy_394.json`.
