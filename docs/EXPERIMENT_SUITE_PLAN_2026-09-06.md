# 논문 실험 스위트 계획 (MIDAS + MultiJail/Lingua-SafetyBench 기반)

작성 2026-09-06. 방향: 강화학습 없음 = 오프라인 프라이어(자원-순서, 문헌 고정) + 온라인 적응
(퍼즐 해결=재구성 구동 그리디). 3축: 언어순서(고정) · 양 n(적응) · 혼란도 δ(적응).

## A. 이미 실행/진행됨

- **가족×크기 격자** (Qwen2.5 3/7/14, Llama-3 3/8, Gemma-2 2/9): 자원-순서 순차추가 완료
  (results/sequential_resource_20260906, 12모델). → 크기↑=재구성↑, 병목 재구성→정렬 이동.
- **양(n) 스케일링**: results/harmful_amount_20260905 (8모델). 강/약 해독기 두 레짐.
- **언어 선택 축**: results/harmful_weaklang_20260904. **자원-순서(MultiJail)** = 현재 기본.
- **혼란도 δ 스윕**: results/disorder_sweep_20260906 (진행 중, 가족×크기 7모델, n=4).

## B. MIDAS ablation 대응 (일부 구현됨 — 패널에 재실행 필요)

MIDAS Table 5(모듈 제거 시 ASR): w/o Game-framing이 최대 하락(80→22).
- **[구현] 게임 프레이밍 ablation**: interleave_nogame_prompt (run_polyjig_gated.py --with-nogame).
  "재조립 게임 지시" 유무 = 우리 방법 핵심 가설. → 가족×크기 패널에 재실행. **최우선.**
- **[구현] Dispersion/Multi-language ablation**: 단일 외국어 통번역 = translated_direct baseline
  (= 우리 n=1, 조각 없음). 이미 anchor로 수집됨. 정리만.
- **[대기] Granularity sweep** (MIDAS Fig.3a): fragments-per-language coarse(5) → fine.
  build_puzzle granularity 인자로 확장. → 신규 실행.
- **[선택] Persona/role induction** (역방향 ablation): 페르소나 층 추가 시 상승폭. 신규.

## C. MIDAS 평가 프로토콜 대응 (구현됨 — 패널에 실행)

- **[구현] HR(0–5) judge** (judge_harmfulness_rating.py): gated 성공에 완성도 0–5, ≥3 성공.
  → "뚫렸다"뿐 아니라 "얼마나 실행가능"까지. 최종 최적 설정에 실행.
- **[구현] 입력 detectability gap** (judge_input_detectability.py): interleaved 프롬프트를
  Qwen3Guard로 판정 → english_direct 대비 "safe 통과율" (MIDAS Fig.4a, 텍스트 채널 입증).
- **[구현] 방어 프롬프트 하 ASR** (--system-prompt self_reminder/todo_list): MIDAS Table 7/8.
- **[구현] cross-judge 일관성** (compare_judges.py): MD-Judge 교차, Cohen's kappa (MIDAS A.10).

## D. 사용자 요청 항목

- **모델 파라미터(크기)별 차이**: 가족×크기 격자로 커버됨(A). 크기↑=recon↑, 병목 이동 = 핵심 표.
- **모델 특성별 차이**: 지문 2D 회귀 — 정렬 강도(A: english_direct raw ASR) × 해독 능력
  (C: benign recon)로 4사분면 분류, 각 사분면의 최적 축(양 vs 혼란도) 매핑. → 분석 스크립트 신규.
- **seed 구분 실험 (robustness)**: 현재 40문항·단일 seed라 스텝별 노이즈 ±0.05. **핵심 설정
  (대상별 적응 선택 n, best config)을 3~5 seed로 반복 → mean±sd, 유의성**. → 신규, **최우선**
  (노이즈 우려 직접 해소, 그리디 정지 규칙의 안정성 근거).

## E. MultiJail/Lingua-SafetyBench 대응

- **자원 등급 순서** = 채택(현재 기본). MultiJail "저자원↑→unsafe↑"를 우리 joint(R AND U)에서
  재구성 게이트로 재검(저자원 Finnish는 recon 붕괴로 손해 = 확장 발견).
- **언어-tier 분석** (HRL/MRL/LRL별 joint): lang_rank_20260905 데이터에서 집계.
- **text-dominant subset**: 우리가 그 부분집합 사용 명시(Lingua-SafetyBench 분할).

## 실행 우선순위 (혼란도 이후 자동 큐)

1. **seed robustness** (3 seed, 핵심 설정) — 노이즈·안정성 방어. [최우선]
2. **게임 프레이밍 ablation** (가족×크기 패널) — MIDAS 최대 레버. [최우선]
3. **모델 특성 2D 분석** (정렬×해독 사분면 → 축 선택). [분석, GPU 불필요]
4. granularity sweep, HR judge, detectability, 방어프롬프트 — 최종 최적 설정에.
5. persona induction (선택).

모두 GPU1 순차. 혼란도 → seed → 게임ablation 순으로 끊김 없이 이어지게 큐잉.
