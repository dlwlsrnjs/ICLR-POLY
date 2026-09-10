#!/bin/bash
# gemma-2-27b 다운로드 완료 → 27B arm 10개 채우기 → 9모델 패널 확장까지 자동 연결.
# 각 단계는 idempotent. 중단하려면 이 프로세스를 kill 하면 됨.
set -u
cd /home/ubuntu/342/jinkwon/poly/PolyJigsaw || exit 1
HF=/home/ubuntu/342/jinkwon/hf_cache
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "1/4 27B 다운로드 대기"
while pgrep -u jinkwon -f 'hf download google/gemma-2-27b-it' > /dev/null; do sleep 60; done

SNAP=$(ls -d "$HF"/hub/models--google--gemma-2-27b-it/snapshots/*/ 2>/dev/null | head -1)
for f in tokenizer.json tokenizer.model model.safetensors.index.json; do
  [ -e "$SNAP$f" ] || { log "중단: $f 없음 — 다운로드가 불완전합니다"; exit 1; }
done
NSHARD=$(ls "$SNAP"model-*.safetensors 2>/dev/null | wc -l)
log "다운로드 검증 OK (샤드 $NSHARD개, $(du -sh "$HF"/hub/models--google--gemma-2-27b-it | cut -f1))"

log "2/4 진행 중인 big_models 종료 대기"
while pgrep -u jinkwon -f 'big_models_run.sh' > /dev/null; do sleep 60; done

log "3/4 big_models 재실행 — 27B arm 10개 생성"
mv -f logs/big_models.log logs/big_models_run2.log 2>/dev/null
bash scripts/big_models_run.sh > logs/big_models.log 2>&1
log "big_models 종료"

MISSING=0
for d in mj_sequential mj_disorder mj_combo mj_method mj_triple \
         sequential_resource disorder_sweep combo method_baselines_v2 lingua_triple; do
  [ -f results/${d}_20260906/gemma2_27b.json ] || [ -f results/${d}_20260907/gemma2_27b.json ] || {
    log "  누락: ${d}/gemma2_27b.json"; MISSING=1; }
done
[ "$MISSING" = 1 ] && log "경고: 27B arm 일부 누락 — 워처가 대기 상태로 남습니다"

log "4/4 panel9 워처 기동 (밴딧 재학습 → 부트스트랩 CI → 논문 그림)"
rm -f results/PANEL9_DONE.marker
bash scripts/panel9_watcher.sh > logs/panel9_watcher.log 2>&1
log "체인 완료"
