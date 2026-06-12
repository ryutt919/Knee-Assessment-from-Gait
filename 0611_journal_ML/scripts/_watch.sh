#!/usr/bin/env bash
# 파이프라인 실시간 모니터 — 5개 스크립트 상태를 5초마다 갱신해서 보여준다.
# 사용법:  bash scripts/_watch.sh           (가장 최근 detached 런 자동 선택)
#          bash scripts/_watch.sh <로그디렉토리>
B=/Users/ryutt/Desktop/mini_ryutt/Walking/0611_journal_ML

resolve_dir() {
  # 1순위: 인자로 받은 경로
  if [ -n "$1" ]; then echo "$1"; return; fi
  # 2순위: 지금 실행 중인 프로세스가 쓰는 로그의 디렉토리 (가장 정확)
  local p f
  for p in $(pgrep -f 'scripts/0' 2>/dev/null); do
    f=$(lsof -p "$p" 2>/dev/null | grep -oE '/[^ ]*/logs/[^ ]*/[0-9][^ ]*\.log' | head -1)
    [ -n "$f" ] && { dirname "$f"; return; }
  done
  # 3순위: .log를 가진 가장 최근 디렉토리
  local d
  for d in $(ls -dt "$B"/logs/*/ 2>/dev/null); do
    ls "$d"/*.log >/dev/null 2>&1 && { echo "${d%/}"; return; }
  done
}

D="$(resolve_dir "$1")"
[ -z "$D" ] || [ ! -d "$D" ] && { echo "로그 디렉토리를 찾을 수 없습니다 (D=[$D])"; exit 1; }

while true; do
  clear
  echo "ACL 파이프라인 모니터  $(date '+%H:%M:%S')"
  echo "로그: $D"
  echo "────────────────────────────────────────────────────────────"

  # 실행 중 프로세스
  echo "[실행 중 프로세스]"
  ps aux | grep -E '[s]cripts/0' | awk '{printf "  %-30s %5.0f%%CPU %.0fMB\n",$12,$3,$6/1024}'
  [ -z "$(pgrep -f '[s]cripts/0')" ] && echo "  (없음 — 전부 종료됨)"
  echo ""

  # 각 스크립트 상태 + 최근 줄
  echo "[스크립트별 상태]"
  for f in 01 02 03 02b 04b 04c; do
    log="$D/$f.log"
    if [ ! -f "$log" ]; then st="—"; last="(로그 없음)"
    elif grep -qE 'Traceback|Error:|FAILED' "$log" 2>/dev/null; then st="❌실패"; last=$(grep -E 'Error|Value|Type' "$log" | tail -1)
    elif grep -qE 'Total elapsed|Elapsed:|완료\.|BEST:|FINAL RESULT|Figures saved|SHAP SUMMARY' "$log" 2>/dev/null; then st="✅완료"; last=$(grep -vE '^[[:space:]]*$' "$log" | tail -1)
    else st="🔄진행"; last=$(grep -vE '^[[:space:]]*$' "$log" | tail -1)
    fi
    printf "  %-4s %-7s %s\n" "$f" "$st" "${last:0:78}"
  done
  echo ""

  # CPU 요약
  echo "[CPU]"
  top -l 1 -n 0 | grep -E "CPU usage|Load Avg" | sed 's/^/  /'

  [ -f "$D/_done.marker" ] && echo "" && echo "=== 전체 완료: $(cat "$D/_done.marker") ==="
  echo ""
  echo "(Ctrl-C로 종료 · 5초마다 갱신)"
  sleep 30
done
