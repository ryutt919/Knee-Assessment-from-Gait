#!/usr/bin/env bash
# Resume-capable sequential runner (v3).
# 같은 <rundir>로 재실행하면 이미 끝난 스크립트는 SKIP하고 중단/실패 지점부터 이어간다.
# 완료 판정: $rundir/<name>.done 센티넬 OR $rundir/_progress.log 의 "DONE  <name>".
# 본 학습용(SMOKE 미설정), 풀코어 순차. 권장 실행:
#   ( nohup bash scripts/_run_resume.sh <rundir> > logs/_resume_boot.log 2>&1 < /dev/null & )
set -u
cd /Users/ryutt/Desktop/mini_ryutt/Walking/0611_journal_ML
source .venv/bin/activate
export TABPFN_TOKEN=""          # tabpfn auto-skips (no token)

RUNDIR="${1:?usage: _run_resume.sh <rundir>}"
mkdir -p "$RUNDIR"
P="$RUNDIR/_progress.log"

is_done() {  # <name> → 0 if already completed
  [ -f "$RUNDIR/$1.done" ] && return 0
  grep -q "DONE  $1 " "$P" 2>/dev/null && return 0
  return 1
}

run() {  # run <name> <script>
  local name="$1" script="$2"
  if is_done "$name"; then
    echo "===== SKIP  $name (already done) $(date +%H:%M:%S) =====" >> "$P"
    return
  fi
  echo "===== START $name $(date +%H:%M:%S) =====" >> "$P"
  if python "scripts/$script" > "$RUNDIR/$name.log" 2>&1; then
    touch "$RUNDIR/$name.done"
    echo "===== DONE  $name $(date +%H:%M:%S) =====" >> "$P"
  else
    echo "===== FAIL  $name (exit $?) $(date +%H:%M:%S) =====" >> "$P"
  fi
}

run 01  01_speed_ablation.py
run 02  02_raw_waveform_ml.py
run 03  03_shap_waveform.py
run 04b 04b_multiclass_optimal.py
run 04c 04c_speed_analysis.py
run 02b 02b_optimal_pipeline.py

echo "ALL SEQUENTIAL DONE $(date +%Y-%m-%d_%H:%M:%S)" > "$RUNDIR/_done.marker"
