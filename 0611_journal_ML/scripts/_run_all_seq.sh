#!/usr/bin/env bash
# Sequential full-core runner (v2 hardened rerun).
# Each script runs alone with all cores (no OMP/LOKY cap) → no oversubscription,
# fast with the Plain-boosting catboost. Launch via `( nohup ... & )` so it
# survives terminal/session reconnects (orphaned to launchd).
#
# Order: 01 → 02 → 03 → 04b → 04c → 02b   (03 needs 02's outputs)
# Per-script done markers let the monitor show progress; _done.marker at the end.
set -u
cd /Users/ryutt/Desktop/mini_ryutt/Walking/0611_journal_ML
source .venv/bin/activate
export TABPFN_TOKEN=""          # tabpfn auto-skips (no token)

RUNDIR="${1:?usage: _run_all_seq.sh <rundir>}"
mkdir -p "$RUNDIR"

run() {  # run <name> <script>
  local name="$1" script="$2"
  echo "===== START $name $(date +%H:%M:%S) =====" >> "$RUNDIR/_progress.log"
  if python "scripts/$script" > "$RUNDIR/$name.log" 2>&1; then
    echo "===== DONE  $name $(date +%H:%M:%S) =====" >> "$RUNDIR/_progress.log"
  else
    echo "===== FAIL  $name (exit $?) $(date +%H:%M:%S) =====" >> "$RUNDIR/_progress.log"
  fi
}

run 01  01_speed_ablation.py
run 02  02_raw_waveform_ml.py
run 03  03_shap_waveform.py
run 04b 04b_multiclass_optimal.py
run 04c 04c_speed_analysis.py
run 02b 02b_optimal_pipeline.py

echo "ALL SEQUENTIAL DONE $(date +%Y-%m-%d_%H:%M:%S)" > "$RUNDIR/_done.marker"
