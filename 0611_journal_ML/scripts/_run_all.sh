#!/usr/bin/env bash
# Detached parallel runner for the side-fixed multi-model rerun.
# Launched via `setsid nohup` so it survives terminal/session reconnects.
# Writes per-script logs to $RUNDIR and a _done.marker when all finish.
set -u
cd /Users/ryutt/Desktop/mini_ryutt/Walking/0611_journal_ML
source .venv/bin/activate

export TABPFN_TOKEN="" \
       OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       LOKY_MAX_CPU_COUNT=2

RUNDIR="${1:?usage: _run_all.sh <rundir>}"
mkdir -p "$RUNDIR"

python scripts/01_speed_ablation.py      > "$RUNDIR/01.log"  2>&1 &
( python scripts/02_raw_waveform_ml.py   > "$RUNDIR/02.log"  2>&1 \
  && python scripts/03_shap_waveform.py  > "$RUNDIR/03.log"  2>&1 ) &
python scripts/02b_optimal_pipeline.py   > "$RUNDIR/02b.log" 2>&1 &
python scripts/04b_multiclass_optimal.py > "$RUNDIR/04b.log" 2>&1 &
python scripts/04c_speed_analysis.py     > "$RUNDIR/04c.log" 2>&1 &
wait

echo "ALL PARALLEL DONE $(date +%Y-%m-%d_%H:%M:%S)" > "$RUNDIR/_done.marker"
