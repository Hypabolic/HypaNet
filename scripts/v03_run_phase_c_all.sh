#!/bin/sh
# Run the whole pre-registered v0.3 Phase C schedule sequentially (experiments/v0_3/phase_c/README.md §3).
#
#   scripts/v03_run_phase_c_all.sh            # 11 runs, in README §3 table order, one process at a time
#
# Each run is scripts/v03_train_phase_c.py --budget B --seed K (output
# experiments/phase-c-trained/v0_3/<B>/seed<K>/). A run whose model.pt and model.sha256 already exist and
# whose sha256 matches is skipped, so the script can be restarted after an interruption; an interrupted run
# (no model.pt) is started again from scratch. Stops at the first failing run. Threads: 2 (registered value);
# PHASE_C_THREADS overrides it, and the trainer records the value in each training_log.json.
# Logs go to experiments/phase-c-trained/v0_3/logs/<B>_seed<K>.log (a restart appends).
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="$REPO/experiments/phase-c-trained/v0_3"
LOGDIR="$OUT_ROOT/logs"
THREADS="${PHASE_C_THREADS:-2}"
mkdir -p "$LOGDIR"
cd "$REPO"
START=$(date +%s)
for RUN in 32:0 32:1 32:2 128:0 128:1 128:2 512:0 512:1 512:2 2048:0 all:0; do
  B="${RUN%%:*}"; K="${RUN##*:}"
  D="$OUT_ROOT/$B/seed$K"
  if [ -f "$D/model.pt" ] && [ -f "$D/model.sha256" ]; then
    if (cd "$D" && shasum -a 256 -c model.sha256 >/dev/null 2>&1); then
      echo "skip budget $B seed $K (done: $D)"
      continue
    fi
    echo "refusing: $D/model.pt does not match model.sha256" >&2
    exit 2
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] start budget $B seed $K (threads $THREADS)"
  T0=$(date +%s)
  FORCE=""
  [ -f "$D/model.pt" ] && FORCE="--force"   # model.pt without a sha256: an interrupted write
  uv run python scripts/v03_train_phase_c.py --budget "$B" --seed "$K" --threads "$THREADS" $FORCE \
    >>"$LOGDIR/${B}_seed${K}.log" 2>&1 || { echo "budget $B seed $K failed; see $LOGDIR/${B}_seed${K}.log" >&2; exit 1; }
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done budget $B seed $K in $(( $(date +%s) - T0 ))s"
done
echo "all Phase C runs done in $(( $(date +%s) - START ))s"
