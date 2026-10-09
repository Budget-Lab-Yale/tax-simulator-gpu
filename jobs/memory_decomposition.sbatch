#!/bin/bash
#SBATCH --job-name=memory_decomposition
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/memory_decomposition_%j.log

# One run with no graphics card, its memory sampled from outside every ten
# seconds and reported from inside at each named phase. This is how the peak
# host memory of a run is broken into named holders.
#
#   sbatch --export=ALL,RUNSCRIPT=config/runscripts/tests/laptop_gains_30yr.csv,\
#     LABEL=cold,CACHE_DIR=/path/to/an/empty/directory jobs/memory_decomposition.sbatch
#
# Variables it reads, all optional:
#   RUNSCRIPT    the runscript to run
#   LABEL        a word for the log and the output directory
#   CACHE_DIR    the compile cache directory; empty means a cold compile
#   LEDGER       "on" (the default) prints the phase ledger
#   PYTHON_HEAP  "on" traces Python-level allocations, which is slow
#   TRIM_EVERY_YEAR  "on" returns the C library's freed memory after each year
#   OUTPUT_ROOT  where the run directory goes
#
# The run sets the C library's allocation thresholds itself at startup on a
# machine with no graphics card, so a run measured here already has them.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

RUNSCRIPT="${RUNSCRIPT:-config/runscripts/tests/laptop_gains_30yr.csv}"
LABEL="${LABEL:-unlabelled}"
EXPERIMENT=/nfs/roberts/scratch/pi_nrs36/jar335/experiments/laptop_2026-09-04/memory_decomposition
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT/$LABEL}"

PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export TAXMODEL_MEMORY_LEDGER="${LEDGER:-on}"
export TAXMODEL_MEMORY_PYTHON_HEAP="${PYTHON_HEAP:-off}"
export TAXMODEL_MEMORY_TRIM_EVERY_YEAR="${TRIM_EVERY_YEAR:-off}"
if [ -n "${CACHE_DIR:-}" ]; then
  mkdir -p "$CACHE_DIR"
  export TAXMODEL_COMPILE_CACHE="$CACHE_DIR"
fi

echo "label:      $LABEL"
echo "runscript:  $RUNSCRIPT"
echo "cache:      ${CACHE_DIR:-the default}"
echo "output:     $OUTPUT_ROOT"
echo "node:       $(hostname)"
echo "started:    $(date +%H:%M:%S)"

"$PYTHON" -u -m taxmodel.run.cli "$RUNSCRIPT" \
  --output-root "$OUTPUT_ROOT" &
RUN_PID=$!

# The kernel's own view of the run, sampled from outside every ten seconds.
# VmHWM is the high-water mark `sacct` reports as MaxRSS.
(
  START=$(date +%s)
  while kill -0 "$RUN_PID" 2>/dev/null; do
    if [ -r "/proc/$RUN_PID/status" ]; then
      awk -v elapsed="$(( $(date +%s) - START ))" '
        /VmRSS/ {rss=$2} /VmHWM/ {hwm=$2}
        END {printf "memory at %5ds: now %6.2f GB, highest so far %6.2f GB\n",
             elapsed, rss/1048576, hwm/1048576}' "/proc/$RUN_PID/status" || true
    fi
    sleep 10
  done
) &
SAMPLER_PID=$!

set +e
wait "$RUN_PID"
STATUS=$?
set -e
kill "$SAMPLER_PID" 2>/dev/null || true
echo "finished:   $(date +%H:%M:%S), exit status $STATUS"
exit "$STATUS"
