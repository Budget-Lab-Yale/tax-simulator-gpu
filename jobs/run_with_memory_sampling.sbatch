#!/bin/bash
#SBATCH --job-name=memory_sampled_run
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/memory_sampled_run_%j.log

# One run with the host's memory sampled while it runs, so the log says which
# phase the peak belongs to. This is how host memory is measured before and
# after a change. It runs with no graphics card by default, which is the
# laptop case.
#
#   sbatch --mem=64G --export=ALL,RUNSCRIPT=config/runscripts/tests/laptop_gains_30yr.csv,\
#     OUTPUT_ROOT=/nfs/roberts/scratch/.../before,LABEL=before \
#     jobs/run_with_memory_sampling.sbatch
#
# The same run on a card:
#
#   sbatch --partition=gpu_devel --gres=gpu:1 --mem=48G --time=01:00:00 \
#     --export=ALL,PLATFORMS=default,... jobs/run_with_memory_sampling.sbatch
#
# Variables it reads, all optional:
#   RUNSCRIPT    the runscript to run
#   OUTPUT_ROOT  where the run directory goes
#   SRC          the source tree to run, so a worktree can be measured against
#                the main checkout
#   PLATFORMS    what JAX runs on: "cpu", or "default" to use a card
#   LABEL        a word for the log, so a pair of runs is readable afterwards
#
# The memory the job asks for is deliberately generous. The point is to measure
# what the run holds, not to find out whether it fits under a cap.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

RUNSCRIPT="${RUNSCRIPT:-config/runscripts/tests/laptop_gains_30yr.csv}"
SRC="${SRC:-$SLURM_SUBMIT_DIR/src}"
LABEL="${LABEL:-unlabelled}"

PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

# With no graphics card everything runs on the host, which is the laptop case.
PLATFORMS="${PLATFORMS:-cpu}"
if [ "$PLATFORMS" != "default" ]; then
  export JAX_PLATFORMS="$PLATFORMS"
fi

echo "label:      $LABEL"
echo "runscript:  $RUNSCRIPT"
echo "source:     $SRC"
echo "runs on:    $PLATFORMS"
echo "output:     ${OUTPUT_ROOT:-the default in config/output_roots.yaml}"
echo "memory cap: $(scontrol show job "$SLURM_JOB_ID" | grep -o 'mem=[0-9A-Za-z]*' | head -1)"
echo "started:    $(date +%H:%M:%S)"

ARGUMENTS=("$RUNSCRIPT")
if [ -n "${OUTPUT_ROOT:-}" ]; then
  ARGUMENTS+=(--output-root "$OUTPUT_ROOT")
fi

"$PYTHON" -u -m taxmodel.run.cli "${ARGUMENTS[@]}" &
RUN_PID=$!

# Sample the run's memory every ten seconds and print it beside the clock. The
# run's own progress lines carry the same clock, so reading the two together
# says whether the peak belongs to the population, a compile, or the
# arithmetic. VmHWM is the kernel's high-water mark for this process, the same
# quantity `sacct` reports as MaxRSS at the end.
(
  START=$(date +%s)
  while kill -0 "$RUN_PID" 2>/dev/null; do
    if [ -r "/proc/$RUN_PID/status" ]; then
      HWM=$(awk '/VmHWM/ {print $2}' "/proc/$RUN_PID/status" 2>/dev/null || true)
      RSS=$(awk '/VmRSS/ {print $2}' "/proc/$RUN_PID/status" 2>/dev/null || true)
      if [ -n "$HWM" ]; then
        printf 'memory at %5ds: now %6.2f GB, highest so far %6.2f GB\n' \
          "$(( $(date +%s) - START ))" \
          "$(echo "$RSS" | awk '{print $1/1048576}')" \
          "$(echo "$HWM" | awk '{print $1/1048576}')"
      fi
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
echo "ask sacct for the peak: sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS"
exit "$STATUS"
