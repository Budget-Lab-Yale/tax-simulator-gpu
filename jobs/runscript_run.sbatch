#!/bin/bash
#SBATCH --job-name=runscript_run
# The H200s sit in their own partition. The `gpu` partition holds L40S cards
# and refuses a job that asks for an H200, which is why the partition named
# here is `gpu_h200`.
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/runscript_run_%j.log

# Run every scenario a runscript names and write the run directory. This is what
# `taxmodel-run --slurm` submits; the whole-population projection does not fit
# under the login nodes' memory limit, so it always runs as a job.
#
#   sbatch jobs/runscript_run.sbatch <checkout> <runscript> [output root]
#
# The checkout is the working tree whose code should run — the main repository
# or a git worktree. It is required, so a job always runs the code it names
# rather than the code in the directory `sbatch` was called from.

set -euo pipefail

CHECKOUT="${1:?name the checkout (repository or worktree) whose code should run}"
RUNSCRIPT="${2:?name the runscript to run}"
OUTPUT_ROOT="${3:-}"

if [ ! -d "$CHECKOUT" ]; then
  echo "the checkout $CHECKOUT does not exist" >&2
  exit 1
fi
if [ ! -d "$CHECKOUT/src/taxmodel" ]; then
  echo "$CHECKOUT is not a tax-simulator-gpu checkout: no src/taxmodel in it" >&2
  exit 1
fi

cd "$CHECKOUT"
export PATH="$HOME/.local/bin:$PATH"

# The shared environment supplies the approved dependencies. PYTHONPATH puts
# the named checkout's source ahead of the checkout that owns the environment.
PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python
export PYTHONPATH="$CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
# Unbuffered output, so a job that dies part-way still logs its traceback.
export PYTHONUNBUFFERED=1

# Say in the log which tree and which commit actually ran, so a run that turns
# out to be surprising can be traced without guessing.
echo "checkout $CHECKOUT at $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

if [ -n "$OUTPUT_ROOT" ]; then
  "$PYTHON" -m taxmodel.run.cli "$RUNSCRIPT" --output-root "$OUTPUT_ROOT"
else
  "$PYTHON" -m taxmodel.run.cli "$RUNSCRIPT"
fi
