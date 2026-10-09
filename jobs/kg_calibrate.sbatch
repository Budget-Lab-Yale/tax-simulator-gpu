#!/bin/bash
#SBATCH --job-name=kg_calibrate
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/kg_calibrate_%j.log

# Run the calibrate command for real: solve for the capital-gains model's two
# fitted numbers and write them to the configuration files.
#
# The job makes one pricing pass over the whole population under current law
# for thirty-one years, which is one compile and one execution, and then about
# ten Newton steps, each of which re-solves the realization schedule on 63 age
# cells and takes little time. It fits in the short development queue.
#
#   sbatch jobs/kg_calibrate.sbatch [worktree]

set -euo pipefail
# The first argument, when there is one, is the worktree to run in; anything
# after it is passed straight to the calibrate command.
if [ $# -gt 0 ]; then
  WORKTREE="$1"
  shift
else
  WORKTREE="$SLURM_SUBMIT_DIR"
fi
if [ ! -d "$WORKTREE" ]; then
  echo "the checkout $WORKTREE does not exist" >&2
  exit 1
fi
cd "$WORKTREE"
# Say in the log which tree and which commit ran, so a surprising result
# can be traced to the code that produced it rather than guessed at.
echo "checkout $WORKTREE at $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
export PATH="$HOME/.local/bin:$PATH"
# The shared environment supplies the approved dependencies; PYTHONPATH puts
# this worktree's source ahead of the checkout that owns the environment.
export PYTHONPATH="$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python

# The command writes the fitted values to config/calibrations/kg/realization.yaml
# and the record of the solve to derived_internals.yaml. Both are committed
# configuration files, so there is no output root.
"$PYTHON" -m taxmodel.calibration.cli "$@"
