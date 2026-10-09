#!/bin/bash
#SBATCH --job-name=above_threshold_income
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/above_threshold_income_%j.log

# Report table 1: income above the top groups' own thresholds, and the federal
# tax on it, under current law (projects/top_tax/above_threshold_income.py).
#
#   sbatch jobs/above_threshold_income.sbatch <checkout> <output csv> [year]
#
# The checkout is the working tree whose code should run: the main repository
# or a git worktree.

set -euo pipefail

CHECKOUT="${1:?name the checkout (repository or worktree) whose code should run}"
OUTPUT="${2:?name the CSV file to write}"
YEAR="${3:-2027}"

if [ ! -d "$CHECKOUT/src/taxmodel" ]; then
  echo "$CHECKOUT is not a tax-simulator-gpu checkout: no src/taxmodel in it" >&2
  exit 1
fi

cd "$CHECKOUT"
PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python
export PYTHONPATH="$CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "checkout $CHECKOUT at $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

"$PYTHON" projects/top_tax/above_threshold_income.py --output "$OUTPUT" --year "$YEAR"
