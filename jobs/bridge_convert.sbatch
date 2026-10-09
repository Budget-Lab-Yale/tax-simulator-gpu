#!/bin/bash
#SBATCH --job-name=bridge_convert
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/bridge_convert_%j.log

# Convert one Tax-Data vintage into the form the model reads: one Parquet
# file per year with the wage split already done and the random draws already
# in columns, plus a manifest saying what was converted and what was wrong
# with it. Run once per vintage; every run afterwards just reads the result.
#
#   sbatch jobs/bridge_convert.sbatch [tax-data root] [output root]
#
# It parses every year of the vintage — about 18 GB of text — so it is a job
# and not a login-node command.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cpu
# The shared environment supplies the approved dependencies. PYTHONPATH puts
# this submitted worktree's source ahead of the checkout that owns the environment.
PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

SHARED=/nfs/roberts/project/pi_nrs36/shared/model_data
# The output mirrors the source layout one for one — vintage folder, then the
# Tax-Data scenario folder inside it — so pointing a run at the bridge instead
# of at Tax-Data is a change of the root and nothing else.
SOURCE=${1:-$SHARED/Tax-Data/v1/2026070814/baseline}
OUT=${2:-$SHARED/Tax-Data-Bridge/v1/2026070814/baseline}

"$PYTHON" -m taxmodel.frame.bridge "$SOURCE" "$OUT"
