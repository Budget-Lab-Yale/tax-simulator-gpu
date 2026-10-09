#!/bin/bash
#SBATCH --job-name=null_reform_smoke
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/null_reform_smoke_%j.log

# The projection end to end on the real population: three years, full
# sample, baseline law scored against itself. Every channel of the reform's
# revenue effect must come out exactly zero and the wealth stock must stay
# empty.
#
# CPU on purpose: a null-reform check needs no GPU, and a CPU node is
# easier to get.
#
#   sbatch jobs/null_reform_smoke.sbatch [tax-data root] [first year] [last year]

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

TAX_DATA=${1:-/nfs/roberts/project/pi_nrs36/shared/model_data/Tax-Data-Bridge/v1/2026070814/baseline}
FIRST=${2:-2026}
LAST=${3:-2028}

PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" tests/null_reform_smoke.py "$TAX_DATA" "$FIRST" "$LAST"
