#!/bin/bash
#SBATCH --job-name=bridge_bytes
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/bridge_byte_identity_%j.log

# Build every year's prepared population both ways — by parsing the Tax-Data
# text and by reading the bridge output — and compare every column exactly.
# The bridge's whole claim is that it changed the container and not the
# numbers, so this exits nonzero on a difference of any size.
#
#   sbatch jobs/bridge_byte_identity.sbatch [tax-data root] [bridge root] \
#          [--years first:last] [--window first:last]

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cpu
PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python
# The repository root is on the path too, for the comparison helpers in
# tests/unit/frame/population_checks.py.
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SHARED=/nfs/roberts/project/pi_nrs36/shared/model_data
TAX_DATA=${1:-$SHARED/Tax-Data/v1/2026070814/baseline}
BRIDGE=${2:-$SHARED/Tax-Data-Bridge/v1/2026070814/baseline}

"$PYTHON" tests/bridge_byte_identity.py "$TAX_DATA" "$BRIDGE" "${@:3}"
