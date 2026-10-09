#!/bin/bash
#SBATCH --job-name=pytest_cpu
#SBATCH --partition=day
# The suite takes a little over an hour on eight cores.
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/pytest_cpu_%j.log

# The full test suite on CPU. It does not fit under the login nodes' memory
# cgroup once JAX is loaded, so it runs as a job.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cpu
# The shared environment supplies the approved dependencies. PYTHONPATH puts
# this submitted worktree's source ahead of the checkout that owns the environment.
PYTHON=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/python
RUFF=/nfs/roberts/project/pi_nrs36/jar335/Repositories/tax-simulator-gpu/.venv/bin/ruff
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m pytest tests/ -q
"$RUFF" check src tests
