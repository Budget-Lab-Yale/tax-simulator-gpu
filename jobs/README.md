# Slurm job files

Every `.sbatch` file here submits one job. Read the comment block at the
top of a file for what it runs and what arguments it takes.

Job output does not stay in the project tree. Every job writes its
`--output` log to

    /nfs/roberts/scratch/pi_nrs36/jar335/slurm_logs/

alongside the run outputs on scratch. Slurm will not create a missing
log directory, so that directory has to exist before a job is submitted;
it does today.

Every job finds Python the same way: the main checkout's `.venv`
supplies the approved dependencies, and `PYTHONPATH` puts the submitted
worktree's `src/` ahead of it, so a job can run against work that is
not on master yet.
