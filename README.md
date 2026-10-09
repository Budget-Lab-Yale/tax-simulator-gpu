# tax-simulator-gpu

The Budget Lab at Yale's microsimulation model of the US federal tax
system. It replaces the lab's R
[Tax-Simulator](https://github.com/Budget-Lab-Yale/Tax-Simulator). The
model is written in Python with JAX, and the same code runs on a CPU or a
GPU.

The model calculates individual income, payroll, estate and wealth taxes
for every tax unit in the microdata, in each year of a scenario. It
compares current law with one or more reforms. It also includes the
corporate income tax and household behavioral responses, among them a
model of when households realize capital gains.

## Outputs

For each scenario, a run writes:

- revenue estimates and federal receipts by fiscal year
- totals for each tax by year
- a distribution table, by income group and by net worth group
- a table of average marginal tax rates

Revenue is reported both before and after households respond to the
reform.

## Input data

The model reads two datasets published by other Budget Lab models:
microdata from [Tax-Data](https://github.com/Budget-Lab-Yale/Tax-Data)
and economic projections from
[Macro-Projections](https://github.com/Budget-Lab-Yale/Macro-Projections).
Tax-Data is built from the IRS Public Use File, which cannot be
redistributed. This repository contains no microdata, and the model
cannot be run without a copy of Tax-Data.

Each Tax-Data release is converted once into the format the model reads
(`docs/TAX_DATA_BRIDGE.md`). The locations of the input data are set in
`config/output_roots.yaml`, which points at the Budget Lab's computing
cluster.

## Installation

The model requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync --locked                 # CPU
uv sync --locked --extra cuda    # NVIDIA GPU
```

## Running the model

A run is described by a runscript: a CSV file with one row per scenario.
Each row names a tax law, an economy, a set of behavioral responses, and a
range of years. The law and other settings are YAML files under
`config/scenarios/`, described in `config/scenarios/README.md`.

```
uv run taxmodel-run config/runscripts/rebate_2025.csv --dry-run
uv run taxmodel-run config/runscripts/rebate_2025.csv
```

`--dry-run` checks the runscript and the configuration without running
anything. `--slurm` submits the run as a Slurm job.

The tests are run with `uv run pytest`.

## Repository layout

- `src/taxmodel/` — the model
- `config/` — tax law, scenarios, calibrated parameters and runscripts
- `tests/` — unit tests
- `jobs/` — Slurm job files for the Budget Lab's cluster
- `docs/` — performance measurements, the format of compiled tax law,
  and the Tax-Data conversion
- `projects/top_tax/` — the scripts that produce the figures and the
  interactive data for the Budget Lab's report on taxes at the top

`ARCHITECTURE.md` describes how the model is built, and `CONTRIBUTING.md`
gives the rules for changing it.
