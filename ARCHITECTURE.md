# Architecture of the Tax-Simulator rebuild

This document describes how the model is built. The decision numbers
(D1, D4, ...) are cited from code comments and other documents, so they
are kept even where earlier decisions have been removed. The earlier
design history is in git history.

---

## 1. Purpose and hard requirements

The model is a microsimulation of the US federal tax system: individual
income, payroll, estate and wealth taxes, with behavioral feedback, over
30-year budget windows. It is written in Python on JAX and replaces the
Budget Lab's R Tax-Simulator.

Its uses, in priority order: production scoring for Budget Lab
publications; interactive analysis, with a reform scored in seconds;
grids and sweeps of many reforms; research that needs exact marginal
rates, which are derivatives of the tax calculation; and outside use by
researchers who hold the PUF and the Budget Lab's published dependencies.

Hard requirements:

- One codebase runs on a laptop CPU and on Bouchet's H200 GPUs. Model code
  makes no statements about hardware.
- Arithmetic is in double precision (D13).
- The code should read like what it simulates, so that an owner who did
  not write it can check it.
- Comments and documentation are written for a tax economist. The rule is
  in `CONTRIBUTING.md`.
- The law and behavior layers are editable by analysts with an R and
  economics background who are not JAX experts.

Correctness is checked by the unit tests (D12). The model began as a port
of the R model and was compared with it household by household; those
comparisons are retired. A smoothed version of a statute that the law
writes as a step, such as the child tax credit phase-out, is a change in
policy specification and is written in a law file.

## 2. Design rule

Taxing every household for one year under one law takes milliseconds on
a GPU and seconds on a CPU for the whole 221,831-household population,
and its derivative costs a few times that. The design therefore does not
add complexity to avoid calculator evaluations. Compilation, memory and
output are the real costs (D18, docs/PERFORMANCE.md). Sequential structure
exists only where the economics requires it: the stocks carried from one
year to the next, and the capital-gains selling decision, which reads the
future.

## 3. Decisions

### D1. Language and stack

The model core is JAX: compiled, vectorized over households and
scenarios, differentiated for marginal rates, with fixed array shapes and
branches written as `where`. Data preparation, law compilation and output
writing are ordinary Python with NumPy, pandas and pyarrow.

The jax and jaxlib versions are pinned in `uv.lock`, and a startup check
stops a run on any other version, because an earlier version miscompiled
the tax calculation (`config/toolchain.py`). The run manifest records the
versions and the device.

### D3. Data model: a frame of arrays and a law of arrays

- **Frame**: a dictionary of equal-length arrays, one entry per household
  column. `frame["wages"]` is an array over households.
- **Law**: compiled arrays indexed by year, filing status and schedule
  slot, padded to `MAX_BRACKETS` slots. The field inventory is
  `docs/compiled_law_schema.md`.
- All array shapes are settled when the law is compiled. Nothing looks up
  a schema while a run is simulating.

### D4. The law layer

Law is written in YAML parameter files, layered as `default/` and
`alternatives/` under `config/scenarios/tax_law/`, with indexation blocks
and filing-status mappers. The law compiler (`law/`) runs on the host
before simulation: it evaluates the mappers (a small expression language
of names, numbers, `+ - * /` and parentheses, `law/mappers.py`), applies
indexation with the statute's rounding rule (`law/indexation.py`), and
writes fixed-shape arrays. Every run also writes the resolved law as a
readable `tax_law.csv`.

- Every compiled field declares its dtype, units and whether it varies by
  filing status. Indicators, counts, calendar years and regime codes are
  marked as not differentiable (`law/compile.py`).
- Unused schedule slots are NaN. A calculator counts a slot as part of a
  schedule only where both the threshold and the rate are present
  (`taxes/calc/_schedules.py`).
- The lint checks in `law/lint.py` are errors, not warnings. Among them: a
  reform that replaces an indexed parameter without repeating its
  indexation must say `remove_indexation: true`; a schedule longer than
  `MAX_BRACKETS` is refused; indexation resumed after a freeze is refused.
- Marginal rates are derivatives with respect to the compiled annual
  arrays (rates, thresholds and amounts by year). They are not derivatives
  with respect to YAML base values, indexation choices or mapper
  expressions, which pass through rounding and other discrete steps.

### D5. Population preparation, frame building and the tax calculation

1. **Population preparation** (`frame/population.py`, `frame/prepare.py`,
   `frame/bridge.py`): reads Tax-Data, builds the record universe (D17),
   and supplies the random draws (D8). It does not depend on the law, runs
   once per run on the host, and is never compiled.
2. **Frame building** (`frame/build_frame.py`,
   `frame/steps_jax.build_world_frame`): the law-dependent frame steps
   (head-of-household recode, the law join, the state-tax workaround, the
   Social Security step, the capital-gains basis ratio, estate mortality,
   net worth), then the corporate and employer-payroll income adjustments
   for the reform worlds. These run inside the compiled year.
3. **The tax calculation** (`taxes/statutory_order.run_taxes`): calculator
   arithmetic only.

Behavior is a separate step that maps the mechanical frame to the
conventional frame (D9).

Repricing: a reform world's frame is built under the reform law,
including the law-dependent frame steps. The price of the same household
under baseline law swaps the law arrays and holds that frame fixed. Whether
a reform world should instead reuse the frame built under baseline law is
an open question.

### D6. The simulation core: five worlds a year

One calendar year is compiled once, and a plain Python loop calls it once
per year (see the docstring of `years/loop.py`). A scenario
that runs the capital-gains model runs that compiled year twice, with the
selling decision solved on the host between the two executions. The call
tree, with the file each step lives in, is in README.md under "How a run
works".

Each year evaluates five worlds, each one combination of household frame
and law (`years/worlds.py`):

| World | What it is |
|---|---|
| baseline | current law on the household as Tax-Data delivers it |
| first_order | the reform's law on that same household |
| mechanical_ex_saving | plus the corporate and employer-payroll income adjustments |
| mechanical | plus last year's wealth shortfall marked against the balance sheet; marginal rates are priced here |
| conventional | plus the scenario's behavior modules |

The differences between worlds:

| Difference | What it is |
|---|---|
| first_order − baseline | the reform's law effect |
| mechanical_ex_saving − first_order | the incidence of the corporate and employer-payroll income changes |
| mechanical − mechanical_ex_saving | the wealth drawdown, which the reported tables leave out |
| conventional − mechanical | behavior |
| conventional − baseline | the answer |

One more difference is computed and never reported: the mechanical frame
priced under reform law and under baseline law, which is the price change
behavior responds to.

- The baseline is an ordinary scenario, one more lane in the compiled
  year.
- There is no fixed-point iteration.
- Two stocks are carried from one year to the next: the wealth shortfall,
  kept separately for the mechanical and the conventional world
  (`years/wealth_state.py`), and the change in
  each age cohort's unrealized capital gains (`behavior/kg/stepper.py`).
- The capital-gains selling decision reads future tax rates, so it is
  solved on the host between the two executions (`years/gains_solve.py`,
  `behavior/kg/`).

### D8. Random draws

Every random column is derived from the run seed, the column's name and
the household's id (`frame/draws.py`). Adding or removing a column does
not move any other column's draws, and a household's draw does not depend
on its position in the file. The draws are computed once per Tax-Data
vintage by the bridge and stored as columns (`frame/bridge.py`).

### D9. Behavior modules

A behavior module is a plain JAX function. It takes the household
columns, the prices under the reform's law, the prices under baseline
law, its parameters and the household's random draws, and returns the
columns it changed. Each module carries a manifest naming its family, the
columns it reads and writes, and the marginal rates it reads.
`behavior/loader.py` checks every manifest against the frame before
anything compiles, runs the families in one fixed order written in that
file, and composes them into the function the year calls.

A module follows three rules so that it compiles: `where(cond, a, b)`
instead of `if` on data values, no change in array sizes, and no file
reads, writes or prints.

Every family runs in every scenario. A scenario without a response gives
the family its neutral parameter values, at which the module returns its
columns unchanged. Parameters are
in the module with citations; a calibrated value lives in a calibration
file (D20).

### D9a. Prices

A derivative needs one named number to differentiate. The run
differentiates four taxes under each law: individual income tax after
refundable credits; the same plus the employee's payroll tax; the annual
wealth tax; and the estate tax owed if the household dies this year. The
first two are stored for twenty-seven priced dollars, the last two for
each asset and debt column of the balance sheet. The list, the column
names and the reasons are in the docstring of `taxes/marginal_rates.py`.

- A response calibrated to the income tax alone reads the first rate; a
  response to the price a worker faces on a dollar of earnings reads the
  second. The choice is made by which column a module reads.
- Consumers read gradients by column name. The one compound price is the
  general partner's dollar, `part_active` plus `part_se1`, because the law
  taxes the same partnership dollar under two regimes.
- Where a response needs the add-a-dollar-and-recompute measure, or the
  change is too large for a derivative,
  `taxes/individual_income_tax.dollar_increment` provides it.
- Each distinct tax under each law is one reverse-mode pass.

### D11. Outputs

A run writes summary tables only: revenue totals, receipts and revenue
estimates, the per-tax totals files, the distribution table, the
marginal-rate table and the wealth-channel diagnostics. The layout of the
run directory is in the docstring of `run/output_directory.py`.
Per-household columns are written only when a runscript row names a
detail spec (`config/detail.py`). Those files are PUF-derived microdata
and never leave Bouchet or enter the repository.

### D12. Testing

The tests are under `tests/` and run as a Slurm job
(`jobs/pytest_cpu.sbatch`).

- Unit tests of the calculators, the law compiler, the frame steps, the
  behavior modules and the year loop.
- Stored chain outputs and marginal rates on 300 synthetic households,
  matched to a cent and to a millionth
  (`tests/unit/taxes/calc/test_pinned_chain_outputs.py`).
- A frozen compiled baseline law and indexation rounding cases
  (`tests/golden/fixtures/law/`).
- Synthetic households built to hit kinks and branches
  (`tests/synthetic/generator.py`).
- A check that every threshold is written with the model's corner
  convention (D22).

Nothing PUF-derived is stored in the repository.

### D13. Precision

Double precision everywhere, switched on at package import
(`taxmodel/__init__.py`). Single precision is not used, because
the decomposition columns are differences between nearly equal numbers.

### D17. The record universe

- One record universe per run: the union of household ids across every
  Tax-Data year the run reads.
- Each id keeps one slot in every year. A year in which a record does not
  exist is marked invalid and carries zero weight, so sums need no special
  case.
- Random draws depend on the id, not the slot (D8).
- The construction is in `frame/prepare.py`; its tests are in
  `tests/unit/frame/test_prepare.py`.

### D18. Execution

- The host layer (population preparation, law compilation, the
  capital-gains solve, output writing) is plain Python and is never
  compiled.
- The compiled programs are the projection year, which holds the baseline
  and the four reform worlds for one year, and the small programs that
  form the population totals, the tax at death spread to heirs, and the
  distribution and marginal-rate sums.
- Scenarios whose laws switch on the same code paths share one compiled
  year and run side by side in it as lanes. The number of lanes per
  program is chosen from measured memory costs (`run/group_size.py`). A
  laptop runs the same program with fewer lanes.
- On a GPU the whole population is placed on the device once; on a laptop
  it is read one year at a time (`years/device.py`).
- Compiled programs are kept in an on-disk cache, on by default
  (`run/compile_cache.py`).

### D20. Provenance and configuration

- A scenario is three configuration legs: tax law, economy and behavior
  (`config/`). Every configured number declares its kind: `calibrated`,
  `sourced`, `judgment` or `structural`, with the matching record (a
  citation, a note, or the conditions a calibrated value was derived
  under).
- A calibrated value records the data vintages it was derived under
  (`derived_under`) and the files whose change invalidates it
  (`invalidated_by`). A stale value stops the run unless the run is given
  `--accept-stale-calibration` (`config/staleness.py`).
- Every number fitted to data lives in a calibration file under
  `config/calibrations/`. A constant in the code is a model assumption and
  carries its citation.
- Every run writes a manifest: the resolved configuration, data vintages,
  calibrated values in force, the git commit with a flag for uncommitted
  changes, and the runtime stack (`config/manifest.py`).

### D22. Kink conventions

At an exact threshold the household is treated as having crossed it in
the direction of more income. The convention and how to write it are in
CONTRIBUTING.md, "Writing a threshold in a calculator", and in
`taxes/calc/_corners.py`.
