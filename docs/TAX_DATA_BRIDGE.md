# The Tax-Data bridge

Written as the brief before the code existed; it is now the design of record for code that is built and
in use — `src/taxmodel/frame/bridge.py`, converted by
`jobs/bridge_convert.sbatch` and checked by `jobs/bridge_byte_identity.sbatch`.
Two things came out differently from what the brief expected, and both
are marked below where they arise: net worth and the record universe
stayed in the model.

## The problem

Tax-Data hands the model 200 MB of CSV per year. Parsing that text is the
slow part of starting a run, and because it was slow we built machinery to
compensate: a saved copy of the parsed population, staleness checks on
that copy, and a rebuild job. None of that is tax modeling. On top of
that, Tax-Data's wage columns overlap (an earner's wages include their
tips and overtime), so the model splits them into exclusive pieces at
read time — work that is upstream's to do, living downstream.

## The fix

A bridge: a script that converts one Tax-Data vintage, once, into the
interface the model actually wants, stored where the lab keeps model
interchange data. A run reads only the converted form. When the
Tax-Data build itself is reworked to emit this shape, the bridge is deleted and the model's
input path points at real Tax-Data — nothing else in the model changes.

```
Tax-Data CSVs                       one bridge run per vintage
  shared/model_data/Tax-Data/v1/<vintage>/tax_units_<year>.csv
      |
      v
  parse -> wage split -> random draws
      |
      v
  shared/model_data/Tax-Data-Bridge/v1/<vintage>/
      tax_units_<year>.parquet     one per year, exclusive columns
      manifest.yaml                what was converted, by what code, with
                                   what warnings
      |
      v
  MODEL, per run: read columns -> scenario steps (law, economy) -> pad -> jit
```

## What moves into the bridge

Everything that is true of a vintage regardless of any scenario:

- **CSV parsing.** The bridge is the only thing a run uses to read
  Tax-Data text. The model's own CSV reader was not deleted: it is the
  leg the byte-identical comparison below is made against, and it stays
  until Tax-Data emits this shape upstream.
- **The wage split** into mutually exclusive per-earner pieces
  (wages_base, tips, overtime; per-earner sole prop and farm). The ~24
  records per year whose tips plus overtime exceed their wages, a Tax-Data
  defect, are carried with negative base wages exactly as
  today; the manifest records their count per year so the warning is
  permanent and visible instead of scrolling past in a log.
- **The random draws.** They are a deterministic function of household
  id, purpose, and seed — the same numbers in every run by design —
  so they are data, and they belong in the parquet columns.

## What stays in the model, on purpose

- Every step that needs the law or the scenario's economy: the law join,
  the head-of-household recode, the SALT workaround, the Social Security
  COLA, the capital adjustment and capital-gains CPI ratio, estate
  mortality. That IS scenario analysis.
- Padding to fixed record counts, which depends on which years a run
  spans (the GPU compiles one graph for all years).
- **The record universe** — which households exist in which years. The
  brief put this in the bridge. It stayed in the model
  (`frame/prepare.build_id_universe`) because the universe is the union
  over the years a *run* names, which is scenario information; the bridge
  writes one file per year and knows nothing about a run's window.
- **Net worth.** The brief called this plain arithmetic on balance-sheet
  columns and put it in the bridge. It stayed in the model
  (`frame/steps.add_net_worth`) because it is not fixed data: the
  incentives layer differentiates through it, and the wealth avoidance
  module replaces it. A column a scenario can change cannot be vintage
  data.

## What gets deleted from the model

The saved-population machinery exists only because parsing was expensive:
the write/read of cached population slices, the population arm of the
staleness checks, and the standalone population-preparation job. After
the bridge, the "cache" is the bridge output itself — versioned by
vintage directory, inspectable with any parquet reader.

## Why parquet and not something rawer

Parquet stores each column as a contiguous block of typed binary numbers:
reading it is a memory copy, not text parsing, and feeding a column to
the GPU is one more copy. A raw numpy format would be marginally faster
still, but parquet is readable by R during the dual-run period, is
inspectable with standard tools, and is the format the reworked Tax-Data
will emit — so the temporary interface and the final one are the same
shape, and repointing later really is just a path change.

## Proof obligations (met when the bridge was built)

1. **Byte-identical frames.** For the current vintage, build the frame
   the old way (CSV path) and the new way (bridge path) and compare every
   column of every year exactly. Zero tolerance — the bridge is a change
   of container, not of numbers.
2. **Full test suite green** in a Slurm job.
3. **One full oracle re-grade** (the reform parity array plus baseline)
   run from the bridge path, matching the signed M1 footprint.
4. The bridge run itself is a Slurm job; the manifest records the source
   vintage path, the bridge code version, row counts per year, and the
   negative-base-wage count per year.

## Sequencing (done)

The bridge was built after the M2 merges, for the reason recorded at the
time: the engine wiring and the sampling tear-out touched the same files.
R was unaffected throughout and still reads the CSVs it always read; the
old CSV shape retires when R does.
