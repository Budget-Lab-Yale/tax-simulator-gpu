# Law compiler fixtures

Tax law parameters and published price and wage series. Nothing here comes
from the microdata, so these files are safe to keep in the repository.

## The frozen baseline law

| file | what it is |
| --- | --- |
| `compiled_baseline.npz` | one `[year, filing_status, slot]` array and one active mask per field |
| `compiled_baseline_schema.json` | the field inventory: dtype, units, slots used, indexed, mapper-built, differentiable |

Both are this repository's default law (`config/scenarios/tax_law/default`)
compiled for 2025 through 2036 with the index series in `indexes.csv`. The
calculator tests read them through the `compiled_baseline` fixture in
`tests/conftest.py`. Rebuild them from the repository root with

    PYTHONPATH=src python tests/golden/freeze_compiled_law.py

which also regenerates `docs/compiled_law_schema.md`. Rebuild whenever the
default law changes in a way the calculator tests should see, and update any
pinned test values that move.

## Fixed inputs

These were written once by the R model's law parser and are kept as fixed test
data. Nothing regenerates them.

| file | what it is |
| --- | --- |
| `indexes.csv` | CPI-U, chained CPI-U and AWI growth rates by year, with no VAT price offset. Tests read it as their index table. A model run derives the same table from the pinned Macro-Projections publication (`law/indexes.py`), and `tests/unit/law/test_index_table.py` checks that the derivation reproduces this file |
| `rounding_goldens.csv` | floor, ceiling and round-to-nearest at an indexation increment, on values that land exactly halfway. Ties round to even |
