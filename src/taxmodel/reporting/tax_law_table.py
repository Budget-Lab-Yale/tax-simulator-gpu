"""`tax_law.csv`: the resolved law each scenario ran, by year.

Every scenario writes out the resolved year-by-parameter law it ran under, so
a reader can look up the standard deduction in 2031 in this run without
re-reading the YAML tree and re-doing the indexation.

The table has no `world` column. The five worlds differ in the household frame
the tax is calculated on, not in the law: the `baseline` world runs baseline
law and the other four run the reform's law. The `law` column says which of
the two a row belongs to. A scenario that names no reform folder runs the same
law twice, and its two blocks of rows are identical.

Rows are long in year and filing status, with one column per compiled law
field and a field with several brackets spread across numbered columns. A
bracket slot the law does not use, or a year before the first one the
parameter file states a value for, is written as NA. A limit that does not
bind is written as Inf.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# The values of the `law` column. The reform block is the law the scenario's
# reform folders produced; the baseline block is current law, against which
# the reform's revenue effect is measured.
LAW_BASELINE = "baseline"
LAW_REFORM = "reform"


def tax_law_table(scenario_law: Any, baseline_law: Any) -> pd.DataFrame:
    """Both compiled laws of one scenario, as one long table.

    Each argument is a `law.compile.CompiledLaw`. The columns are `law`,
    `year`, `filing_status`, and then one column per compiled law field, in
    the order the compiler lays them out.
    """
    blocks = [
        _one_law(baseline_law, LAW_BASELINE),
        _one_law(scenario_law, LAW_REFORM),
    ]
    return pd.concat(blocks, ignore_index=True)


def _one_law(law: Any, label: str) -> pd.DataFrame:
    header, rows = law.to_frame()
    table = pd.DataFrame(rows, columns=header)
    table.insert(0, "law", label)
    return table
