"""Growth rates of the index series, built from macro projections.

Three series index dollar amounts in the Code: CPI-U, which moved the brackets
and most other dollar amounts before the TCJA and still moves a few; chained
CPI-U, which moves them after it; and the average wage index, which moves the
Social Security taxable maximum. Macro-Projections publishes the level of each
one, history and projection. This module turns those levels into year-over-year
growth rates for the law compiler and the frame layer.

The price series used are the ones the IRS indexes with. Section 1(f) sets tax
year y's amounts from the average price level over September (y-2) through
August (y-1), so Macro-Projections labels that twelve-month average with the
tax year it governs and publishes it as `cpiu_irs`, with `ccpiu_irs` the
chained counterpart. The column named plain `cpiu` in the same file is the
calendar-year average, which the law does not use.

The macro series begin in 1970, but capital gains basis indexation chains CPI
back further, so pre-1970 CPI-U annual averages (BLS, 1982-84 = 100) are
spliced on from `resources/cpiu_historical.csv`. The splice also supplies the
growth rate for 1970 itself, which the macro series has no prior year to
compute from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indexation import IndexTable

# BLS CPI-U annual averages for 1913-1970, a fixed historical record kept in
# the repo.
DEFAULT_HISTORICAL_CPI = Path("resources/cpiu_historical.csv")

# Macro-Projections column name to tax law series name.
MACRO_SERIES = {
    "cpiu_irs": "cpi",
    "ccpiu_irs": "chained_cpi",
    "awi": "awi",
}


def read_macro_spliced(macro_root: str | Path) -> pd.DataFrame:
    """History followed by projection, as one table of annual macro series.

    A Macro-Projections publication is two files that do not overlap in years:
    `historical.csv` through the last completed year, `projections.csv` from
    the next one on.
    """
    root = Path(macro_root)
    return pd.concat(
        [
            pd.read_csv(root / "historical.csv"),
            pd.read_csv(root / "projections.csv"),
        ],
        ignore_index=True,
    )


def derive_index_table(
    macro_root: str | Path,
    historical_cpi: str | Path = DEFAULT_HISTORICAL_CPI,
) -> pd.DataFrame:
    """Year-over-year growth of each index series, one row per series and year.

    Columns are `series`, `year` and `growth`, sorted by series then year. The
    first year of a series has no growth rate and is left missing, so that a
    chain running off the end of the projection stops the compile.
    """
    macro = read_macro_spliced(macro_root)
    missing = set(MACRO_SERIES) - set(macro.columns)
    if missing:
        raise ValueError(
            f"{macro_root} does not publish {', '.join(sorted(missing))}; "
            f"the index table cannot be built from it"
        )

    levels = (
        macro[["year", *MACRO_SERIES]]
        .rename(columns=MACRO_SERIES)
        .melt(id_vars="year", var_name="series", value_name="level")
        .sort_values(["series", "year"])
    )
    levels["growth"] = levels.groupby("series")["level"].transform(
        lambda series: series / series.shift(1) - 1
    )
    macro_growth = levels[["series", "year", "growth"]]

    # Pre-1970 CPI-U, in growth form, for every year the macro series does not
    # already give a CPI growth rate for -- 1913 through 1970.
    historical = pd.read_csv(historical_cpi).sort_values("year")
    historical["growth"] = historical["cpiu"] / historical["cpiu"].shift(1) - 1
    have_growth = set(
        macro_growth.loc[
            (macro_growth["series"] == "cpi") & macro_growth["growth"].notna(), "year"
        ]
    )
    splice = historical.loc[
        historical["growth"].notna() & ~historical["year"].isin(have_growth),
        ["year", "growth"],
    ].assign(series="cpi")

    keep = ~(
        (macro_growth["series"] == "cpi")
        & macro_growth["year"].isin(set(splice["year"]))
    )
    return (
        pd.concat([macro_growth[keep], splice[["series", "year", "growth"]]])
        .sort_values(["series", "year"])
        .reset_index(drop=True)
    )


def index_table_from_frame(frame: pd.DataFrame) -> IndexTable:
    """The compiler's lookup object, from the three-column growth table.

    Years with no growth rate are dropped; the compiler reports a chain that
    reaches one of them as running past the end of the projection.
    """
    rows = frame.loc[frame["growth"].notna()]
    return IndexTable.from_rows(
        [
            (str(series), int(year), float(growth))
            for series, year, growth in zip(
                rows["series"], rows["year"], rows["growth"], strict=True
            )
        ]
    )


__all__ = [
    "DEFAULT_HISTORICAL_CPI",
    "derive_index_table",
    "index_table_from_frame",
    "read_macro_spliced",
]
