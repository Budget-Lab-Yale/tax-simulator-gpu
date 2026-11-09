"""Growth rates of the index series, built from macro projections.

Three series index dollar amounts in the Code: CPI-U, which moved the brackets
and most other dollar amounts before the TCJA and still moves a few; chained
CPI-U, which moves them after it; and the average wage index, which moves the
Social Security taxable maximum. Macro-Projections publishes the level of each
one, history and projection. This module turns those levels into year-over-year
growth rates for the law compiler and the frame layer.

The price series are the ones the IRS indexes with. Section 1(f) defines the
price level for calendar year y as the average over September (y-1) through
August (y), and Macro-Projections labels that twelve-month average y. It is
published as `cpiu_irs`, with `ccpiu_irs` and `ccpiu_irs_sept` the chained
counterparts. The column named plain `cpiu` in the same file is the
calendar-year average, which the law does not use.

Chained CPI-U is revised for about a year after BLS first publishes it.
Section 1(f)(6)(A) has the IRS use the values published as of the initial
August release, so the most recent year's average still holds preliminary
values while the base year's average is final. `ccpiu_irs` holds the revised
values and gives the chained growth rates here. `ccpiu_irs_sept` holds each
year's average as published that September, and the `preliminary` column of
the table is its ratio to the revised value. The law compiler applies it to
the most recent year of an index ratio. CPI-U is not revised, and the wage
index has no such rule, so their `preliminary` is 1.

The macro series begin in 1970, but capital gains basis indexation chains CPI
back further, so pre-1970 CPI-U annual averages (BLS, 1982-84 = 100) are
spliced on from `resources/cpiu_historical.csv`. The splice also supplies the
growth rate for 1970 itself, which the macro series has no prior year to
compute from.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .indexation import IndexTable

# BLS CPI-U annual averages for 1913-1970, a fixed historical record kept in
# the repo.
DEFAULT_HISTORICAL_CPI = Path("resources/cpiu_historical.csv")

# Macro-Projections column name to tax law series name. The growth rates come
# from these columns.
MACRO_SERIES = {
    "cpiu_irs": "cpi",
    "ccpiu_irs": "chained_cpi",
    "awi": "awi",
}

# Chained CPI-U as first published each September, divided by the revised
# series above to give the `preliminary` column.
CHAINED_CPI_AS_FIRST_PUBLISHED = "ccpiu_irs_sept"


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

    Columns are `series`, `year`, `growth` and `preliminary`, sorted by series
    then year. The first year of a series has no growth rate and is left
    missing, so that a chain running off the end of the projection stops the
    compile. `preliminary` is the year's level as first published over its
    revised level (see the module docstring); it is 1 where the two agree or
    the first-published value is not available.
    """
    macro = read_macro_spliced(macro_root)
    missing = ({*MACRO_SERIES, CHAINED_CPI_AS_FIRST_PUBLISHED}) - set(macro.columns)
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
    table = pd.concat([macro_growth[keep], splice[["series", "year", "growth"]]])

    chained_preliminary = (
        macro[CHAINED_CPI_AS_FIRST_PUBLISHED] / macro["ccpiu_irs"]
    ).fillna(1.0)
    preliminary = pd.DataFrame(
        {"series": "chained_cpi", "year": macro["year"], "preliminary": chained_preliminary}
    )
    table = table.merge(preliminary, on=["series", "year"], how="left")
    table["preliminary"] = table["preliminary"].fillna(1.0)
    return table.sort_values(["series", "year"]).reset_index(drop=True)


def index_table_from_frame(frame: pd.DataFrame) -> IndexTable:
    """The compiler's lookup object, from the growth table.

    Years with no growth rate are dropped; the compiler reports a chain that
    reaches one of them as running past the end of the projection.
    """
    if "preliminary" not in frame.columns:
        raise ValueError(
            "the index table has no `preliminary` column, so chained CPI-U "
            "cannot be indexed the way the IRS does it; rebuild the table from "
            "a Macro-Projections publication that carries `ccpiu_irs_sept`"
        )
    rows = frame.loc[frame["growth"].notna()]
    table = IndexTable.from_rows(
        [
            (str(series), int(year), float(growth))
            for series, year, growth in zip(
                rows["series"], rows["year"], rows["growth"], strict=True
            )
        ]
    )
    preliminary: dict[str, dict[int, float]] = {}
    for series, year, ratio in zip(
        frame["series"], frame["year"], frame["preliminary"], strict=True
    ):
        if ratio != 1.0:
            preliminary.setdefault(str(series), {})[int(year)] = float(ratio)
    return replace(table, preliminary=preliminary)


__all__ = [
    "DEFAULT_HISTORICAL_CPI",
    "derive_index_table",
    "index_table_from_frame",
    "read_macro_spliced",
]
