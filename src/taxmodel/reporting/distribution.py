"""The layout of the distribution table: its cuts, its group labels, its columns.

Nothing here computes anything. This module states how the distribution table
is laid out: which cuts it is made of, what each group is called, the order
the rows come in, the columns every row carries, and the rounding steps the
metrics use. The table itself is computed in
``reporting/distribution_jax.distribution_table``.

The table has seven group dimensions:

    Overall         the whole population as one group
    Age             five age groups
    Income          quintiles of baseline pretax cash income plus the
                    bequest, a group for negative income, and the top 10, 5,
                    1, 0.1 and 0.01 percent
    Accrual income  the same groups, on baseline pretax Haig-Simons income
                    plus the bequest
    Parent status   parent of a child under 18, or not
    AGI             the same groups, on adjusted gross income
    Net worth       the same groups, on net worth, and one more: every tax
                    unit with a net worth of a billion dollars or more

It has one block of rows per tax: all six taxes together, then each alone.
The whole table is reported twice, once for each way of spreading the
corporate income tax over households (`CORPORATE_SPREADS`). Each version
ranks its Income and Accrual income cuts on its own pretax income, so the
groups of those two cuts differ between versions. Every row carries the change columns, which compare the static world with the
baseline for that row's tax, and the level columns, which report each world's
tax, pretax cash and accrual income, and effective rates.

Every cut reports an income, an accrual income, an AGI and a net-worth cutoff,
whether or not that variable defines the cut. The group average and the group
net change round differently and so cannot be reconciled by multiplying. The
AGI cut labels its negative group "Negative income", the same words the
expanded-income cut uses.
"""

from __future__ import annotations

# Averages are reported to the nearest five dollars.
AVERAGE_ROUNDING_DOLLARS = 5.0

# The dollar sizes of tax change the table counts households at. The first of
# them, five dollars, is also the cutoff for "has a cut" and "has an
# increase", so that rounding noise is not reported as a tax change.
TAX_CHANGE_THRESHOLDS_DOLLARS = (5, 100, 500, 1000, 5000)

# The population is ranked into ten thousand equal-weight groups, one basis
# point of the population each, because the finest cut the table needs is the
# top 0.01 percent. Quintiles and every top cut are read off the same ranking:
# a record is in the top 10 percent when its group index is 9000 or above.
RANK_GROUPS = 10_000

QUINTILE_LABELS = (
    "Quintile 1",
    "Quintile 2",
    "Quintile 3",
    "Quintile 4",
    "Quintile 5",
)

# Each top cut and the smallest group index that belongs to it.
TOP_CUTS = (
    ("Top 10%", 9_000),
    ("Top 5%", 9_500),
    ("Top 1%", 9_900),
    ("Top 0.1%", 9_990),
    ("Top 0.01%", 9_999),
)

# The five age groups. The age used is the older spouse's on a joint return.
AGE_GROUP_LABELS = ("29 and under", "30 - 39", "40 - 49", "50 - 64", "65+")
AGE_GROUP_UPPER_EDGES = (30, 40, 50, 65)

PARENT_LABELS = ("Non-parent", "Parent")

DIMENSION_OVERALL = "Overall"
DIMENSION_AGE = "Age"
DIMENSION_INCOME = "Income"
DIMENSION_ACCRUAL_INCOME = "Accrual income"
DIMENSION_PARENT = "Parent status"
DIMENSION_AGI = "AGI"
DIMENSION_NET_WORTH = "Net worth"

LABEL_NEGATIVE_INCOME = "Negative income"
LABEL_NEGATIVE_NET_WORTH = "Negative net worth"

# The Net worth cut's extra group: every tax unit whose baseline net worth is
# at least one billion dollars. It is a level of net worth, not a percentile,
# so its members are also in the top percentile groups.
BILLIONAIRE_NET_WORTH_DOLLARS = 1e9
LABEL_BILLIONAIRES = "$1 billion or more"

# The rows of one ranked cut: five quintiles, the records whose ranking amount
# is negative, and the five top cuts.
_RANKED_LABELS = (
    *QUINTILE_LABELS,
    LABEL_NEGATIVE_INCOME,
    *(label for label, _lowest_group in TOP_CUTS),
)
_NET_WORTH_LABELS = (
    *QUINTILE_LABELS,
    LABEL_NEGATIVE_NET_WORTH,
    *(label for label, _lowest_group in TOP_CUTS),
    LABEL_BILLIONAIRES,
)

# Every group the table reports, in row order. The Income cut ranks on
# baseline pretax cash income plus the bequest; the Accrual income cut on
# baseline pretax Haig-Simons income plus the bequest. Both pretax incomes
# include the household's share of current-law corporate tax under the row's
# corporate spread.
GROUPS: tuple[tuple[str, str], ...] = (
    ((DIMENSION_OVERALL, DIMENSION_OVERALL),)
    + tuple((DIMENSION_AGE, label) for label in AGE_GROUP_LABELS)
    + tuple((DIMENSION_INCOME, label) for label in _RANKED_LABELS)
    + tuple((DIMENSION_ACCRUAL_INCOME, label) for label in _RANKED_LABELS)
    + tuple((DIMENSION_PARENT, label) for label in PARENT_LABELS)
    + tuple((DIMENSION_AGI, label) for label in _RANKED_LABELS)
    + tuple((DIMENSION_NET_WORTH, label) for label in _NET_WORTH_LABELS)
)

# The values of the `tax` column. `total` is the sum of the six taxes. The
# estate tax and the tax on gains deemed realized at death are charged to the
# heir; the employer's share of payroll tax is in `payroll`; `corporate` is
# the household's top-down share of the corporate income tax.
TAX_TOTAL = "total"
TAXES: tuple[str, ...] = (
    "income_tax",
    "payroll",
    "estate",
    "deemed_at_death",
    "wealth",
    "corporate",
)
TAX_ROWS: tuple[str, ...] = (TAX_TOTAL, *TAXES)

# The two ways the corporate income tax is spread over households, the values
# of the `corporate_spread` column. Both split the tax between the
# above-normal return, the normal return to capital and labor in the same way
# and spread the labor part over labor income; they differ in the two capital
# bases. `income` spreads both capital parts over capital income. `assets`
# spreads the above-normal part over the household's corporate equity and the
# normal-return part over its net capital. `reporting/corporate_spread.py`
# defines the bases.
CORPORATE_SPREAD_INCOME = "income"
CORPORATE_SPREAD_ASSETS = "assets"
CORPORATE_SPREADS: tuple[str, ...] = (CORPORATE_SPREAD_INCOME, CORPORATE_SPREAD_ASSETS)

# The worlds the level columns report. `static` is the first_order world plus
# the top-down share of the reform's static corporate change.
LEVEL_WORLDS: tuple[str, ...] = ("baseline", "static", "conventional")

# The two income measures the effective rates divide by: expanded income
# (cash) and Haig-Simons income (accrual), each before corporate tax.
INCOME_MEASURES: tuple[str, ...] = ("cash", "accrual")


def change_column_names() -> tuple[str, ...]:
    """The columns comparing the static world with the baseline, in order.

    Every one is computed for the row's tax alone. The four cutoffs are the
    lowest baseline value of each amount within the group.
    """
    names = [
        "income_cutoff",
        "accrual_income_cutoff",
        "agi_cutoff",
        "net_worth_cutoff",
        "n_tax_units",
        "after_tax_income_baseline",
        "average_tax_change",
        "average_tax_cut",
        "average_tax_increase",
        "percent_change_in_after_tax_income",
    ]
    names += [
        f"share_with_cut_at_least_{threshold}"
        for threshold in TAX_CHANGE_THRESHOLDS_DOLLARS
    ]
    names += [
        f"share_with_increase_at_least_{threshold}"
        for threshold in TAX_CHANGE_THRESHOLDS_DOLLARS
    ]
    names += ["net_change", "share_of_net_change"]
    return tuple(names)


def level_column_names() -> tuple[str, ...]:
    """The columns reporting each world's tax, income and effective rates.

    For each world: the group's total tax (the row's tax alone), its total
    pretax cash income and pretax accrual income, and the two effective rates
    (tax over each income).
    """
    names: list[str] = []
    for world in LEVEL_WORLDS:
        names.append(f"{world}_tax_dollars")
        names += [f"{world}_{measure}_income_dollars" for measure in INCOME_MEASURES]
        names += [f"{world}_{measure}_etr" for measure in INCOME_MEASURES]
    return tuple(names)


def value_column_names() -> tuple[str, ...]:
    """Every numeric column, in the order of the device table's last axis."""
    return (*change_column_names(), *level_column_names())


# The identifying columns of `distribution.parquet`, then the numeric ones.
KEY_COLUMNS: tuple[str, ...] = (
    "group_dimension",
    "group",
    "tax",
    "corporate_spread",
    "year",
)


def table_columns() -> tuple[str, ...]:
    """Every column of `distribution.parquet`, in the order written."""
    return (*KEY_COLUMNS, *value_column_names())
