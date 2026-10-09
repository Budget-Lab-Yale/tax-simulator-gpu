#!/usr/bin/env python3
"""
Effective tax rates on the top of the wealth distribution, for the right-hand
panel of report figure 1.

The panel shows, for the top 1, 0.1 and 0.01 percent of net worth and for every
tax unit with at least a billion dollars of net worth, the share of income paid
in federal taxes on two income measures: cash income (the model's expanded
income) and accrual income (Haig-Simons income).

Every number comes from the Net worth rows of the scenario's distribution
table, in the baseline world: the total row for the effective rate and one row
per tax for the component rates. Groups are cut on baseline net worth. The
billionaire group is a level cut, so its members are also in the percentile
groups above it. The income each rate divides by is the measure's own income
plus any bequest received plus the household's share of current-law corporate
tax; the header of `readers.py` describes the table.

For each group and income definition the script prints the number of tax
units, income and tax in billions of dollars, the effective rate, and the rate
of each tax in it. The component rates add up to the effective rate; the
estate tax and the tax on gains deemed realized at death together are the tax
owed at death.

Usage:
    python3 projects/top_tax/wealth_group_etr.py RUN_DIR [--scenario baseline]
                                                         [--year 2027]

Reads one Parquet file with pandas, so it runs on a login node.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readers

# The groups the figure shows, as the Net worth cut of the distribution table
# labels them. The last is the level cut for net worth of a billion dollars or
# more.
FIGURE_GROUPS = ["Top 1%", "Top 0.1%", "Top 0.01%", "$1 billion or more"]

# The two income definitions: as the figure names them, the model's measure,
# and readers.py's label.
INCOME_DEFINITIONS = [
    ("Cash income", "expanded income", "expanded"),
    ("Accrual income", "Haig-Simons income", "hs"),
]

# The component rates printed, with their name in readers.ETR_COMPS.
COMPONENTS = [
    ("Income tax", "income_tax"),
    ("Payroll tax", "payroll"),
    ("Estate tax", "estate"),
    ("Tax on gains at death", "deemed"),
    ("Wealth tax", "wealth"),
    ("Corporate tax", "corp"),
]

# EXTERNAL: the year the report's figure 1 is drawn for.
FIGURE_YEAR = 2027


def _percent(rate):
    return f"{100.0 * rate:.1f}%"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="model run directory")
    ap.add_argument("--scenario", default="baseline",
                    help="scenario whose baseline world is read")
    ap.add_argument("--year", type=int, default=FIGURE_YEAR)
    args = ap.parse_args()

    if not readers.has_distribution_table(args.run_dir, args.scenario):
        sys.exit(f"scenario {args.scenario} in {args.run_dir} has no "
                 "distribution table")
    # Each row pairs the baseline with a reform world; only the baseline side
    # is printed.
    rows = {(r["income_definition"], r["group"]): r
            for r in readers.etr_rows(
                args.run_dir, args.scenario, reform_leg="static",
                group_dimension=readers.NET_WORTH_DIMENSION, years=[args.year])}
    if not rows:
        sys.exit(f"the distribution table has no Net worth rows for {args.year}")

    print(f"# Effective tax rates by net worth, {args.year}, baseline world "
          f"of scenario {args.scenario}")
    headings = ["Group", "Tax units", "Income ($B)", "Tax ($B)",
                "Effective rate", *(name for name, _column in COMPONENTS)]
    for title, measure, definition in INCOME_DEFINITIONS:
        print()
        print(f"## {title} ({measure})")
        print()
        print("| " + " | ".join(headings) + " |")
        print("|" + "---|" * len(headings))
        for group in FIGURE_GROUPS:
            r = rows.get((definition, group))
            if r is None:
                print(f"| {group} | "
                      + " | ".join(["missing"] * (len(headings) - 1)) + " |")
                continue
            cells = [
                group,
                f"{r['n_tax_units']:,.0f}",
                f"{r['income_baseline']:,.1f}",
                f"{r['tax_baseline']:,.1f}",
                _percent(r["etr_baseline"]),
                *(_percent(r[f"etr_{comp}_baseline"])
                  for _name, comp in COMPONENTS),
            ]
            print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
