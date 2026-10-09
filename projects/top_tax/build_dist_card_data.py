#!/usr/bin/env python3
"""
Writes dist_card_data.json, the data behind the interactive distribution card.

Reads the Income cut of the stack_ref scenario's distribution table and
writes, per income group, the tax owed under current law, the tax the reform
asks for before households respond, the tax actually collected after they
respond, and the gap between the ask and the collection. Dollar views use disjoint bins differenced out of the cumulative
top groups; the rate view keeps the nested groups as published.

All of these come from the level columns of the table's total rows. The ask is
the static world's tax less the baseline world's, and the collection is the
conventional world's tax less the baseline world's. The page divides every
tax by the group's baseline pretax income on the chosen basis.

Every number on the card is read from that table, so a run without one
produces no card. The script then prints why and exits with NO_TABLE_EXIT.

Usage:
  python3 projects/top_tax/build_dist_card_data.py <run_dir> [scenario]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import readers as X

YEARS = ["2027", "2036"]

# Exit code for "this run has no distribution table".
NO_TABLE_EXIT = 3

# The two reform worlds of the distribution table the card reads.
STATIC_LEG = "static"
CONVENTIONAL_LEG = "conventional"

# The card's two income views, as readers.py labels them: expanded income
# (cash) and Haig-Simons income (accrual).
CASH_INCOME_DEF = "expanded"
ACCRUAL_INCOME_DEF = "hs"

NESTED = ["Quintile 1", "Quintile 2", "Quintile 3", "Quintile 4", "Quintile 5",
          "Top 10%", "Top 5%", "Top 1%", "Top 0.1%", "Top 0.01%"]
NESTED_OUT = ["Q1", "Q2", "Q3", "Q4", "Q5",
              "Top 10%", "Top 5%", "Top 1%", "Top 0.1%", "Top 0.01%"]

# Disjoint dollar bins: quintiles 1-4 as-is, then successive differences of
# the cumulative top groups.
DISJOINT = [
    ("Bottom 20%",  "Quintile 1", None),
    ("20–40%", "Quintile 2", None),
    ("40–60%", "Quintile 3", None),
    ("60–80%", "Quintile 4", None),
    ("80–90%", "Quintile 5", "Top 10%"),
    ("90–95%", "Top 10%",    "Top 5%"),
    ("95–99%", "Top 5%",     "Top 1%"),
    ("99–99.9%",    "Top 1%",   "Top 0.1%"),
    ("99.9–99.99%", "Top 0.1%", "Top 0.01%"),
    ("Top 0.01%",   "Top 0.01%",  None),
]


def load_slice(run_dir, scen_id):
    """Indexes the card's rows by (year, leg, income definition, group).
    Money in the rows is already in billions."""
    out = {}
    for leg in (STATIC_LEG, CONVENTIONAL_LEG):
        for r in X.etr_rows(run_dir, scen_id, reform_leg=leg):
            if r["year"] in YEARS and r["group"] in NESTED:
                out[(r["year"], leg, r["income_definition"], r["group"])] = r
    return out


def group_rec(sl, year, group):
    """Assembles one nested group's record: counts, income under both
    definitions, the tax owed under current law, and the tax change under each
    of the two reform legs.

    Returns: dict of floats (n, cash, accrual, tax_base, static, conv), money
    in billions of dollars.
    """
    st = sl[(year, STATIC_LEG, CASH_INCOME_DEF, group)]
    cv = sl[(year, CONVENTIONAL_LEG, CASH_INCOME_DEF, group)]
    accrual = sl[(year, STATIC_LEG, ACCRUAL_INCOME_DEF, group)]
    return dict(
        n=float(st["n_tax_units"]),
        cash=float(st["income_baseline"]),
        accrual=float(accrual["income_baseline"]),
        tax_base=float(st["tax_baseline"]),
        static=float(st["tax_reform"]) - float(st["tax_baseline"]),
        conv=float(cv["tax_reform"]) - float(cv["tax_baseline"]),
    )


def pack(rec, label):
    """Rounds one record into the card's bin shape.

    Returns: dict matching dist_card_data.json's per-bin schema.
    """
    return dict(
        bin=label, n=round(rec["n"], 1),
        income=dict(cash=round(rec["cash"], 3), accrual=round(rec["accrual"], 3)),
        tax_base=round(rec["tax_base"], 3),
        static=round(rec["static"], 4),
        conv=round(rec["conv"], 4),
        avoid=round(rec["static"] - rec["conv"], 4),
    )


def diff(a, b):
    """Subtracts record b from record a, key by key.

    Returns: dict of floats.
    """
    return {k: a[k] - b[k] for k in a}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: build_dist_card_data.py <run_dir> [scenario]")
    root = sys.argv[1].rstrip("/")
    scen = sys.argv[2] if len(sys.argv) > 2 else "stack_ref"
    if not X.has_distribution_table(root, scen):
        print(f"No distribution table in {root}/scenarios/{scen}. The "
              "distribution card is made entirely from that table, so there "
              "is nothing to build from this run.", file=sys.stderr)
        sys.exit(NO_TABLE_EXIT)
    sl = load_slice(root, scen)

    years = {}
    for year in YEARS:
        recs = {g: group_rec(sl, year, g) for g in NESTED}
        bins = []
        for label, hi, minus in DISJOINT:
            r = recs[hi] if minus is None else diff(recs[hi], recs[minus])
            assert minus is None or r["n"] > 0, f"{year} {label}: empty bin"
            bins.append(pack(r, label))
        nested = [pack(recs[g], lab) for g, lab in zip(NESTED, NESTED_OUT)]

        # The disjoint bins must re-add to the full population.
        assert abs(sum(b["n"] for b in bins) -
                   sum(recs[q]["n"] for q in NESTED[:5])) < 1, year
        years[year] = dict(bins=bins, nested=nested)

    payload = dict(
        meta=dict(
            scenario=scen, vintage=os.path.basename(root),
            ranking="Income percentile, on baseline pretax cash income",
            income_defs=dict(
                cash="expanded (cash) income",
                accrual="Haig-Simons income (accrual, including unrealized "
                        "gains)"),
            taxes=X.TAXES_DESCRIPTION,
            units="$B",
            note=("dollar views use disjoint bins; rate view uses nested "
                  "cumulative groups (atlas set); tax $ invariant to income basis"),
        ),
        years=years,
    )

    out_json = os.path.join(HERE, "dist_card_data.json")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
