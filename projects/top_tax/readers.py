#!/usr/bin/env python3
"""
Read a Python-model run directory for the top-tax surrogate and atlas.

This is the only place in the top-tax project that knows the on-disk layout of
a model run.

What a run directory looks like:

    {run_dir}/manifest.json
    {run_dir}/scenarios/{scenario ID}/receipts.csv
    {run_dir}/scenarios/{scenario ID}/revenue_estimates.csv
    {run_dir}/scenarios/{scenario ID}/form1040.csv
    {run_dir}/scenarios/{scenario ID}/distribution.parquet

Callers name results by "result leg" (baseline, static, mechanical,
conventional). The model writes raw dollars, and everything this module
returns is in billions except effective rates and counts of tax units.

Revenue files. `receipts.csv` and `revenue_estimates.csv` carry a `world`
column. The static leg is the model's first_order world there; LEG_TO_WORLD
maps each leg to its world.

The distribution table. `distribution.parquet` has one row per group
dimension, group, tax, corporate spread and year. The `tax` column is `total`
or one of the six taxes in ETR_COMP_TAXES, and every number on a row is for
that tax alone. The `corporate_spread` column says how the corporate income
tax is spread over households: `income` spreads its capital part over capital
income, `assets` over corporate equity and net capital. Each version is a
whole table with its own Income and Accrual income groups. The readers read
the `income` version unless asked for another (CORPORATE_SPREAD). Each
row carries, for the baseline, static and conventional worlds, the group's tax
dollars, its pretax cash and accrual income dollars, and the two effective
rates (tax over each income). The static world is first_order plus the
household's share of the reform's static change in corporate tax. Pretax
income includes the household's share of current-law corporate tax and, for
an heir, the bequest. The Income cut ranks households on baseline pretax cash
income plus the bequest, and the Accrual income cut on baseline pretax
Haig-Simons income plus the bequest; the Net worth cut includes a group for
tax units worth a billion dollars or more.

The revenue files are read with the standard library. The distribution table
is Parquet, so the functions that read it need pandas; they import it
themselves, so everything else works on a machine without it.
"""

import csv
import json
import os
from pathlib import Path

# Cumulative fiscal-year nominal GDP, the denominator for "share of GDP"
# statements in the atlas.
MACRO = ("/nfs/roberts/project/pi_nrs36/shared/model_data/Macro-Projections/"
         "v3/2026022522/baseline/projections.csv")

# The three ten-year windows the atlas reports. Contiguous by construction:
# leg_deltas_windows assumes the union runs from the first year to the last
# with no gap.
DECADES = [(2027, 2036), (2037, 2046), (2047, 2056)]

# Years for which effective tax rates are displayed. The shipped atlas uses
# the first of these (the year the reform takes effect); the dist card uses
# both.
DIST_YEARS = [2027, 2036]

# The two income definitions the atlas offers as a cash/accrual toggle, and
# the distribution table's name for each: "expanded" is expanded income (cash),
# "hs" is Haig-Simons income (accrual).
ETR_INCOME_DEFS = ["hs", "expanded"]
INCOME_MEASURE_OF = {"expanded": "cash", "hs": "accrual"}

# The distribution table's group dimensions that the scripts read. The atlas
# and the dist card read the Income cut with both denominators, so the same
# households sit in each group whichever income definition is on screen.
ETR_GROUP_DIMENSION = "Income"
ACCRUAL_INCOME_DIMENSION = "Accrual income"
NET_WORTH_DIMENSION = "Net worth"

# The file, and the value of its `tax` column for the sum of all six taxes.
DISTRIBUTION_FILE = "distribution.parquet"
TAX_TOTAL = "total"

# The version of the distribution table the readers read by default: the
# corporate tax's capital part spread over capital income.
CORPORATE_SPREAD = "income"

# The taxes every effective rate here counts, as a caption.
TAXES_DESCRIPTION = ("individual income tax after refundable credits, payroll "
                     "tax including the employer's share, the estate tax, the "
                     "tax on gains deemed realized at death, the annual wealth "
                     "tax, and the corporate income tax spread over households")

# Recorded in the atlas JSON's meta block so a reader can see which slice of
# the distribution table the chart is built from.
ETR_FILTER = {"taxes": TAXES_DESCRIPTION,
              "group_dimension": ETR_GROUP_DIMENSION}

# The six effective-rate components the atlas chart stacks, each with the
# value of the `tax` column it is read from.
ETR_COMP_TAXES = {
    "income_tax": "income_tax",
    "payroll": "payroll",
    "estate": "estate",
    "deemed": "deemed_at_death",
    "wealth": "wealth",
    "corp": "corporate",
}
ETR_COMPS = list(ETR_COMP_TAXES)

# The result legs and the model worlds that correspond to them in the revenue
# files. "static" is the reform with nothing else changed; "conventional" adds
# the household behavioral response. The model's fifth world,
# mechanical_ex_saving, is not used by this pipeline.
LEG_TO_WORLD = {
    "baseline": "baseline",
    "static": "first_order",
    "mechanical": "mechanical",
    "conventional": "conventional",
}

# The worlds the distribution table reports levels for. It has no mechanical
# world, and it names the static leg "static".
LEVEL_WORLDS = ("baseline", "static", "conventional")

# Liability at preferential rates, which the atlas shows as its own
# destination head rather than leaving inside the income-tax head: long-term
# capital gains and qualified dividends (liab_pref), unrecaptured section 1250
# gain (liab_1250), and collectibles (liab_collect).
CG_1040_COLS = ["liab_pref", "liab_1250", "liab_collect"]

# The model writes a missing effective rate where the group's income is not
# positive — in practice the "Negative income" group, whose members owe tax
# but have negative income for the year, so tax as a share of income means
# nothing. That group is dropped from the chart entirely. Any missing rate
# that slips through elsewhere reads as zero.
DROPPED_ETR_GROUPS = ("Negative income",)
UNDEFINED_RATE_READS_AS = 0.0

# The destination ledger the atlas reports revenue into, in display order.
HEADS_ORDER = ["iit", "cg", "pay", "corp", "est", "wealth", "other"]

# The line in revenue_estimates.csv that is the official revenue estimate:
# the whole budget effect, revenues less outlays, against the baseline world.
OFFICIAL_TOTAL_LINE = "total_budget_effect"

# The model writes raw dollars; everything here is billions.
DOLLARS_PER_BILLION = 1e9

# A scenario's destination-ledger deltas are added up from the receipt lines,
# while the headline number comes from the official revenue estimate. The two
# can differ slightly through rounding and fiscal-year booking. Anything
# larger than this is worth a human looking at.
DRIFT_WARN_FLOOR_B = 1.0        # one billion dollars
DRIFT_WARN_SHARE = 0.005        # or half a percent of the window total


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def year_map(rows, col):
    """{year: value} from any table with a `year` column, skipping blanks."""
    return {int(r["year"]): float(r[col]) for r in rows if r[col] not in ("", "NA")}


def scenario_dir(run_dir, scen_id):
    return os.path.join(run_dir, "scenarios", scen_id)


def _world_of(leg):
    if leg in LEG_TO_WORLD:
        return LEG_TO_WORLD[leg]
    if leg in LEG_TO_WORLD.values():
        return leg
    raise KeyError(f"unknown result leg or world: {leg!r}")


# --------------------------------------------------------------------------- #
# Revenue
# --------------------------------------------------------------------------- #
def receipts_heads(run_dir, scen_id, world):
    """Fiscal-year receipt levels by destination head, in billions.

    Reads `receipts.csv`, which is the scenario's own collections by fiscal
    year for each world — not a difference against anything. Heads:

      iit     individual income tax net of refundable credit outlays. The
              credits are paid out of the same line item in the budget, so
              netting them is what makes this head "what the income tax
              raises". Liability at preferential rates is still inside this
              head here; cg_carve() takes it out later.
      pay     payroll taxes
      corp    corporate income tax
      est     estate tax
      wealth  annual wealth tax
      other   everything else

    A year in which any of these lines is missing is skipped.
    """
    world = _world_of(world)
    rows = [r for r in read_csv(os.path.join(scenario_dir(run_dir, scen_id),
                                             "receipts.csv"))
            if r["world"] == world]
    by_year = {}
    for r in rows:
        by_year.setdefault(int(r["year"]), {})[r["line"]] = r["dollars"]

    needed = ["revenues_income_tax", "outlays_tax_credits", "revenues_payroll_tax",
              "revenues_corp_tax", "revenues_estate_tax", "revenues_wealth_tax",
              "revenues_other"]
    out = {}
    for year, lines in by_year.items():
        if any(lines.get(c) in (None, "", "NA") for c in needed):
            continue
        b = {c: float(lines[c]) / DOLLARS_PER_BILLION for c in needed}
        out[year] = {
            "iit": b["revenues_income_tax"] - b["outlays_tax_credits"],
            "pay": b["revenues_payroll_tax"],
            "corp": b["revenues_corp_tax"],
            "est": b["revenues_estate_tax"],
            "wealth": b["revenues_wealth_tax"],
            "other": b["revenues_other"],
        }
    return out


def cg_carve(run_dir, scen_id, world):
    """Calendar-year liability taxed at preferential rates, in billions.

    This is the number the atlas moves out of the income-tax head into its own
    capital-gains head, so a reader can see how much of a reform's revenue
    comes from gains and dividends rather than from ordinary income. Read from
    `form1040.csv`, which covers filers only under the model's own filing
    test.
    """
    world = _world_of(world)
    rows = read_csv(os.path.join(scenario_dir(run_dir, scen_id), "form1040.csv"))
    return {int(r["year"]): sum(float(r[c]) for c in CG_1040_COLS)
            / DOLLARS_PER_BILLION
            for r in rows if r["world"] == world}


def official_estimate(run_dir, scen_id, world):
    """{fiscal year: billions} of the official revenue estimate for a world.

    `revenue_estimates.csv` is the scenario's receipts less the baseline
    world's, line by line. `total_budget_effect` is the whole-budget line and
    is the number that gets quoted; the destination heads are a decomposition
    of it that has to add back up.
    """
    world = _world_of(world)
    rows = read_csv(os.path.join(scenario_dir(run_dir, scen_id),
                                 "revenue_estimates.csv"))
    return {int(r["year"]): float(r["dollars"]) / DOLLARS_PER_BILLION
            for r in rows
            if r["world"] == world and r["line"] == OFFICIAL_TOTAL_LINE
            and r["dollars"] not in ("", "NA")}


def baseline_levels(run_dir, scen_id="baseline", legs=("static", "mechanical",
                                                       "conventional")):
    """Baseline receipt heads and preferential-rate liability, ready to
    subtract from any leg.

    Every leg differences against the same current-law baseline, so this
    returns one reading of the baseline scenario's `baseline` world repeated
    under each leg name — the shape leg_deltas_windows expects.
    """
    heads = receipts_heads(run_dir, scen_id, "baseline")
    carve = cg_carve(run_dir, scen_id, "baseline")
    return ({leg: heads for leg in legs}, {leg: carve for leg in legs})


def leg_deltas_windows(run_dir, scen_id, base_heads, base_cg, leg,
                       windows=DECADES):
    """Revenue effects of one scenario, one leg, over each decade window.

    Returns (result, drifts). The result is:

      totals  one official revenue estimate per window, billions
      byyear  the official estimate for every year in the span the windows
              cover, billions
      heads   per window, the change in each destination head, billions. The
              capital-gains head is the change in preferential-rate liability;
              the income-tax head is the change in the income tax net of
              credits with that carve taken out, so the two together still
              equal the change in the income-tax head as the budget books it.
      span    [first year, last year]

    `drifts` is one number per window: the destination heads added up, minus
    the official estimate. It should be near zero; drift_warnings() turns the
    ones worth reading into sentences.
    """
    heads = receipts_heads(run_dir, scen_id, leg)
    carve = cg_carve(run_dir, scen_id, leg)
    official = official_estimate(run_dir, scen_id, leg)

    span = list(range(min(w[0] for w in windows),
                      max(w[1] for w in windows) + 1))
    totals, heads_w, drifts = [], [], []
    for first_year, last_year in windows:
        window_heads = {k: 0.0 for k in HEADS_ORDER}
        for year in range(first_year, last_year + 1):
            change_in_pref_liability = carve[year] - base_cg[leg][year]
            window_heads["cg"] += change_in_pref_liability
            window_heads["iit"] += ((heads[year]["iit"]
                                     - base_heads[leg][year]["iit"])
                                    - change_in_pref_liability)
            for k in ("pay", "corp", "est", "wealth", "other"):
                window_heads[k] += heads[year][k] - base_heads[leg][year][k]
        window_total = sum(official.get(year, 0.0)
                           for year in range(first_year, last_year + 1))
        totals.append(round(window_total, 3))
        heads_w.append({k: round(v, 3) for k, v in window_heads.items()})
        drifts.append(sum(window_heads.values()) - window_total)

    byyear = [round(official.get(year, 0.0), 3) for year in span]
    return {"totals": totals, "byyear": byyear, "heads": heads_w,
            "span": [span[0], span[-1]]}, drifts


def drift_warnings(scen_id, leg, result, drifts):
    """One sentence per window whose heads do not add back to the official
    estimate closely enough to ignore."""
    out = []
    for i, drift in enumerate(drifts):
        window_total = result["totals"][i]
        allowed = max(DRIFT_WARN_FLOOR_B, DRIFT_WARN_SHARE * abs(window_total))
        if abs(drift) > allowed:
            first_year, last_year = DECADES[i] if i < len(DECADES) else (0, 0)
            out.append(f"{scen_id}/{leg} {first_year}-{last_year}: destination "
                       f"heads add to {drift:+.2f}B more than the official "
                       f"estimate of {window_total:.2f}B")
    return out


# --------------------------------------------------------------------------- #
# Effective tax rates, from the distribution table
# --------------------------------------------------------------------------- #
def _distribution_path(run_dir, scen_id):
    return os.path.join(scenario_dir(run_dir, scen_id), DISTRIBUTION_FILE)


def has_distribution_table(run_dir, scen_id):
    """Did this scenario write a distribution table?

    Runs made without that table still carry every revenue quantity, so the
    surrogate can be fitted from them with the effective-rate quantities left
    out.
    """
    return os.path.exists(_distribution_path(run_dir, scen_id))


def level_columns(world, measure):
    """The distribution table's tax, income and rate columns for one world and
    one income measure ("cash" or "accrual")."""
    return (f"{world}_tax_dollars", f"{world}_{measure}_income_dollars",
            f"{world}_{measure}_etr")


def read_distribution(run_dir, scen_id, group_dimension, years, columns,
                      corporate_spread=CORPORATE_SPREAD):
    """The rows of one group dimension in the given years, under one corporate
    spread, as a DataFrame.

    Returns the identifying columns (group, tax, corporate spread, year) and
    the named value columns. Stops with a message naming the file when it lacks
    a column, as a distribution table written before the table carried one row
    per tax, or one version per corporate spread, does.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    path = _distribution_path(run_dir, scen_id)
    wanted = ["group_dimension", "group", "tax", "corporate_spread", "year",
              *columns]
    missing = [c for c in wanted if c not in pq.read_schema(path).names]
    if missing:
        raise ValueError(f"{path} has no column {', '.join(missing)}; it was "
                         "written by a model version older than these readers")
    table = pd.read_parquet(path, columns=wanted)
    return table[(table["group_dimension"] == group_dimension)
                 & (table["corporate_spread"] == corporate_spread)
                 & table["year"].isin([int(y) for y in years])]


def _level_world_of(leg):
    """The distribution table's world for a result leg. The revenue files'
    first_order world is the table's static world."""
    world = "static" if leg == "first_order" else leg
    if world not in LEVEL_WORLDS:
        raise KeyError(f"the distribution table has no world for {leg!r}")
    return world


def _rate(value):
    """An effective rate from the model's table; see UNDEFINED_RATE_READS_AS."""
    if value != value:  # a missing rate is not a number
        return UNDEFINED_RATE_READS_AS
    return float(value)


def etr_rows(run_dir, scen_id, reform_leg="static",
             group_dimension=ETR_GROUP_DIMENSION, years=DIST_YEARS,
             corporate_spread=CORPORATE_SPREAD):
    """Effective tax rates by group, for the baseline and one reform world.

    Returns one row per (year, income definition, group) carrying the number
    of tax units, baseline and reform income and tax in billions, and the
    baseline and reform effective rate, in total and for each component in
    ETR_COMPS. The component rates share the total's denominator, so they add
    up to it.

    `reform_leg` picks the reform world: "static" is the rate a household is
    asked to pay before it responds, "conventional" the rate it pays after
    responding. Each world divides by its own pretax income.

    Rates stay as fractions here, the way the model writes them; etr_pack()
    turns them into percent. The "Negative income" group is left out
    (DROPPED_ETR_GROUPS). By default the rows are the Income cut in DIST_YEARS,
    under the `income` corporate spread; wealth_group_etr.py reads the Net
    worth cut.
    """
    if not has_distribution_table(run_dir, scen_id):
        return []
    worlds = {"baseline": "baseline", "reform": _level_world_of(reform_leg)}
    columns = ["n_tax_units"]
    for world in worlds.values():
        for measure in INCOME_MEASURE_OF.values():
            columns += [c for c in level_columns(world, measure)
                        if c not in columns]
    table = read_distribution(run_dir, scen_id, group_dimension, years, columns,
                              corporate_spread)
    table = table[~table["group"].isin(DROPPED_ETR_GROUPS)]
    by_key = {(int(r["year"]), r["group"], r["tax"]): r
              for r in table.to_dict("records")}
    group_order = list(dict.fromkeys(table["group"]))

    out = []
    for year in years:
        for group in group_order:
            total = by_key.get((int(year), group, TAX_TOTAL))
            if total is None:
                continue
            for income_def in ETR_INCOME_DEFS:
                measure = INCOME_MEASURE_OF[income_def]
                row = {
                    "year": str(year),
                    "income_definition": income_def,
                    "group": group,
                    "group_dimension": group_dimension,
                    "reform_leg": reform_leg,
                    "n_tax_units": float(total["n_tax_units"]),
                }
                for side, world in worlds.items():
                    tax, income, rate = level_columns(world, measure)
                    row[f"income_{side}"] = total[income] / DOLLARS_PER_BILLION
                    row[f"tax_{side}"] = total[tax] / DOLLARS_PER_BILLION
                    row[f"etr_{side}"] = _rate(total[rate])
                    for comp, tax_row in ETR_COMP_TAXES.items():
                        row[f"etr_{comp}_{side}"] = _rate(
                            by_key[(int(year), group, tax_row)][rate])
                out.append(row)
    return out


def etr_pack(rows, which):
    """{income definition: {year: {group: [the six components]}}} in percent.

    `which` is "baseline" or "reform". Components come out in ETR_COMPS order,
    rounded to three decimal places, which is what the atlas JSON ships.
    """
    out = {}
    for r in rows:
        d = r["income_definition"]
        y = str(r["year"])
        vals = [round(100.0 * float(r[f"etr_{c}_{which}"]), 3) for c in ETR_COMPS]
        out.setdefault(d, {}).setdefault(y, {})[r["group"]] = vals
    return out


def income_cutoffs(run_dir, scen_id, year, group_dimension=ETR_GROUP_DIMENSION,
                   corporate_spread=CORPORATE_SPREAD):
    """{group: income floor in dollars} for one year: the table's
    `income_cutoff`, the lowest baseline pretax cash income among the group's
    tax units, counting the household's share of current-law corporate tax
    under `corporate_spread` and, for an heir, the bequest."""
    table = read_distribution(run_dir, scen_id, group_dimension, [year],
                              ["income_cutoff"], corporate_spread)
    table = table[table["tax"] == TAX_TOTAL]
    return {group: float(cutoff)
            for group, cutoff in zip(table["group"], table["income_cutoff"])
            if cutoff == cutoff}  # a missing cutoff is not a number


# --------------------------------------------------------------------------- #
# Finding a run, and the GDP denominators
# --------------------------------------------------------------------------- #
def newest_complete_run(root):
    """Path to the most recent finished run directory at or under `root`.

    The model builds a run directory with a `.partial` suffix and renames it
    only when the run finishes, so a directory without that suffix is complete.
    Directory names carry the run's own timestamp (2026-08-04T142225_b5996017),
    which sorts chronologically, so the newest is the last by name.

    A `scenarios` folder is what marks a run directory. The run-level
    manifest.json is not required: a directory assembled by hand from the
    scenario folders of several runs has the per-scenario manifests but no
    run-level one.

    Accepts either a run directory itself, the folder holding the runs for one
    runscript, or the whole runs root.
    """
    root = Path(root)
    if _is_run_dir(root):
        return root
    found = [p for depth in (root.glob("*"), root.glob("*/*"))
             for p in depth if _is_run_dir(p)]
    if not found:
        raise FileNotFoundError(f"no finished run directory at or under {root}")
    return max(found, key=lambda p: p.name)


def _is_run_dir(path):
    return (path.is_dir() and not path.name.endswith(".partial")
            and (path / "scenarios").is_dir())


def gdp_fy_decades(windows=DECADES, macro_path=MACRO):
    """Cumulative nominal GDP by fiscal year over each window, in billions.

    The denominator behind every "x percent of GDP" statement in the atlas.
    Read from the baseline macroeconomic projection the model itself is pinned
    to, so the atlas and the model share one view of the size of the economy.
    """
    gdp = year_map(read_csv(macro_path), "gdp_fy")
    return [round(sum(gdp[y] for y in range(first, last + 1)), 1)
            for first, last in windows]


def run_manifest(run_dir):
    """The run-level manifest: which runscript ran, which scenarios, which
    model commit, and every file written.

    Returns None when the run directory has no run-level manifest, which is
    the case for a directory assembled by hand from several runs' scenario
    folders. Each scenario folder still carries its own manifest.json.
    """
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def scenario_manifest(run_dir, scen_id):
    """One scenario's manifest: the law layers it ran, the years, the model
    commit and the calibrated constants it used."""
    with open(os.path.join(scenario_dir(run_dir, scen_id), "manifest.json")) as fh:
        return json.load(fh)


def has_world(run_dir, scen_id, world):
    """Did this scenario write revenue for the named world?

    A scenario run without a behavioral response writes no conventional world,
    and one with no corporate or employer-payroll transmission writes no
    mechanical world. The surrogate substitutes the rung below when a rung is
    missing, which is what the reporting layer does.
    """
    path = os.path.join(scenario_dir(run_dir, scen_id), "receipts.csv")
    if not os.path.exists(path):
        return False
    return any(r["world"] == world for r in read_csv(path))


def revenue_estimate_lines(run_dir, scen_id, world):
    """{fiscal year: {budget line: change in billions}} for one scenario and world.

    revenue_estimates.csv is the scenario's receipts less the baseline world's,
    line by line, so these are already differences against current law. The
    lines are the seven published revenue lines plus total_budget_effect.
    """
    path = os.path.join(scenario_dir(run_dir, scen_id), "revenue_estimates.csv")
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["world"] != world or row["dollars"] in ("", "NA", "NaN"):
                continue
            year = int(row["year"])
            out.setdefault(year, {})[row["line"]] = (
                float(row["dollars"]) / DOLLARS_PER_BILLION)
    return out
