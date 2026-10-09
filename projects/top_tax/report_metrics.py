#!/usr/bin/env python3
"""
The numbers for the top-tax report, printed as markdown to standard output.

Three inputs:
  - a dial run directory: the model run over config/runscripts/top_tax/dials.csv,
    which holds the baseline, the eight dials on their own, the pairs and
    triples, the holdout packages and the full eight-dial stack;
  - a revmax run directory: the capital-gains rate by death-regime grid.
    For the report's figure 4 this is the gains-only grid, a higher rate on
    long-term gains alone with qualified dividends at current law
    (config/runscripts/top_tax/revmax_gains_only.csv and
    revmax_gains_only_death.csv, merged into one run directory). The grid
    that raises the preferential rate on gains and dividends together
    (config/runscripts/top_tax/revmax_cg.csv) gives a revenue-maximizing
    rate about ten points higher, because dividends have no realization
    response; section 8a, composed from the dial runs, is on that basis;
  - atlas2_data.json, the fitted surrogate over the dial runs.

Usage:
    python3 projects/top_tax/report_metrics.py DIAL_RUN REVMAX_RUN [--atlas PATH]

All revenue figures are billions of dollars, scenario minus baseline, over the
fiscal-year windows 2027-2036 ("ten years") and 2027-2056 ("thirty years")
unless a heading says otherwise. Constants that are not model output are marked
EXTERNAL with their source.

Every run-directory read goes through readers.py, which converts the model's
raw dollars to billions. Section 11 needs a per-scenario realization-model
diagnostic file that this model does not write; it says so and prints nothing.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readers

# Result legs. readers.py accepts either the leg name or the model's world
# name.
ASK_LEG = "static"                # the model's first_order world: reform law,
                                  # nothing else changed
COLLECTED_LEG = "conventional"    # after households respond

# The atlas's two income definitions, as readers.etr_rows labels them:
# "expanded" is the cash measure, "hs" the Haig-Simons (accrual) measure.
CASH_INCOME_DEF = "expanded"
ACCRUAL_INCOME_DEF = "hs"

W10 = range(2027, 2037)
W30 = range(2027, 2057)
DECADES = [(2027, 2036), (2037, 2046), (2047, 2056)]

# EXTERNAL: CBO February 2026 baseline, via the Committee for a Responsible
# Federal Budget. Cumulative fiscal 2027-2036 deficits, and the single-year
# deficits at each end of the window. Billions of dollars.
CUM_DEFICIT_10Y_B = 24_400.0
DEFICIT_2026_B = 1_900.0
DEFICIT_2036_B = 3_100.0

# Things this report cannot compute from the current model, collected as it
# goes and printed at the end.
GAPS = []


def note_gap(text):
    if text not in GAPS:
        GAPS.append(text)


def wsum(d, win):
    return sum(v for y, v in d.items() if y in win)


def fmt(x, nd=1):
    return f"{x:,.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dial_run", help="model run directory for the dial batch")
    ap.add_argument("revmax_run",
                    help="model run directory for the capital-gains revmax grid")
    ap.add_argument("--atlas",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "atlas2_data.json"),
                    help="fitted surrogate JSON (default: alongside this script)")
    args = ap.parse_args()

    dial_run, revmax_run = args.dial_run, args.revmax_run
    for path in (dial_run, revmax_run):
        if not os.path.isdir(os.path.join(path, "scenarios")):
            sys.exit(f"not a model run directory (no scenarios/ folder): {path}")
    if not os.path.exists(args.atlas):
        sys.exit(f"surrogate JSON not found: {args.atlas}\n"
                 "Fit it first with fit_surrogate.py, or pass --atlas.")

    def rev(run, scen, world):
        """{fiscal year: revenue change in billions}, whole-budget."""
        out = readers.official_estimate(run, scen, world)
        if not out:
            raise KeyError(f"{scen}: no revenue estimate for world {world!r}")
        return out

    # ------------------------------------------------------------- atlas load
    with open(args.atlas) as f:
        atlas = json.load(f)
    # A fit made from a run without the distribution table carries no
    # income levels or baseline rates; sections 1 and 2 then say so.
    etr_available = "income_levels" in atlas and "etr_base" in atlas
    inc_lv = atlas.get("income_levels")      # {income definition: {group: $B}}, 2027
    etrb = atlas.get("etr_base")             # {def: {"2027": {group: [components]}}}
    comps = atlas["meta"].get("etr_comps", [])
    sur = atlas["surrogate"]
    gdp_dec = atlas["meta"]["gdp_fy_decades"]  # cumulative fiscal-year GDP per decade, $B
    placeholder = atlas["meta"].get("placeholder")

    print("# Metrics for the top-tax report")
    print()
    print(f"*Dial runs: `{os.path.basename(os.path.normpath(dial_run))}`. "
          f"Capital-gains revmax grid: `{os.path.basename(os.path.normpath(revmax_run))}`. "
          f"Surrogate: `{os.path.basename(args.atlas)}`. "
          "All dollar figures are billions unless noted; windows are fiscal 2027-2036 "
          "and fiscal 2027-2056.*")
    print()
    if placeholder:
        print(f"> **Surrogate carries a placeholder:** {placeholder}")
        print()
        note_gap("The surrogate JSON is marked as carrying placeholder data "
                 f"({placeholder}) — sections 1, 2, 7d, 7e and 8a inherit it.")

    # ================================================================ 1. incomes
    if not etr_available:
        print("## 1. Income at the top, 2027 (baseline)")
        print()
        print("*(not available: this run has no distribution table)*")
        print()
        print("## 2. Baseline federal taxes and effective rates by group, 2027")
        print()
        print("*(not available: this run has no distribution table)*")
        print()
        note_gap("Sections 1 and 2 need the distribution table, which "
                 "this model does not write.")
    else:
        print("## 1. Income at the top, 2027 (baseline)")
        print()
        # The atlas data drops the negative-income group, whose rate cannot
        # be read as an effective rate, so the population total here is the
        # five quintiles. The group's income is negative and small.
        whole_population = [g for g in ["Negative income", "Quintile 1", "Quintile 2",
                                        "Quintile 3", "Quintile 4", "Quintile 5"]
                            if g in inc_lv["expanded"]]
        tot_c = sum(inc_lv["expanded"][g] for g in whole_population)
        tot_h = sum(inc_lv["hs"][g] for g in whole_population)
        print("| Group | Cash (expanded) $B | Share of cash income | "
              "Accrual (Haig-Simons) $B | Share of accrual income | Accrual/cash |")
        print("|---|---|---|---|---|---|")
        groups_show = ["Quintile 5", "Top 10%", "Top 5%", "Top 1%", "Top 0.1%", "Top 0.01%"]
        for g in groups_show:
            c, h = inc_lv["expanded"][g], inc_lv["hs"][g]
            print(f"| {g} | {fmt(c,0)} | {100*c/tot_c:.1f}% | {fmt(h,0)} | "
                  f"{100*h/tot_h:.1f}% | {h/c:.2f} |")
        print()
        print(f"- Total income, all tax units 2027: cash **${fmt(tot_c/1000,1)}T**, "
              f"accrual **${fmt(tot_h/1000,1)}T** (shares are of these totals, which "
              "include the negative-income group).")

        # How far down the distribution you have to go to match the top 1 percent.
        quintile_tops = [("Quintile 1", 20), ("Quintile 2", 40),
                         ("Quintile 3", 60), ("Quintile 4", 80)]

        def bottom_share_matching_top_one_percent(levels, label):
            target = levels["Top 1%"]
            cum, prev_pct = 0.0, 0
            for g, pct in quintile_tops:
                prev_cum = cum
                cum += levels[g]
                if cum >= target:
                    frac = (target - prev_cum) / levels[g]
                    xpct = prev_pct + frac * (pct - prev_pct)
                    print(f"- {label} (${fmt(target,0)}B) equals the combined income of "
                          f"roughly the **bottom {xpct:.0f}%** of tax units "
                          "(interpolated within the quintile it lands in).")
                    return
                prev_pct = pct

        bottom_share_matching_top_one_percent(inc_lv["expanded"], "Top 1% cash income")
        bottom_share_matching_top_one_percent(inc_lv["hs"], "On an accrual basis the top 1%")
        print()

        # ============================================= 2. baseline effective rates
        print("## 2. Baseline federal taxes and effective rates by group, 2027")
        print()
        print(f"(taxes included: {readers.TAXES_DESCRIPTION}.)")
        print()
        print("| Group | Taxes $B | Effective rate, cash | Effective rate, accrual | "
              "of which: income tax | payroll | estate tax | tax on gains at death | "
              "corporate |")
        print("|---|---|---|---|---|---|---|---|---|")
        for g in ["Quintile 3", "Quintile 4", "Quintile 5", "Top 10%", "Top 5%",
                  "Top 1%", "Top 0.1%", "Top 0.01%"]:
            v_c = etrb["expanded"]["2027"][g]
            v_h = etrb["hs"]["2027"][g]
            etr_c, etr_h = sum(v_c), sum(v_h)
            taxes = etr_c / 100.0 * inc_lv["expanded"][g]
            by_comp = {c: v_c[k] for k, c in enumerate(comps)}
            print(f"| {g} | {fmt(taxes,0)} | {etr_c:.1f}% | {etr_h:.1f}% | "
                  f"{by_comp['income_tax']:.1f}% | {by_comp['payroll']:.1f}% | "
                  f"{by_comp['estate']:.1f}% | {by_comp['deemed']:.1f}% | "
                  f"{by_comp['corp']:.1f}% |")
        print()
        print("(The two taxes owed at death are reported apart: the estate tax is a tax "
              "on the transfer, and the tax on gains deemed realized at death is the "
              "income tax the stepped-up basis rule forgives today. Adding the two "
              "columns gives what earlier versions of this table showed as one "
              "'tax at death' figure.)")
        print()

        # ================================ 3. thresholds and levels from the run
    print("## 3. Group thresholds and 2036 levels (baseline columns of the "
          "distribution table)")
    print()
    levels = {}
    # readers.etr_rows returns one row per (year, income definition, group) with
    # both the baseline and the reform side, for the two report years.
    try:
        stack_rows = readers.etr_rows(dial_run, "stack_ref", ASK_LEG)
    except (FileNotFoundError, OSError):
        stack_rows = []
    try:
        cutoffs = readers.income_cutoffs(dial_run, "stack_ref", 2027)
    except (FileNotFoundError, OSError):
        cutoffs = {}
        note_gap("The distribution table is not in this run directory, so the "
                 "income floor that defines each group is blank in section 3.")

    def def_row(rows, group, year, income_def):
        for r in rows:
            if (r["group"] == group and int(r["year"]) == year
                    and r["income_definition"] == income_def):
                return r
        return None

    def cash_row(rows, group, year):
        return def_row(rows, group, year, CASH_INCOME_DEF)

    def accrual_row(rows, group, year):
        return def_row(rows, group, year, ACCRUAL_INCOME_DEF)

    # The same taxes measured against Haig-Simons income instead of cash
    # income. The tax dollars are identical — only the denominator changes.
    accrual_levels = {}
    if not stack_rows:
        print("*(the full-stack scenario `stack_ref` is not in this run directory)*")
        print()
    else:
        print("| Group | 2027 income floor $ | 2027 cash income $B | 2036 cash income $B | "
              "2027 taxes $B | 2036 taxes $B | n (2027, millions) |")
        print("|---|---|---|---|---|---|---|")
        for g in ["Top 10%", "Top 5%", "Top 1%", "Top 0.1%", "Top 0.01%"]:
            r27 = cash_row(stack_rows, g, 2027)
            r36 = cash_row(stack_rows, g, 2036)
            if r27 is None or r36 is None:
                continue
            levels[g] = dict(inc27=r27["income_baseline"],
                             inc36=r36["income_baseline"],
                             tax27=r27["tax_baseline"],
                             tax36=r36["tax_baseline"],
                             cut27=cutoffs.get(g, float("nan")),
                             n27=float(r27["n_tax_units"]))
            v = levels[g]
            print(f"| {g} | {fmt(v['cut27'],0)} | {fmt(v['inc27'],0)} | {fmt(v['inc36'],0)} | "
                  f"{fmt(v['tax27'],0)} | {fmt(v['tax36'],0)} | {v['n27']/1e6:.2f} |")
        print()
        print("Accrual (Haig-Simons) income for the same groups: gains are counted as "
              "they accrue rather than when they are sold, so the same taxes are a "
              "smaller share of a larger income.")
        print()
        print("| Group | 2027 accrual income $B | 2036 accrual income $B | "
              "Accrual/cash, 2027 | Effective rate, cash 2027 | "
              "Effective rate, accrual 2027 |")
        print("|---|---|---|---|---|---|")
        for g in ["Top 10%", "Top 5%", "Top 1%", "Top 0.1%", "Top 0.01%"]:
            h27 = accrual_row(stack_rows, g, 2027)
            h36 = accrual_row(stack_rows, g, 2036)
            if h27 is None or h36 is None or g not in levels:
                continue
            accrual_levels[g] = dict(inc27=h27["income_baseline"],
                                     inc36=h36["income_baseline"],
                                     tax27=h27["tax_baseline"],
                                     tax36=h36["tax_baseline"])
            h, v = accrual_levels[g], levels[g]
            print(f"| {g} | {fmt(h['inc27'],0)} | {fmt(h['inc36'],0)} | "
                  f"{h['inc27']/v['inc27']:.2f} | "
                  f"{100*v['tax27']/v['inc27']:.1f}% | "
                  f"{100*h['tax27']/h['inc27']:.1f}% |")
        print()

    # ================================= 4. stylized doubling of the top 1% rate
    print("## 4. Stylized calculation: doubling the top 1 percent effective rate")
    print()
    if "Top 1%" in levels:
        v = levels["Top 1%"]
        # Grow both income and taxes at the constant rate implied by their own
        # 2027-to-2036 path, then sum over the ten-year window.
        grow_i = (v["inc36"] / v["inc27"]) ** (1 / 9)
        grow_t = (v["tax36"] / v["tax27"]) ** (1 / 9)
        inc10 = sum(v["inc27"] * grow_i ** k for k in range(10))
        tax10 = sum(v["tax27"] * grow_t ** k for k in range(10))
        etr27 = v["tax27"] / v["inc27"]
        print(f"- Top 1% baseline cash effective rate in 2027, for this set of taxes: "
              f"**{100*etr27:.1f}%**; taxes ${fmt(v['tax27'],0)}B on "
              f"${fmt(v['inc27'],0)}B of income.")
        print(f"- Ten-year (fiscal 2027-2036) top 1% cash income, growing at the rate "
              f"implied by 2027 to 2036: **${fmt(inc10/1000,1)}T**; taxes "
              f"${fmt(tax10/1000,1)}T.")
        print(f"- Doubling the effective rate ({100*etr27:.1f}% to {200*etr27:.1f}%) with "
              f"the base held fixed adds revenue equal to baseline taxes: "
              f"**+${fmt(tax10/1000,1)}T over ten years**.")
        print(f"- EXTERNAL yardstick: cumulative fiscal 2027-2036 deficits "
              f"**${fmt(CUM_DEFICIT_10Y_B/1000,1)}T** (CBO February 2026 baseline, via "
              "the Committee for a Responsible Federal Budget). Single-year deficits at "
              f"the ends of the window: ${fmt(DEFICIT_2026_B/1000,1)}T in fiscal 2026 "
              f"and ${fmt(DEFICIT_2036_B/1000,1)}T in fiscal 2036.")
        print(f"- So the fixed-base doubling closes "
              f"**{100*tax10/CUM_DEFICIT_10Y_B:.0f}%** of the ten-year deficit.")
        if "Top 1%" in accrual_levels:
            # The same arithmetic on the accrual denominator. The taxes are
            # the same dollars, so what changes is the rate being doubled and
            # the income it is doubled against — and doubling a rate with the
            # base held fixed still adds exactly baseline taxes, whichever
            # income the rate was measured against. The interesting number is
            # therefore the rate itself, which is lower on accrual income.
            h = accrual_levels["Top 1%"]
            grow_hi = (h["inc36"] / h["inc27"]) ** (1 / 9)
            hinc10 = sum(h["inc27"] * grow_hi ** k for k in range(10))
            hetr27 = h["tax27"] / h["inc27"]
            print(f"- On an accrual (Haig-Simons) denominator the same taxes are "
                  f"**{100*hetr27:.1f}%** of top 1% income in 2027 — ${fmt(h['tax27'],0)}B "
                  f"on ${fmt(h['inc27'],0)}B — against {100*etr27:.1f}% on cash income.")
            print(f"- Ten-year top 1% accrual income on the same growth rule: "
                  f"**${fmt(hinc10/1000,1)}T**, against ${fmt(inc10/1000,1)}T of cash "
                  f"income.")
            print(f"- Doubling the accrual rate ({100*hetr27:.1f}% to "
                  f"{200*hetr27:.1f}%) with the base held fixed raises the same "
                  f"**+${fmt(tax10/1000,1)}T over ten years**, because it is the same "
                  "taxes doubled; the accrual measure changes how large a rate that "
                  "is, not how much money it is.")
    else:
        print("*(needs the baseline levels in section 3)*")
    print()

    # ================================================ 5. one dial at a time
    print("## 5. Standalone dial scores (direct runs)")
    print()
    solos = [
        ("s_ord_r39p6",        "Top ordinary rate 37% -> 39.6%"),
        ("s_ord_r44p8",        "Top ordinary rate 37% -> 44.8% (reference)"),
        ("s_ord_r50",          "Top ordinary rate 37% -> 50%"),
        ("s_cg_r25",           "Capital gains and dividends top rate 20% -> 25%"),
        ("s_cg_r30",           "Capital gains and dividends top rate 20% -> 30%"),
        ("s_cg_r35",           "Capital gains and dividends top rate 20% -> 35%"),
        ("s_cg_r40",           "Capital gains and dividends top rate 20% -> 40% (reference)"),
        ("s_cg_r45",           "Capital gains and dividends top rate 20% -> 45%"),
        ("s_cg_r50",           "Capital gains and dividends top rate 20% -> 50%"),
        ("s_corp_r24p5",       "Corporate rate 21% -> 24.5%"),
        ("s_corp_r28",         "Corporate rate 21% -> 28% (reference)"),
        ("s_corp_r31p5",       "Corporate rate 21% -> 31.5%"),
        ("s_corp_r35",         "Corporate rate 21% -> 35%"),
        ("s_wealth_r1_t50",    "Wealth tax 1% above $50M"),
        ("s_wealth_r2_t50",    "Wealth tax 2% above $50M (reference)"),
        ("s_deemed_carryover", "Carryover basis at death (no rate change)"),
        ("s_deemed_deemed",    "Deemed realization at death (no rate change)"),
        ("s_estate_r50_e8p46", "Estate tax 40% -> 50%, exemption $15.4M -> $8.46M (reference)"),
        ("s_estate_r60_e5",    "Estate tax 40% -> 60%, exemption -> $5M"),
        ("s_qbi_on",           "Repeal the qualified business income deduction (§199A)"),
        ("s_taxmax_on",        "Eliminate the Social Security taxable maximum"),
    ]
    print("| Dial | 10y before response | 10y after response | after/before | "
          "30y after response |")
    print("|---|---|---|---|---|")
    solo_res = {}
    for sid, label in solos:
        try:
            ask = rev(dial_run, sid, ASK_LEG)
            got = rev(dial_run, sid, COLLECTED_LEG)
        except (FileNotFoundError, KeyError, OSError):
            print(f"| {label} | (missing: {sid}) | | | |")
            continue
        s10, c10, c30 = wsum(ask, W10), wsum(got, W10), wsum(got, W30)
        solo_res[sid] = dict(s10=s10, c10=c10, s30=wsum(ask, W30), c30=c30)
        ratio = c10 / s10 if abs(s10) > 1e-9 else float("nan")
        print(f"| {label} | {fmt(s10,0)} | {fmt(c10,0)} | {ratio:.2f} | {fmt(c30,0)} |")
    print()
    print("(\"Before response\" is the reform law applied with nothing else changed; "
          "\"after response\" is with the household behavioral response.)")
    print()

    # ============================== 6. where the money comes from, by budget line
    print("## 6. Destination ledgers — revenue change by budget line, fiscal 2027-2036, "
          "after response ($B)")
    print()
    lines_pretty = OrderedDict([
        ("revenues_income_tax",  "income tax"),
        ("revenues_payroll_tax", "payroll"),
        ("revenues_corp_tax",    "corporate"),
        ("revenues_estate_tax",  "estate"),
        ("revenues_wealth_tax",  "wealth"),
        ("outlays_tax_credits",  "credits (outlay)"),
    ])
    feature = ["s_cg_r40", "s_cg_r25", "s_corp_r28", "s_wealth_r2_t50",
               "s_wealth_r1_t50", "s_deemed_deemed", "s_ord_r44p8",
               "s_estate_r50_e8p46", "s_taxmax_on", "stack_ref"]
    print("| Scenario | " + " | ".join(lines_pretty.values()) + " | TOTAL |")
    print("|---|" + "---|" * (len(lines_pretty) + 1))
    for sid in feature:
        try:
            by_year = readers.revenue_estimate_lines(dial_run, sid, COLLECTED_LEG)
        except (FileNotFoundError, OSError):
            continue
        window = OrderedDict()
        for key in list(lines_pretty) + ["total_budget_effect"]:
            window[key] = sum(by_year[y].get(key, 0.0) for y in W10 if y in by_year)
        cells = " | ".join(fmt(window[k], 0) for k in lines_pretty)
        print(f"| {sid} | {cells} | {fmt(window['total_budget_effect'],0)} |")
    print()
    print("(The TOTAL column is the model's own whole-budget line, not a sum of the "
          "columns shown: the 'other' line is left out of the table because no dial "
          "moves it.)")
    print()

    # ========================================== 7. the parts against the whole
    print("## 7. Parts against the whole (naive sum versus the actual package)")
    print()

    def parts_vs_whole(label, part_ids, whole_id):
        missing = [p for p in part_ids if p not in solo_res]
        if missing:
            print(f"**{label}**: not available (missing dial runs {missing})")
            print()
            return
        try:
            ask = rev(dial_run, whole_id, ASK_LEG)
            got = rev(dial_run, whole_id, COLLECTED_LEG)
        except (FileNotFoundError, KeyError, OSError):
            print(f"**{label}**: not available (missing package run {whole_id})")
            print()
            return
        ps10 = sum(solo_res[p]["s10"] for p in part_ids)
        pc10 = sum(solo_res[p]["c10"] for p in part_ids)
        pc30 = sum(solo_res[p]["c30"] for p in part_ids)
        ws10, wc10, wc30 = wsum(ask, W10), wsum(got, W10), wsum(got, W30)
        print(f"**{label}** (ten years unless noted)")
        print()
        print(f"- Sum of the standalone scores before response:  ${fmt(ps10,0)}B")
        print(f"- Sum of the standalone scores after response:   ${fmt(pc10,0)}B")
        print(f"- The package before response:                   ${fmt(ws10,0)}B")
        print(f"- The package after response (the answer):       ${fmt(wc10,0)}B")
        print(f"- Interaction (package minus the sum of parts, both after response): "
              f"${fmt(wc10-pc10,0)}B ({100*(wc10-pc10)/pc10:+.1f}% of the naive sum)")
        print(f"- Share of the pre-response score that survives the behavioral "
              f"response: naive {100*pc10/ps10:.0f}%, package {100*wc10/ws10:.0f}%")
        print(f"- Thirty years: naive ${fmt(pc30,0)}B against the package's "
              f"${fmt(wc30,0)}B ({100*(wc30-pc30)/pc30:+.1f}%)")
        print()

    reference_solos = ["s_ord_r44p8", "s_cg_r40", "s_corp_r28", "s_wealth_r2_t50",
                       "s_deemed_deemed", "s_estate_r50_e8p46", "s_qbi_on", "s_taxmax_on"]
    parts_vs_whole("The full eight-dial reference stack against the sum of its eight "
                   "standalone scores", reference_solos, "stack_ref")
    parts_vs_whole("Corporate 21% -> 35% plus capital gains 20% -> 30%",
                   ["s_corp_r35", "s_cg_r30"], "pc_corpr35_cgr30")
    parts_vs_whole("Capital gains 20% -> 40% plus deemed realization at death",
                   ["s_cg_r40", "s_deemed_deemed"], "pr_cg_deemed")

    # 7d and 7e use the surrogate to price combinations that were never run.
    print("**Corporate and capital gains both to 25 percent — surrogate estimate, "
          "no direct run**")
    print()
    if "s_cg_r25" in solo_res and "s_corp_r28" in solo_res:
        c25_s10 = solo_res["s_cg_r25"]["s10"]
        c25_c10 = solo_res["s_cg_r25"]["c10"]
        # The corporate dial's revenue is linear in the rate change by
        # construction, so 25% is 4/7 of the way from 21% to the 28% reference.
        corp_frac = (25.0 - 21.0) / (28.0 - 21.0)
        k25_s10 = corp_frac * solo_res["s_corp_r28"]["s10"]
        k25_c10 = corp_frac * solo_res["s_corp_r28"]["c10"]
        naive_s, naive_c = c25_s10 + k25_s10, c25_c10 + k25_c10
        pair = sur["pairs"].get("cg|corp", {})
        inter10 = 0.0
        if "ct" in pair:
            g_cg = sur["g"]["cg"][1][0]     # capital-gains knot 1 is 25%, first decade
            inter10 = g_cg * corp_frac * pair["ct"][0]
        print(f"- Sum of standalone scores before response: ${fmt(naive_s,0)}B "
              f"(gains at 25% {fmt(c25_s10,0)} plus corporate at 25% {fmt(k25_s10,0)}, "
              "the corporate piece scaled to 4/7 of the 28% run)")
        print(f"- Sum of standalone scores after response:  ${fmt(naive_c,0)}B")
        print(f"- The package after response (surrogate: the sum plus the scaled "
              f"gains-corporate interaction of {fmt(inter10,1)}): "
              f"**${fmt(naive_c + inter10,0)}B**")
        print("- Treat surrogate-composed figures as good to a few percent; the "
              "validation report says how far off the holdout packages were.")
    else:
        print("*(needs the standalone gains and corporate runs)*")
    print()

    print("**Capital gains 20% -> 25% plus deemed realization at death**")
    print()
    if "s_cg_r25" in solo_res and "s_deemed_deemed" in solo_res:
        c25_s10 = solo_res["s_cg_r25"]["s10"]
        c25_c10 = solo_res["s_cg_r25"]["c10"]
        d_s10 = solo_res["s_deemed_deemed"]["s10"]
        d_c10 = solo_res["s_deemed_deemed"]["c10"]
        naive_s4, naive_c4 = c25_s10 + d_s10, c25_c10 + d_c10
        pair_cd = sur["pairs"].get("cg|deemed", {})
        inter4 = 0.0
        by_pos = pair_cd.get("byPos", {})
        if "deemed" in by_pos and "ct" in by_pos["deemed"]:
            inter4 = sur["g"]["cg"][1][0] * by_pos["deemed"]["ct"][0]
        elif "ct" in pair_cd:
            inter4 = sur["g"]["cg"][1][0] * pair_cd["ct"][0]
        print("- Surrogate composition at a 25 percent gains rate:")
        print(f"  - Sum of standalone scores before response: ${fmt(naive_s4,0)}B "
              f"(gains {fmt(c25_s10,0)} plus gains-at-death {fmt(d_s10,0)})")
        print(f"  - Sum of standalone scores after response:  ${fmt(naive_c4,0)}B")
        print(f"  - The package after response: **${fmt(naive_c4 + inter4,0)}B** "
              f"(interaction {fmt(inter4,1)})")
    try:
        su_ask = rev(revmax_run, "cg_05pp_stepup", ASK_LEG)
        su_got = rev(revmax_run, "cg_05pp_stepup", COLLECTED_LEG)
        d0_ask = rev(revmax_run, "cg_00pp_deemed", ASK_LEG)
        d0_got = rev(revmax_run, "cg_00pp_deemed", COLLECTED_LEG)
        d5_ask = rev(revmax_run, "cg_05pp_deemed", ASK_LEG)
        d5_got = rev(revmax_run, "cg_05pp_deemed", COLLECTED_LEG)
        print("- Direct runs from the revmax grid, ten years:")
        print(f"  - Parts before response: gains +5pp {fmt(wsum(su_ask,W10),0)} plus "
              f"deemed realization {fmt(wsum(d0_ask,W10),0)} = "
              f"${fmt(wsum(su_ask,W10)+wsum(d0_ask,W10),0)}B")
        print(f"  - Parts after response:  gains +5pp {fmt(wsum(su_got,W10),0)} plus "
              f"deemed realization {fmt(wsum(d0_got,W10),0)} = "
              f"${fmt(wsum(su_got,W10)+wsum(d0_got,W10),0)}B")
        print(f"  - The package before response ${fmt(wsum(d5_ask,W10),0)}B; "
              f"after response **${fmt(wsum(d5_got,W10),0)}B**")
    except (FileNotFoundError, KeyError, OSError) as e:
        print(f"  (the revmax direct runs are not available: {e})")
    print()

    # ============================ 8. capital-gains Laffer curves by death regime
    print("## 8. Capital-gains Laffer curves by death regime")
    print()
    print("### 8a. Step-up basis directly; deemed realization and carryover composed "
          "from the surrogate")
    print()
    knots = [25, 30, 35, 40, 45, 50]
    print("| Top gains rate | Step-up, 10y after response | With deemed realization "
          "already in place* | With carryover already in place* | Step-up, 30y after "
          "response |")
    print("|---|---|---|---|---|")
    for i, k in enumerate(knots):
        sid = f"s_cg_r{k}"
        if sid not in solo_res:
            continue
        c10, c30 = solo_res[sid]["c10"], solo_res[sid]["c30"]
        row_idx = i + 1  # row 0 of the curve is the off state, a 20 percent rate
        conditional = {}
        pair_cd = sur["pairs"].get("cg|deemed", {})
        for pos in ("deemed", "carryover"):
            by_pos = pair_cd.get("byPos", {})
            if pos in by_pos and "ct" in by_pos[pos]:
                conditional[pos] = c10 + sur["g"]["cg"][row_idx][0] * by_pos[pos]["ct"][0]
            else:
                conditional[pos] = float("nan")
        print(f"| {k}% | {fmt(c10,0)} | {fmt(conditional['deemed'],0)} | "
              f"{fmt(conditional['carryover'],0)} | {fmt(c30,0)} |")
    print()
    print("*The two conditional columns are what the gains-rate increase alone raises "
          "over ten years when the death regime is already law: the standalone gains "
          "score plus the scaled gains-by-death-regime interaction. The regime's own "
          "revenue is not included. These are surrogate-composed and good to a few "
          "percent.*")
    print()

    print("### 8b. The direct grid: total ten-year revenue after response, by cell")
    print()
    print("| Top gains rate | Step-up | Carryover | Deemed | Step-up, before response | "
          "Deemed, before response |")
    print("|---|---|---|---|---|---|")
    for pp in [0, 5, 10, 15, 20, 25, 30]:
        rate = 20 + pp
        row = [f"| {rate}%"]
        for regime in ["stepup", "carryover", "deemed"]:
            if pp == 0 and regime == "stepup":
                row.append("0")            # this cell is the baseline itself
                continue
            try:
                row.append(fmt(wsum(rev(revmax_run, f"cg_{pp:02d}pp_{regime}",
                                        COLLECTED_LEG), W10), 0))
            except (FileNotFoundError, KeyError, OSError):
                row.append("--")
        for regime in ["stepup", "deemed"]:
            if pp == 0 and regime == "stepup":
                row.append("0")
                continue
            try:
                row.append(fmt(wsum(rev(revmax_run, f"cg_{pp:02d}pp_{regime}",
                                        ASK_LEG), W10), 0))
            except (FileNotFoundError, KeyError, OSError):
                row.append("--")
        print(" | ".join(row) + " |")
    print()
    print("Leakage check (revenue before response minus revenue after response, ten "
          "years; positive means the behavioral response costs revenue).")
    for sid, lab in [("cg_05pp_stepup", "gains +5pp under step-up basis"),
                     ("cg_00pp_deemed", "deemed realization alone"),
                     ("cg_05pp_deemed", "gains +5pp with deemed realization")]:
        try:
            ask = rev(revmax_run, sid, ASK_LEG)
            got = rev(revmax_run, sid, COLLECTED_LEG)
            print(f"- {lab}: before {fmt(wsum(ask,W10),0)}, after {fmt(wsum(got,W10),0)}, "
                  f"leakage {fmt(wsum(ask,W10)-wsum(got,W10),0)}")
        except (FileNotFoundError, KeyError, OSError):
            pass
    print()

    # ================== 9. what the reform asks for against what it collects
    print("## 9. What the full stack asks for against what it collects, 2027")
    print()
    print(f"(taxes are the {readers.TAXES_DESCRIPTION}, as in section 2)")
    print()
    # The "asked for" rows come from the pass where the reform law applies but
    # nothing else has changed; the "collected" rows from the pass with the
    # behavioral response. The baseline side of both is the same.
    try:
        ask_rows = readers.etr_rows(dial_run, "stack_ref", ASK_LEG)
        got_rows = readers.etr_rows(dial_run, "stack_ref", COLLECTED_LEG)
    except (FileNotFoundError, OSError):
        ask_rows = got_rows = []
    if ask_rows and got_rows:
        print("Cash denominator (expanded income).")
        print()
        print("| Group | Baseline | Asked for | Collected | Avoidance margin "
              "(percentage points) |")
        print("|---|---|---|---|---|")
        for g in ["Quintile 5", "Top 1%", "Top 0.1%", "Top 0.01%"]:
            ra = cash_row(ask_rows, g, 2027)
            rc = cash_row(got_rows, g, 2027)
            if ra is None or rc is None:
                continue
            b = 100 * ra["etr_baseline"]
            a = 100 * ra["etr_reform"]
            c = 100 * rc["etr_reform"]
            print(f"| {g} | {b:.1f}% | {a:.1f}% | {c:.1f}% | {a-c:.1f} |")
        print()
        print("Accrual denominator (Haig-Simons income). Every rate is lower because "
              "the income is larger; the avoidance margin is what the response costs, "
              "measured against that larger income.")
        print()
        print("| Group | Baseline | Asked for | Collected | Avoidance margin "
              "(percentage points) |")
        print("|---|---|---|---|---|")
        for g in ["Quintile 5", "Top 1%", "Top 0.1%", "Top 0.01%"]:
            ra = accrual_row(ask_rows, g, 2027)
            rc = accrual_row(got_rows, g, 2027)
            if ra is None or rc is None:
                continue
            b = 100 * ra["etr_baseline"]
            a = 100 * ra["etr_reform"]
            c = 100 * rc["etr_reform"]
            print(f"| {g} | {b:.1f}% | {a:.1f}% | {c:.1f}% | {a-c:.1f} |")
    else:
        print("*(the full-stack scenario `stack_ref` is not in this run directory)*")
    print()

    # ======================== 10. how the stack's revenue holds up over decades
    print("## 10. Decade profile of the full stack")
    print()
    try:
        ask = rev(dial_run, "stack_ref", ASK_LEG)
        got = rev(dial_run, "stack_ref", COLLECTED_LEG)
        print("| Decade | Before response $B | After response $B | Share surviving | "
              "After response, % of GDP |")
        print("|---|---|---|---|---|")
        for i, (a, b) in enumerate(DECADES):
            win = range(a, b + 1)
            s, c = wsum(ask, win), wsum(got, win)
            print(f"| {a}-{b} | {fmt(s,0)} | {fmt(c,0)} | {100*c/s:.0f}% | "
                  f"{100*c/gdp_dec[i]:.2f}% |")
    except (FileNotFoundError, KeyError, OSError):
        print("*(the full-stack scenario `stack_ref` is not in this run directory)*")
    print()

    # ================================ 11. color from the realization model
    print("## 11. Realization-model color")
    print()
    print("- **Not available from this model yet.** This section needs a per-scenario "
          "diagnostic file (kg_dynamics_summary.csv) holding the realization "
          "semi-elasticity the model's own choices implied, the gains realized at "
          "death, and the stock of taxable gains held until death. The Python model "
          "publishes no equivalent, so the point this section made — that the policy "
          "chooses its own elasticity, and you can read it off by comparing regimes — "
          "cannot be made from these runs.")
    note_gap("The realization model's diagnostics (implied semi-elasticity, gains "
             "realized at death, stock of taxable gains at death) are not published "
             "by this model, so section 11 is empty.")
    print()

    # ================================================= what is missing, listed
    print("## Not available from this model yet")
    print()
    if GAPS:
        for g in GAPS:
            print(f"- {g}")
    else:
        print("- Nothing: every section above was computed from the runs.")
    print()


if __name__ == "__main__":
    main()
