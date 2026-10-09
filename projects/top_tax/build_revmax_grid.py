#!/usr/bin/env python3
"""
Build the capital-gains rate by death-regime grid used to find the
revenue-maximizing top capital-gains rate.

The point of the grid: the realization elasticity in this model is not a fixed
number. It comes out of the household's own choice of when to realize, so it
depends on what happens to unrealized gains at death. Under step-up basis,
holding until death escapes the tax entirely, which makes realizations very
responsive to the rate and pushes the revenue-maximizing rate down. Under
carryover basis or deemed realization at death that escape is closed. The
revenue-maximizing rate should therefore differ by regime, and the size of that
spread is the finding.

Design (retune the grid below and re-run; it only writes files, so it is safe on
a login node):
  - Enactment in 2027, with the run starting in 2026 so the fiscal-year lead-in
    is there. One continuous run per scenario gives both the ten-year
    (FY2027-2036) and the thirty-year answer, because revenue totals are written
    for every year of the run.
  - The full behavioral stack: the behavior leg names
    the top_tax_full folder, the same one the dial batch uses.
  - No corporate provision, so nothing feeds the corporate channel.

The baseline top preferential rate is 20 percent statutory (23.8 percent
including the net investment income tax). A grid point of +pp sets the top
bracket to 0.20 + pp/100. The pp = 0 point under a regime other than step-up is
the "regime change alone, rate unchanged" anchor of each schedule.

Regenerating is safe:

    python3 projects/top_tax/build_revmax_grid.py && git status --short config/

An empty status is the pass. This script only adds and overwrites the folders
its own grid names; it never deletes, so the extra +2pp scenarios that were
added by hand and are not in the grid below survive untouched.
"""
import csv
import os

# The repository root, two directories up from projects/top_tax/.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAXLAW_DIR = os.path.join(REPO, "config/scenarios/tax_law/alternatives/tests/revmax")
RUNSCRIPT = os.path.join(REPO, "config/runscripts/tests/revmax_cg.csv")

ENACT = 2027
BASE_PREF = [0.0, 0.15, 0.20]          # the baseline preferential-rate schedule
# Treatment of unrealized gains at death: 0 = step-up basis (current law, so it
# writes nothing), 1 = carryover basis, 2 = deemed realization.
REGIMES = {"stepup": None, "carryover": 1, "deemed": 2}
PP_POINTS = [0, 5, 10, 15, 20, 25, 30]  # up to a 50 percent top rate; the
                                        # revenue peaks sit inside that range

BEHAVIOR = "top_tax_full"              # the same behavior folder the dials use
YEARS = "2026:2057"

REGIME_KEYS = ["equities", "pass_throughs", "primary_home", "other_home", "re_fund"]


def pref_yaml(pp, regime_code):
    top = round(BASE_PREF[2] + pp / 100.0, 4)
    lines = ["---"]
    if pp > 0:
        lines += [
            f"# CG top rate +{pp}pp (0.20 -> {top}); enactment {ENACT}",
            "rates:",
            "  value:",
            "    '2014': [0.0, 0.15, 0.20]",
            f"    '{ENACT}': [0.0, 0.15, {top}]",
            # Lift the rule that holds the preferential rate at or below the
            # ordinary rate, so a gains rate above the 37 percent top ordinary
            # rate actually binds.
            "no_ord_cap:",
            "  value: 1",
        ]
    else:
        lines += ["# CG rate unchanged (regime-only anchor of the schedule)"]
    if regime_code is not None:
        for k in REGIME_KEYS:
            lines += [f"kg_death_regime_{k}:",
                      "  value:",
                      "    '2014': 0",
                      f"    '{ENACT}': {regime_code}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(TAXLAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RUNSCRIPT), exist_ok=True)

    rows = [["baseline", "default", "default", "default", YEARS]]
    n_dirs = 0
    for regime, code in REGIMES.items():
        for pp in PP_POINTS:
            if regime == "stepup" and pp == 0:
                continue  # step-up basis with no rate change is the baseline
            sid = f"cg_{pp:02d}pp_{regime}"
            d = os.path.join(TAXLAW_DIR, sid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "pref.yaml"), "w") as f:
                f.write(pref_yaml(pp, code))
            n_dirs += 1
            rows.append([sid, f"tests/revmax/{sid}", "default", BEHAVIOR, YEARS])

    with open(RUNSCRIPT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ID", "tax_law", "economy", "behavior", "years"])
        w.writerows(rows)

    print(f"wrote {n_dirs} reform dirs under {TAXLAW_DIR}")
    print(f"wrote runscript {RUNSCRIPT} with {len(rows)} rows "
          f"(1 baseline + {len(rows)-1} scenarios)")


if __name__ == "__main__":
    main()
