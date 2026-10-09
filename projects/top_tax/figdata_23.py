#!/usr/bin/env python3
"""
The numbers behind report figures 2 and 3: where the revenue comes from, in the
third decade (fiscal 2047-2056), as a share of GDP, on all three rungs — before
any response, after the corporate and payroll income adjustments, and after the
household behavioral response.

Each figure is a stack of bars built by adding one policy at a time, so each bar
after the first shows only what that addition changes. Revenue raised by taxing
gains held until death is moved out of the income-tax bar and into the
capital-gains bar, because economically it is a tax on capital gains.

Usage: python3 projects/top_tax/figdata_23.py <atlas2_data.json>

Reads only the fitted surrogate JSON.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_surrogate as F

# The destination ledger, in the order the surrogate stores it: individual
# income tax, capital gains, payroll, corporate, estate, wealth, everything else.
HEADS = ["iit", "cg", "pay", "corp", "est", "wealth", "other"]
DECADE_INDEX = 2  # the third decade, fiscal 2047-2056


def heads_raw(data, state, qid):
    """Third-decade revenue by head for one dial setting, as a share of GDP."""
    gdp = data["meta"]["gdp_fy_decades"][DECADE_INDEX]
    vec = F.eval_state(data, qid, state)  # 21 numbers: 3 decades x 7 heads
    seg = vec[DECADE_INDEX * len(HEADS):(DECADE_INDEX + 1) * len(HEADS)]
    return {h: 100.0 * v / gdp for h, v in zip(HEADS, seg)}


def heads_at(data, state, qid):
    """The same, with revenue from gains at death moved into the gains head.

    The gains-at-death dial's own contribution to the income-tax head is the
    difference between the income-tax head with the dial on and with it off.
    That amount is subtracted from the income-tax head and added to the
    capital-gains head.
    """
    h = heads_raw(data, state, qid)
    if "deemed" in state:
        without_death_dial = {k: v for k, v in state.items() if k != "deemed"}
        portion = h["iit"] - heads_raw(data, without_death_dial, qid)["iit"]
        h["iit"] -= portion
        h["cg"] += portion
    return {k: round(v, 3) for k, v in h.items()}


def sub(a, b):
    return {h: round(a[h] - b[h], 3) for h in a}


def nonzero(d):
    """Drop heads that round to zero at the third decimal place of a GDP share."""
    return {h: v for h, v in d.items() if abs(v) >= 0.0005}


def main(path):
    with open(path) as fh:
        data = json.load(fh)
    cg25 = {"cg": {"rate": 25.0}}
    cg25corp25 = {"cg": {"rate": 25.0}, "corp": {"rate": 25.0}}
    death = {"deemed": {"pos": "deemed"}}
    death_cg = {"deemed": {"pos": "deemed"}, "cg": {"rate": 25.0}}
    death_cg_corp = {"deemed": {"pos": "deemed"}, "cg": {"rate": 25.0},
                     "corp": {"rate": 25.0}}

    def show(label, state, minus=None):
        """One bar, on all three rungs.

        The surrogate stores the head vectors under three names: sh before any
        response, mh after the corporate and payroll income adjustments, ch
        after the household behavioral response.
        """
        out = {}
        for name, qid in (("before response", "sh"),
                          ("plus income adjustments", "mh"),
                          ("after response", "ch")):
            h = heads_at(data, state, qid)
            if minus:
                h = sub(h, heads_at(data, minus, qid))
            out[name] = h
        for name in ("before response", "plus income adjustments", "after response"):
            tag = label if name == "before response" else ""
            print(f"  {tag:16} {name:24} {nonzero(out[name])}")

    print(f"### {os.path.basename(path)}  (third decade, fiscal 2047-2056, % of GDP)")
    print("FIGURE 2 (step-up basis at death kept):")
    show("Capital gains", cg25)
    show("+ Corporate", cg25corp25, minus=cg25)
    show("Package", cg25corp25)
    print("FIGURE 3 (deemed realization at death):")
    show("Gains at death", death)
    show("+ Capital gains", death_cg, minus=death)
    show("+ Corporate", death_cg_corp, minus=death_cg)
    show("Package", death_cg_corp)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: figdata_23.py <atlas2_data.json>")
    main(sys.argv[1])
