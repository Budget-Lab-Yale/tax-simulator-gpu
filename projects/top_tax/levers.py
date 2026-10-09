#!/usr/bin/env python3
"""
The eight policy dials of the top-tax project, as data.

This module is pure data plus small functions that return YAML text. Nothing
here computes anything about taxes; it says what each dial is, what values it
takes, and what law file a given setting writes. It is imported by
build_dial_runs.py (which turns the dials into scenario folders and a
runscript) and by the surrogate fitting and validation scripts.

Eight dials, eleven parameters (the gains-at-death dial has both a treatment
position and a dollar exclusion). Every dial is off at current law and moves in
one direction — toward more tax. The surrogate that is fitted over these runs
has the form

    V(x) = sum_i f_i(x_i) + sum_{i<j} I_ij g_i g_j + sum_ijk T_ijk g_i g_j g_k

where f_i is the dial's own revenue curve through its anchor points, I_ij is a
pair interaction measured once with both dials at their reference settings, and
g_i(x) is the dial's strength at setting x: its ten-year revenue divided by its
ten-year revenue at the reference setting.
"""

import math
import os

import yaml

# The repository root, two directories up from projects/top_tax/.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --------------------------------------------------------------------------- #
# Run-level constants
# --------------------------------------------------------------------------- #
# Runs start in 2027, the year the reforms take effect, so the realization model
# sees no pre-enactment year. Fiscal 2027 therefore omits October-December 2026
# anticipation revenue. The window ends in 2056 because the inheritance data end
# in 2055 and the CBO receipts data are blank for 2057.
YEARS = "2027:2056"

# Behavior-leg alternatives: folders under config/scenarios/behavior/
# alternatives/ naming the response stack. top_tax_full is the capital-gains
# realization model, income conversion, entity shifting, evasion, the charitable
# response at an elasticity of -0.5, and the estate reporting response;
# top_tax_full_wealth adds wealth-tax concealment. What each contains, and the
# order the modules run in, is in the folder's behavior.yaml and in
# src/taxmodel/behavior/loader.py.
BEHAVIOR_BASE = "top_tax_full"
BEHAVIOR_WEALTH = "top_tax_full_wealth"

# The top-tax batches use no off-model estimates. Both dial states name the
# model's default economy.
#
# The corporate rate is scored on-model (src/taxmodel/years/corporate/
# scorer.py): the rate in corp.yaml is measured against the CBO corporate
# receipts level, mechanically as the (new rate minus old rate) times the
# baseline base, and conventionally with an avoidance elasticity. An economy
# alternative that also pinned an off-model corporate estimate would count the
# same rate increase twice — roughly $49 billion in 2027 rising to $96 billion,
# doubled — so no such pin is used.
CORP_ON_ECONOMY = "default"
CORP_OFF_ECONOMY = "default"

# --------------------------------------------------------------------------- #
# Current-law 2027 estate exemption, computed rather than hardcoded, because it
# is indexed. This repeats the model's own indexation arithmetic
# (src/taxmodel/law/indexation.py):
#     value(t) = floor(base_value * index(t-1) / index(base_year) / incr) * incr
# with base_value the 2026-regime published amount from the baseline estate.yaml
# and index the chained-CPI level series the IRS uses, from Macro-Projections.
# --------------------------------------------------------------------------- #
MACRO_ROOT = ("/nfs/roberts/project/pi_nrs36/shared/model_data/"
              "Macro-Projections/v3/2026022522/baseline")


def _ccpiu_irs(year):
    """The chained-CPI level the IRS indexes by, for one calendar year."""
    import csv
    for f in ("historical.csv", "projections.csv"):
        with open(os.path.join(MACRO_ROOT, f), newline="") as fh:
            for r in csv.DictReader(fh):
                if int(r["year"]) == year:
                    return float(r["ccpiu_irs"])
    raise KeyError(year)


def current_estate_exemption_2027():
    path = os.path.join(REPO, "config", "scenarios", "tax_law", "default", "estate.yaml")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    ex = raw["exemption"]
    base_value = ex["value"]["2026"]                    # the 2026 published amount
    base_year = ex["i_base_year"]["2026"]               # 2025
    incr = ex["i_increment"]                            # rounded to $10,000
    assert ex["i_direction"] == -1                      # rounding direction: floor
    factor = _ccpiu_irs(2026) / _ccpiu_irs(base_year)   # the index lags one year
    return math.floor(base_value * factor / incr) * incr


ESTATE_CUR = current_estate_exemption_2027()            # $15,390,000 on vintage 2026022522
# Two intermediate exemptions, evenly spaced between current law and $5 million,
# rounded to the $10,000 increment the statute indexes in.
ESTATE_M1 = round((2 * ESTATE_CUR + 5_000_000) / 3, -4)
ESTATE_M2 = round((ESTATE_CUR + 2 * 5_000_000) / 3, -4)

# --------------------------------------------------------------------------- #
# YAML text for each dial. Each function returns {filename: [text blocks]} for
# the dial at the given parameter values. When two dials write the same law
# file, the generator concatenates their blocks.
# --------------------------------------------------------------------------- #

# The corporate rate dial writes corp.yaml and nothing else. The full baseline
# year series is repeated because a subparameter override replaces the whole
# series, not just the years mentioned.
def corp_files(vals):
    r = _pct(vals["rate"])
    return {"corp.yaml": [f"""\
rate:
  value:
    '2014': 0.35
    '2018': 0.21
    '2027': {r}
"""]}


# Appended to pref.yaml, last, whenever the capital-gains dial is on. It lifts
# the rule that holds the preferential rate schedule at or below the ordinary
# schedule; without it a top preferential rate above the top ordinary rate
# cannot bind. Its position at the end of the file carries no meaning — it is
# where the original hand-written patch put it, and leaving it there is what
# keeps regenerating the tree byte-identical to what is checked in.
NO_ORD_CAP_BLOCK = """\
no_ord_cap:
  value: 1
"""


def _pct(x):
    """Rate in percent -> YAML decimal string (39.6 -> '0.396')."""
    return f"{x / 100:.6g}"


def ord_files(vals):
    r = _pct(vals["rate"])
    return {"ord.yaml": [f"""\
# top_tax dial ORD: top ordinary rate -> {vals['rate']}% from 2027 (top bracket only)
rates:
  value:
    '2014': [0.1, 0.15, 0.25, 0.28, 0.33, 0.35, 0.396]
    '2018': [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
    '2027': [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, {r}]
"""]}


def cg_files(vals):
    r = _pct(vals["rate"])
    return {"pref.yaml": [f"""\
# top_tax dial CG: top preferred (LTCG/qual-div) rate -> {vals['rate']}% from 2027
rates:
  value:
    '2014': [0.0, 0.15, 0.20]
    '2027': [0.0, 0.15, {r}]
"""]}


DEEMED_CODES = {"carryover": 1, "deemed": 2}


def deemed_files(vals):
    code = DEEMED_CODES[vals["pos"]]
    blk = [f"# top_tax dial DEEMED: {vals['pos']} at death, all asset classes, from 2027",
           "# (code 0=step-up baseline, 1=carryover, 2=deemed_realization)"]
    for asset in ("equities", "pass_throughs", "primary_home", "other_home", "re_fund"):
        blk.append(f"kg_death_regime_{asset}:")
        blk.append("  value:")
        blk.append("    '2014': 0")
        blk.append(f"    '2027': {code}")
    parts = ["\n".join(blk) + "\n"]
    # A per-decedent exclusion on top of the Section 121 home-sale exclusion:
    # the first dollars of gain at death keep their stepped-up basis, and the
    # married amount is double. Zero is current law for this parameter and
    # writes nothing at all, which is what keeps the scenario folders that
    # predate the exclusion axis byte-identical when the tree is regenerated.
    exem = float(vals.get("exem", 0.0))
    if exem:
        e = int(exem)
        parts.append(f"""\
# top_tax dial DEEMED exclusion: first ${e:,} of gain at death per decedent
# keeps step-up (on top of Sec. 121); married amount is double
kg_death_gain_excl_single:
  value:
    '2014': 0
    '2027': {e}
kg_death_gain_excl_married:
  value:
    '2014': 0
    '2027': {e * 2}
""")
    return {"pref.yaml": parts}


def wealth_files(vals):
    r, thr = _pct(vals["rate"]), int(vals["thr"])
    parts = [f"""\
# top_tax dial WEALTH: {vals['rate']}% annual net-worth tax above ${thr:,} from 2027
# (threshold applied per tax unit for all filing statuses)
rates:
  value:
    '2014': [0.0, 0.0]
    '2027': [0.0, {r}]
"""]
    for status in ("single", "married", "head"):
        parts.append(f"""\
brackets_{status}:
  value:
    '2014': [0, {thr}]
    '2027': [0, {thr}]
  i_measure: default
  i_base_year: default
  i_direction: default
  i_increment: default
""")
    return {"wealth.yaml": ["\n".join(parts)]}


def estate_files(vals):
    blocks = []
    if vals["exem"] != ESTATE_CUR:
        e = int(vals["exem"])
        blocks.append(f"""\
# top_tax dial ESTATE exemption -> ${e:,} at 2027, chained-CPI indexed after
# (re-anchored 2027 regime; pre-2027 path = current law)
exemption:
  value:
    '2014': 5340000
    '2018': 11180000
    '2026': 15000000
    '2027': {e}
  i_measure:
    '2013': cpi
    '2018': chained_cpi
  i_base_year:
    '2014': 2013
    '2018': 2017
    '2026': 2025
    '2027': 2026
  i_direction: -1
  i_increment: 10000
""")
    if vals["rate"] != 40:
        r = _pct(vals["rate"])
        blocks.append(f"""\
# top_tax dial ESTATE top rate -> {vals['rate']}% from 2027 (graduated ladder kept)
rates:
  value:
    '2014': [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.37, 0.39, 0.40]
    '2027': [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.37, 0.39, {r}]
""")
    return {"estate.yaml": blocks} if blocks else {}


def qbi_files(vals):
    return {"qbi.yaml": ["""\
# top_tax dial QBI: repeal Section 199A (QBI deduction) from 2027
rate:
  value:
    '2014': 0.0
    '2018': 0.2
    '2027': 0.0
min_value:
  value:
    '2014': 0
    '2026': 400
    '2027': 0
  i_measure:
    '1987': cpi
    '2017': chained_cpi
  i_base_year: 2025
  i_direction: 0
  i_increment: 5
"""]}


def taxmax_files(vals):
    return {"pr.yaml": ["""\
# top_tax dial TAXMAX: eliminate the OASDI taxable maximum from 2027.
# Earnings above the (unchanged, still-indexed) cap are taxed at the full
# 6.2% on both employee and employer sides; FICA and SECA read these same
# rate parameters, so self-employment earnings above the cap are covered too.
# Rates are unindexed scalars -- no i_ fields.
oasdi_ee_rates:
  value:
    '2014': [0.062, 0.0]
    '2027': [0.062, 0.062]
oasdi_er_rates:
  value:
    '2014': [0.062, 0.0]
    '2027': [0.062, 0.062]
"""]}


# --------------------------------------------------------------------------- #
# The eight dials, in canonical order. The four in the cluster — capital gains,
# wealth, gains at death, estate — are the ones whose interactions with each
# other were expected to be large.
#
# Per dial:
#   key/label/grp     : identity and how the interactive page groups it
#                       ('rate' for a rate change, 'struct' for a structural one)
#   kind              : 'yaml' — the dial writes reform law files
#   cluster           : in the four-dial cluster described above
#   wealth_behavior   : the run needs the wealth-concealment behavior module
#   interp            : how the surrogate interpolates between the dial's
#                       anchors — 'linear1d', 'bilinear', 'ladder',
#                       'ladder_x' (discrete positions with a dollar sub-axis),
#                       or 'binary'
#   params            : the dial's axes. Continuous axes carry off/min/max/
#                       anchors/ref, and scale='log' where the axis is dollars
#                       interpolated on a log scale.
#   files(vals)       : the law-file text for a run at these values
# --------------------------------------------------------------------------- #
LEVERS = [
    dict(key="ord", label="Top ordinary rate", grp="rate", kind="yaml",
         cluster=False, wealth_behavior=False, interp="linear1d", files=ord_files,
         params=[dict(key="rate", label="Top rate", unit="%", fmt="pct",
                      # min = current law (37%): the input can sit at zero
                      # effect, and the curve interpolates from current law up
                      # to 39.6 the same way the gains curve runs 20 to 25
                      off=37.0, min=37.0, max=50.0,
                      anchors=[39.6, 44.8, 50.0], ref=44.8)]),
    dict(key="cg", label="Capital gains & dividends top rate", grp="rate", kind="yaml",
         cluster=True, wealth_behavior=False, interp="linear1d", files=cg_files,
         params=[dict(key="rate", label="Top rate", unit="%", fmt="pct",
                      off=20.0, min=20.0, max=50.0,
                      anchors=[25.0, 30.0, 35.0, 40.0, 45.0, 50.0], ref=40.0)]),
    # The corporate rate is scored on-model, so its revenue curve needs several
    # anchors. The reference stays at 28% so pairs and triples remain
    # comparable with earlier vintages.
    dict(key="corp", label="Corporate rate", grp="rate", kind="yaml",
         cluster=False, wealth_behavior=False, interp="linear1d", files=corp_files,
         params=[dict(key="rate", label="Rate", unit="%", fmt="pct",
                      off=21.0, min=21.0, max=35.0,
                      anchors=[24.5, 28.0, 31.5, 35.0], ref=28.0)]),
    # The wealth anchors are denser than first planned, because the original
    # grid of rates 1 through 5 and thresholds $50M, $500M, $1B overpredicted by 5 to 11 percent at
    # thresholds inside the $50M-to-$500M segment (true revenue against
    # log(threshold) is convex) and underpredicted below a 1% rate (the Laffer
    # curve is concave in the rate). Thresholds $100M, $150M, $250M and a 0.5%
    # rate were added.
    dict(key="wealth", label="Annual wealth tax", grp="rate", kind="yaml",
         cluster=True, wealth_behavior=True, interp="bilinear", files=wealth_files,
         params=[dict(key="rate", label="Rate", unit="%", fmt="pct",
                      off=0.0, min=0.0, max=5.0,
                      anchors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0], ref=2.0),
                 dict(key="thr", label="Threshold", unit="$", fmt="usd", scale="log",
                      off=50e6, min=50e6, max=1000e6,
                      anchors=[50e6, 100e6, 150e6, 250e6, 500e6, 1000e6], ref=50e6)]),
    # The exclusion axis is a per-decedent dollar exclusion under carryover
    # basis or deemed realization, on top of the Section 121 home-sale exclusion. The
    # 'ladder_x' interpolation is a hybrid — the discrete treatment positions
    # are never scaled by a dial strength (each is a real run), while the
    # exclusion sub-axis interpolates within a position. The exclusion's
    # reference stays at zero so every pair, triple and stack measured before
    # the axis existed is still valid.
    dict(key="deemed", label="Gains at death", grp="struct", kind="yaml",
         cluster=True, wealth_behavior=False, interp="ladder_x", files=deemed_files,
         params=[dict(key="pos", label="Treatment", unit="", fmt="pos",
                      positions=["off", "carryover", "deemed"], off="off", ref="deemed"),
                 dict(key="exem", label="Exclusion", unit="$", fmt="usd",
                      off=0.0, min=0.0, max=5e6,
                      anchors=[0.0, 1e6, 5e6], ref=0.0)]),
    dict(key="estate", label="Estate tax", grp="struct", kind="yaml",
         cluster=True, wealth_behavior=False, interp="bilinear", files=estate_files,
         params=[dict(key="rate", label="Top rate", unit="%", fmt="pct",
                      off=40.0, min=40.0, max=60.0,
                      anchors=[40.0, 50.0, 60.0], ref=50.0),
                 dict(key="exem", label="Exemption", unit="$", fmt="usd",
                      off=float(ESTATE_CUR), min=5e6, max=float(ESTATE_CUR),
                      anchors=[5e6, float(ESTATE_M2), float(ESTATE_M1), float(ESTATE_CUR)],
                      ref=float(ESTATE_M2))]),
    dict(key="qbi", label="Repeal QBI (§199A)", grp="struct", kind="yaml",
         cluster=False, wealth_behavior=False, interp="binary", files=qbi_files,
         params=[dict(key="on", label="Repeal", unit="", fmt="bool",
                      positions=[0, 1], off=0, ref=1)]),
    dict(key="taxmax", label="Eliminate SS taxable max", grp="struct", kind="yaml",
         cluster=False, wealth_behavior=False, interp="binary", files=taxmax_files,
         params=[dict(key="on", label="Eliminate", unit="", fmt="bool",
                      positions=[0, 1], off=0, ref=1)]),
]
LEVER_KEYS = [lv["key"] for lv in LEVERS]
CLUSTER = [lv["key"] for lv in LEVERS if lv["cluster"]]
BY_KEY = {lv["key"]: lv for lv in LEVERS}

# Reference settings. Pairs, triples and the full stack are measured here.
REF = {
    "ord":    {"rate": 44.8},
    "cg":     {"rate": 40.0},
    "corp":   {"rate": 28.0},
    "wealth": {"rate": 2.0, "thr": 50e6},
    "deemed": {"pos": "deemed", "exem": 0.0},
    "estate": {"rate": 50.0, "exem": float(ESTATE_M2)},
    "qbi":    {"on": 1},
    "taxmax": {"on": 1},
}


def is_off(key, vals):
    """True when vals is the dial's current-law off state, so its curve is zero."""
    lv = BY_KEY[key]
    if lv["interp"] == "bilinear" and key == "wealth":
        return vals["rate"] == 0.0                       # a zero rate raises nothing
                                                         # at any threshold
    if lv["interp"] == "ladder_x":
        return vals["pos"] == "off"                      # the exclusion is moot
                                                         # under step-up basis
    return all(vals[p["key"]] == p["off"] for p in lv["params"])


def solo_cells(key):
    """Anchor cells for a dial's own revenue curve: [(vals, run_flag)].

    A run_flag of False means the cell is current law by construction, raises
    nothing, and is not run; its curve rows are zeros. The two-parameter dials
    return the complete anchor grid — every combination of the two axes — so
    the interactive page interpolates on a full lattice. The wealth dial's rate
    axis leaves out zero (the whole zero-rate edge is implicitly nothing), and
    the estate dial's rate axis keeps 40 percent, because lowering the
    exemption at the current rate is real revenue; only the corner at the
    current rate and the current exemption is nothing.
    """
    lv = BY_KEY[key]
    if lv["interp"] == "linear1d":
        return [({"rate": a}, True) for a in lv["params"][0]["anchors"]]
    if lv["interp"] == "bilinear":
        p1, p2 = lv["params"]
        return [({p1["key"]: a1, p2["key"]: a2}, not is_off(key, {p1["key"]: a1, p2["key"]: a2}))
                for a1 in p1["anchors"] for a2 in p2["anchors"]]
    if lv["interp"] == "ladder":
        return [({"pos": p}, True) for p in lv["params"][0]["positions"] if p != "off"]
    if lv["interp"] == "ladder_x":
        p_pos, p_x = lv["params"]
        return [({"pos": p, p_x["key"]: a}, True)
                for p in p_pos["positions"] if p != "off"
                for a in p_x["anchors"]]
    if lv["interp"] == "binary":
        return [({"on": 1}, True)]
    raise ValueError(lv["interp"])


# --------------------------------------------------------------------------- #
# Pair corners: the pairs expected to interact most (from the earlier factorial
# runs — capital gains with gains at death was worth $580 billion, capital
# gains with the ordinary rate, the wealth tax's spillovers, the corporate
# rate's markdown of equity values) re-run at one off-reference setting each.
# They test whether the interaction really does scale with the two dial
# strengths away from the reference point. They are held out of the fit.
# --------------------------------------------------------------------------- #
PAIR_CORNERS = [
    ("cg", {"rate": 50.0}, "deemed", {"pos": "deemed"}),
    ("cg", {"rate": 30.0}, "wealth", {"rate": 3.0, "thr": 500e6}),
    ("cg", {"rate": 50.0}, "estate", {"rate": 60.0, "exem": 5e6}),
    ("wealth", {"rate": 4.0, "thr": 50e6}, "estate", {"rate": 60.0, "exem": 5e6}),
    ("wealth", {"rate": 1.0, "thr": 1000e6}, "deemed", {"pos": "deemed"}),
    ("ord", {"rate": 50.0}, "cg", {"rate": 30.0}),
    ("ord", {"rate": 50.0}, "qbi", {"on": 1}),
    ("corp", {"rate": 35.0}, "cg", {"rate": 30.0}),
    # Exclusion corners: test that the within-position exclusion
    # ratio scales the gains-at-death pair terms away from a zero exclusion.
    ("cg", {"rate": 50.0}, "deemed", {"pos": "deemed", "exem": 5e6}),
    ("wealth", {"rate": 2.0, "thr": 50e6}, "deemed", {"pos": "carryover", "exem": 1e6}),
]

# The quiz: seeded holdout packages, strictly excluded from the fit. They are
# drawn in build_dial_runs.py and are the same every time under these seeds.
# Sizes 3 to 8 dials; continuous parameters sit off the anchors at presentable
# values; at least four packages have three or more cluster members and at
# least two put carryover basis inside a stack of three or more. The
# gains-at-death, QBI and taxable-maximum dials enter at their discrete
# positions.
QUIZ = dict(seed=20260709, n=16, size_min=3, size_max=8,
            min_cluster_heavy=4, min_carryover=2,
            # The four exclusion packages are drawn from their own
            # seeded stream so the original sixteen do not reshuffle. Each puts
            # the gains-at-death dial at an off-anchor exclusion inside a stack.
            n_exem=4, seed_exem=20260728)

# Patch scenarios, added after a holdout failure: the validator names the dial
# or pair that missed, targeted anchors and corners are added here, and
# build_dial_runs.py --patch N writes them to dials_patch{N}.csv, which runs
# into the same vintage before the surrogate is refitted. The main build
# appends every patch listed in LIVE_PATCHES to dials.csv.
#
# Each entry is dict(id=..., levers={key: vals, ...}, kind=...). A 'solo' row
# joins a dial's own curve. 'pairco', 'tripco' and 'pairexem' rows fit pair and
# triple terms at one gains-at-death position or exclusion anchor. Positions
# on the gains-at-death ladder are never reached by scaling a dial strength:
# every interaction that touches the ladder is measured by a real run at each
# position.
LIVE_PATCHES = ("4",)

# Patch 4: every gains-at-death pair measured at each nonzero exclusion anchor.
# A large exclusion shrinks the base the other dial interacts with faster than
# it shrinks the gains-at-death dial's own revenue, so scaling the pair term by
# the dial's revenue ratio missed by up to 30 percent (corner
# pc_cgr50_deemede5). The surrogate interpolates between these measured anchors
# instead.
PATCHES = {
    "4": [
        dict(id=f"prx_{a}_deemed_e{int(e / 1e6)}", kind="pairexem",
             levers={a: REF[a], "deemed": {"pos": "deemed", "exem": float(e)}})
        for a in LEVER_KEYS if a != "deemed"
        for e in (1e6, 5e6)
    ],
}
