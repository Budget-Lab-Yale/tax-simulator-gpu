#!/usr/bin/env python3
"""
Fit the top_tax dial surrogate from the dials batch and write atlas2_data.json.

Model (the interactive page evaluates the same formula for each quantity):

    V(x_1..x_k) = sum_i f_i(x_i)
                + sum_{i<j} I_ij * g_i(x_i) * g_j(x_j)
                + sum_{cluster triples} T_ijk * g_i * g_j * g_k

  f_i : per-lever solo curve. Stored as a COMPLETE knot grid per lever —
        1-D knot rows for continuous levers (off-knot row = zeros), row-major
        (axis1 outer x axis2 inner) grids for the 2-param levers, position
        rows for ladder/binary. Anchor rows are the run output VERBATIM
        (post 3-dp rounding); interpolation is piecewise-linear between
        knots (bilinear on grids; wealth threshold on a log($) axis).
  g_i : PER-DECADE dial-strength function: the lever's FIRST-ORDER solo
        decade-d total at x divided by its value at REF (g_d(off)=0,
        g_d(ref)=1), stored as an n_decade vector per knot-grid row.
        Interaction terms for a decade-d quantity element scale by that
        decade's g — every decade is estimated independently (etr, a 2027
        quantity, uses decade-1 g). Guard per decade: |first-order ref total|
        < 1e-6 -> g 0 for that decade.
  I_ij: pair interference measured once per pair at REF settings.
  T_ijk: triple interference for the flagged cluster {cg, wealth, deemed,
        estate} only, measured at REF.

Quantities (all deltas vs the same-run baseline world, $B except etr in pp).
DECADE-STRUCTURED (30-year runs, reported one decade at a time):
  ct/st  conventional/first-order decade totals (m=3: 2027-36, 2037-46,
         2047-56)
  cy/sy  conventional/first-order byyear        (m=30, span 2027-2056)
  ch/sh  conventional/first-order decade heads  (m=21, decade-major: d1
         HEADS_ORDER, d2 HEADS_ORDER, d3 HEADS_ORDER)
  etr    static-world ETR less baseline ETR (m=2 defs x groups x 6 comps,
         2027 only)
  etrc   conventional-world ETR less baseline ETR (same m). Each world's rate
         divides by that world's own pretax income.
  tax    first-order tax change in $B (same m and layout as etr, 2027 only)
  taxc   conventional tax change in $B (same m and layout as etr)
A rate change times baseline income is not the dollar change when the policy
changes pretax income, as realizations do on cash income. Dollar figures on
the page should use tax and taxc.
Every fitted object (f, g, I, T) is estimated independently per decade from
the same runs; nothing is extrapolated across decades. What remains shared
is the FORM (solo + I.g.g pairs + T.g.g.g triples, 4-way+ assumed zero) —
the per-decade quiz holdouts measure how that form holds up out-decade.

The three result legs map onto three of this model's five worlds: "static" is
`first_order`, "mechanical" is `mechanical`, "conventional" is
`conventional`. The quantity identifiers (st, sy, sh) are fixed by the JSON
schema the interactive page reads.

Runs without a distribution table
---------------------------------
A model run that writes no distribution.parquet still carries every revenue
quantity. When the baseline scenario has no such table, the fit covers the
revenue quantities only: the etr and etrc quantities and the baseline
effective-rate levels are left out of the JSON, and meta.etr_available is set
to false so the page and the validator know not to look for them. Pass
--without-etr to leave them out even when the tables are present.

Usage (pure file I/O — login-node safe; needs pandas and pyarrow for the
distribution table):
  python3 projects/top_tax/fit_surrogate.py <run_dir> [out_json] [--without-etr]

where <run_dir> is a model run directory: the timestamped folder holding
manifest.json and scenarios/.
"""

import csv
import datetime
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import levers as L
import readers as X

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_OUT = os.path.join(HERE, "atlas2_data.json")
LEGEND = os.path.join(HERE, "dials_legend.csv")

# The scenario whose baseline-world rows supply the levels every delta is
# measured against. Every scenario writes a baseline world, but taking them all
# from one scenario keeps the deltas comparable across scenarios by
# construction.
BASELINE_SCENARIO = "baseline"

# The result legs keep their R names here and in readers.py, which maps each to
# the model world it now comes from: static -> first_order, mechanical ->
# mechanical, conventional -> conventional.
STATIC_LEG = "static"
MECHANICAL_LEG = "mechanical"
CONVENTIONAL_LEG = "conventional"

DECADES = X.DECADES                    # [(2027,2036), (2037,2046), (2047,2056)]
SPAN = list(range(DECADES[0][0], DECADES[-1][1] + 1))   # 2027..2056
ETR_YEAR = 2027                        # the page shows the first year only
HEADS_ORDER = ["iit", "cg", "pay", "corp", "est", "wealth", "other"]
MONEY_QIDS = ["ct", "cy", "ch", "mt", "my", "mh", "st", "sy", "sh"]
# etr  = first-order-numerator (welfare) ETR deltas; etrc = conventional-
# numerator (realized) ETR deltas — same denominators/groups/comps, a
# numerator-only swap
ETR_QIDS = ("etr", "etrc")
# Tax change in $B by income definition, group and component, for the
# first-order and conventional worlds.
DIST_DOLLAR_QIDS = ("tax", "taxc")
# Quantities reported for the impact year only, not by decade.
IMPACT_YEAR_QIDS = ETR_QIDS + DIST_DOLLAR_QIDS
# Near-zero omission: money vectors are kept unless zero at shipped 3-dp
# precision (preserves fit exactness); ETR vectors get a real 0.005pp floor
# (they are 160-wide — omission is where the size saving lives).
PAIR_TOL = {"money": 0.0005, "etr": 0.005}


# --------------------------------------------------------------------------- #
# Knot grids: the client contract. Every lever is a full grid over per-param
# knot axes (off value included), row-major, with identically-zero rows for
# current-law cells. States in this file / fixtures / the page are
# {leverKey: {param: value}} for ON levers only.
# --------------------------------------------------------------------------- #
def lever_axes(key):
    """[(param_key, [knots], scale)] — off value prepended when not an anchor."""
    lv = L.BY_KEY[key]
    axes = []
    for p in lv["params"]:
        if "positions" in p:
            axes.append((p["key"], list(p["positions"]), "pos"))
        else:
            knots = list(p["anchors"])
            if p["off"] not in knots:
                knots = [p["off"]] + knots
            axes.append((p["key"], knots, p.get("scale", "linear")))
    return axes


def lever_grid_states(key):
    """Row-major [(vals, is_zero)] over the full knot grid."""
    axes = lever_axes(key)
    if len(axes) == 1:
        pk, knots, _ = axes[0]
        return [({pk: k}, _is_zero_state(key, {pk: k})) for k in knots]
    (p1, k1, _), (p2, k2, _) = axes
    return [({p1: a, p2: b}, _is_zero_state(key, {p1: a, p2: b}))
            for a in k1 for b in k2]


def _is_zero_state(key, vals):
    lv = L.BY_KEY[key]
    if lv["interp"] in ("ladder", "ladder_x", "binary"):
        return vals[lv["params"][0]["key"]] == lv["params"][0]["off"]
    if key == "wealth":
        return vals["rate"] == 0.0
    return L.is_off(key, vals)


def _param_off(key, pk):
    """A param's off value — the default when a state omits the param."""
    for p in L.BY_KEY[key]["params"]:
        if p["key"] == pk:
            return p["off"]
    raise KeyError((key, pk))


def state_key(state):
    return json.dumps(state, sort_keys=True)


# --------------------------------------------------------------------------- #
# Evaluator. The interactive page implements the same formula.
# Knot-verbatim rule: an input exactly on a knot returns the stored row with
# no interpolation arithmetic.
# --------------------------------------------------------------------------- #
def _locate(knots, x, scale):
    """(index, t) on a knot axis; x clamped to [knots0, knots-1]. A position
    axis is exact-match only."""
    if scale == "pos":
        return knots.index(x), 0.0
    if scale == "log":
        knots = [math.log(k) for k in knots]
        x = math.log(max(x, 1e-12))
    if x <= knots[0]:
        return 0, 0.0
    if x >= knots[-1]:
        return len(knots) - 1, 0.0
    for i in range(len(knots) - 1):
        if x == knots[i]:
            return i, 0.0
        if x < knots[i + 1]:
            return i, (x - knots[i]) / (knots[i + 1] - knots[i])
    return len(knots) - 1, 0.0


def eval_grid(key, rows, state_vals):
    """Interpolate a stored grid (list of rows, row = list) at state_vals.
    A param the state omits sits at its off value (exclusion 0 on states
    that predate the sub-axis)."""
    axes = lever_axes(key)
    if len(axes) == 1:
        pk, knots, scale = axes[0]
        x = state_vals.get(pk, _param_off(key, pk))
        if scale == "pos":
            i = knots.index(x)
            return list(rows[i])
        i, t = _locate(knots, x, scale)
        if t == 0.0:
            return list(rows[i])
        return [(1 - t) * a + t * b for a, b in zip(rows[i], rows[i + 1])]
    (p1, k1, s1), (p2, k2, s2) = axes
    i1, t1 = _locate(k1, state_vals.get(p1, _param_off(key, p1)), s1)
    i2, t2 = _locate(k2, state_vals.get(p2, _param_off(key, p2)), s2)
    n2 = len(k2)

    def row(a, b):
        return rows[a * n2 + b]
    if t1 == 0.0 and t2 == 0.0:
        return list(row(i1, i2))
    r00, r01 = row(i1, i2), row(i1, min(i2 + 1, n2 - 1))
    r10 = row(min(i1 + 1, len(k1) - 1), i2)
    r11 = row(min(i1 + 1, len(k1) - 1), min(i2 + 1, n2 - 1))
    return [(1 - t1) * ((1 - t2) * a + t2 * b) + t1 * ((1 - t2) * c + t2 * d)
            for a, b, c, d in zip(r00, r01, r10, r11)]


def dec_of(data, qid):
    """Element index -> decade index for quantity qid. Decade-structured money
    quantities are decade-major with m = n_dec * per; the effective-rate and
    dollar distribution quantities are 2027 only."""
    m = data["meta"]["surrogate"]["m"][qid]
    nd = len(data["meta"].get("decades") or [0])
    if qid in IMPACT_YEAR_QIDS or nd < 2:
        return [0] * m
    per = m // nd
    return [d for d in range(nd) for _ in range(per)]


def _ladder_ratio(sur, state):
    """Within-position exclusion ratio g_deemed(pos, exem) / g_deemed(pos, 0),
    per decade. Scales the byPos vectors, which were measured at exem = 0 —
    the exclusion moves the interaction the way it moves the solo, while
    positions themselves are never g-scaled. 1 everywhere at exem = 0."""
    vals = dict(state["deemed"])
    g_x = eval_grid("deemed", sur["g"]["deemed"], vals)
    vals["exem"] = 0.0
    g_0 = eval_grid("deemed", sur["g"]["deemed"], vals)
    return [a / b if abs(b) > 1e-9 else 0.0 for a, b in zip(g_x, g_0)]


def _exem_interp(by_exem, vec0, qid, exem):
    """Interpolate a pair vector linearly in the exclusion between measured
    anchors: exem 0 (the byPos vector) and the byExem anchors. Clamps above
    the top anchor. Anchors missing qid (dropped below tolerance) count as
    zero vectors."""
    knots = [(0.0, vec0)] + sorted(
        (float(e), v.get(qid)) for e, v in by_exem.items())
    m = next((len(v) for _, v in knots if v), None)
    if m is None:
        return None
    vecs = [v if v else [0.0] * m for _, v in knots]
    xs = [x for x, _ in knots]
    if exem >= xs[-1]:
        return vecs[-1]
    i = max(j for j in range(len(xs)) if xs[j] <= exem)
    t = (exem - xs[i]) / (xs[i + 1] - xs[i])
    return [(1 - t) * a + t * b for a, b in zip(vecs[i], vecs[i + 1])]


def _pair_term(sur, a, b, qid, state, gval):
    """(vector, per-decade weights) for pair a|b at qid, honoring byPos
    vectors: when the pair carries per-position vectors and one lever is the
    ladder (deemed), use the position-specific vector scaled by the OTHER
    lever's g. At a nonzero exclusion, a measured byExem vector interpolated
    in the exclusion replaces the within-position solo-ratio scaling; the
    ratio remains the fallback for positions with no measured anchors.
    Weights are n_dec-vectors — element x of the quantity uses the weight of
    its own decade."""
    entry = sur["pairs"].get(f"{a}|{b}")
    if not entry:
        return None, None
    by_pos = entry.get("byPos")
    ladder = "deemed" if "deemed" in (a, b) else None
    if by_pos and ladder:
        pos = state[ladder]["pos"]
        if pos in by_pos:
            other = b if a == ladder else a
            exem = float(state[ladder].get("exem", 0.0) or 0.0)
            by_exem = (entry.get("byExem") or {}).get(pos)
            if by_exem and exem > 0.0:
                vec = _exem_interp(by_exem, by_pos[pos].get(qid), qid, exem)
                return vec, list(gval[other])
            vec = by_pos[pos].get(qid)
            ratio = _ladder_ratio(sur, state)
            return vec, [x * r for x, r in zip(gval[other], ratio)]
    vec = entry.get(qid)
    return vec, [x * y for x, y in zip(gval[a], gval[b])]


def _triple_term(entry, keys, qid, state, gval, sur=None):
    """(vector, per-decade weights) for a triple, honoring byPos: with
    per-position vectors and the ladder (deemed) among the members, use the
    position vector scaled by the product of the OTHER two levers' g times
    the within-position exclusion ratio."""
    by_pos = entry.get("byPos")
    nd = len(next(iter(gval.values())))
    if by_pos and "deemed" in keys:
        pos = state["deemed"]["pos"]
        if pos in by_pos:
            w = _ladder_ratio(sur, state) if sur else [1.0] * nd
            for k in keys:
                if k != "deemed":
                    w = [a * b for a, b in zip(w, gval[k])]
            return by_pos[pos].get(qid), w
    vec = entry.get(qid)
    w = [1.0] * nd
    for k in keys:
        w = [a * b for a, b in zip(w, gval[k])]
    return vec, w


def eval_state(data, qid, state):
    """Predict quantity qid (vector) at a state dict. Mirrors JS evalQ."""
    sur = data["surrogate"]
    m = data["meta"]["surrogate"]["m"][qid]
    deci = dec_of(data, qid)
    tot = [0.0] * m
    on = [k for k in L.LEVER_KEYS if k in state]
    gval = {}
    for k in on:
        f = sur["solo"][k].get(qid)
        if f:
            for i, v in enumerate(eval_grid(k, f, state[k])):
                tot[i] += v
        gval[k] = eval_grid(k, sur["g"][k], state[k])   # per-decade vector
    for i in range(len(on)):
        for j in range(i + 1, len(on)):
            vec, w = _pair_term(sur, on[i], on[j], qid, state, gval)
            if vec:
                for x in range(m):
                    tot[x] += vec[x] * w[deci[x]]
    for trip, qs in sur["triples"].items():
        keys = trip.split("|")
        if all(k in gval for k in keys):
            vec, w = _triple_term(qs, keys, qid, state, gval, sur)
            if vec:
                for x in range(m):
                    tot[x] += vec[x] * w[deci[x]]
    return tot


def shapley(data, state, d=0):
    """Closed-form Shapley split of predicted conventional decade-d total
    across ON levers."""
    sur = data["surrogate"]
    on = [k for k in L.LEVER_KEYS if k in state]
    gval, phi = {}, {}
    for k in on:
        gval[k] = eval_grid(k, sur["g"][k], state[k])
        f = sur["solo"][k].get("ct")
        phi[k] = eval_grid(k, f, state[k])[d] if f else 0.0
    for i in range(len(on)):
        for j in range(i + 1, len(on)):
            vec, w = _pair_term(sur, on[i], on[j], "ct", state, gval)
            if vec:
                half = 0.5 * vec[d] * w[d]
                phi[on[i]] += half
                phi[on[j]] += half
    for trip, qs in sur["triples"].items():
        ks = trip.split("|")
        if all(k in gval and k in phi for k in ks):
            vec, w = _triple_term(qs, ks, "ct", state, gval, sur)
            if vec:
                third = vec[d] * w[d] / 3.0
                for k in ks:
                    phi[k] += third
    return phi


# --------------------------------------------------------------------------- #
# Reading run output (readers from readers.py)
# --------------------------------------------------------------------------- #
def read_legend():
    """Main legend + any dials_patch{N}_legend.csv (same-run patch scenarios).
    Later legends win on duplicate IDs (patch 2 re-lists patch 1's rows)."""
    import glob
    by_id = {}
    paths = [LEGEND] + sorted(glob.glob(LEGEND.replace("_legend.csv", "_patch*_legend.csv")))
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                by_id[r["ID"]] = (r["ID"], r["kind"], json.loads(r["levers_json"]))
    return list(by_id.values())


def read_scenario(root, scen_id, base_heads, base_cg, etr_base_flat, groups,
                  want_etr=True):
    """All quantity vectors (deltas vs the baseline world) for one scenario."""
    static, _ = X.leg_deltas_windows(root, scen_id, base_heads, base_cg,
                                     STATIC_LEG, DECADES)
    conv, _ = X.leg_deltas_windows(root, scen_id, base_heads, base_cg,
                                   CONVENTIONAL_LEG, DECADES)
    # The mechanical rung is written only where a transmission channel is live.
    # Where it is absent the rung below stands in, which is what the reporting
    # layer does, and the mechanical bar then coincides with the static one.
    if X.has_world(root, scen_id, X.LEG_TO_WORLD[MECHANICAL_LEG]):
        mech, _ = X.leg_deltas_windows(root, scen_id, base_heads, base_cg,
                                       MECHANICAL_LEG, DECADES)
    else:
        mech = static

    def flat_heads(res):
        return [res["heads"][d][h] for d in range(len(DECADES)) for h in HEADS_ORDER]
    out = {
        "ct": list(conv["totals"]), "cy": conv["byyear"],
        "ch": flat_heads(conv),
        "mt": list(mech["totals"]), "my": mech["byyear"],
        "mh": flat_heads(mech),
        "st": list(static["totals"]), "sy": static["byyear"],
        "sh": flat_heads(static),
    }
    if want_etr:
        for qid, leg in (("etr", STATIC_LEG), ("etrc", CONVENTIONAL_LEG)):
            rows = X.etr_rows(root, scen_id, reform_leg=leg, years=[ETR_YEAR])
            if not rows:
                # etr_rows() also returns [] for a missing file. Filling that
                # with zeros would make the fitted rate changes equal minus
                # the baseline, so the fit stops instead.
                sys.exit(f"{scen_id}: no {leg}-leg effective-rate rows for "
                         f"{ETR_YEAR} — refusing to fit")
            pack = X.etr_pack(rows, "reform")
            flat = flatten_etr(pack, groups)
            out[qid] = [round(a - b, 3) for a, b in zip(flat, etr_base_flat)]
            dollar_qid = DIST_DOLLAR_QIDS[ETR_QIDS.index(qid)]
            out[dollar_qid] = flatten_etr(tax_change_pack(rows), groups)
    return out


def tax_change_pack(rows):
    """{income definition: {year: {group: [the six components]}}}: the
    reform world's tax less the baseline world's, in $B. Each component's tax
    is its rate times the pretax income that rate was divided by."""
    out = {}
    for r in rows:
        vals = [round(float(r[f"etr_{c}_reform"]) * float(r["income_reform"])
                      - float(r[f"etr_{c}_baseline"]) * float(r["income_baseline"]), 3)
                for c in X.ETR_COMPS]
        out.setdefault(r["income_definition"], {}).setdefault(
            str(r["year"]), {})[r["group"]] = vals
    return out


def income_levels_of(rows):
    """{income definition: {group: baseline income $B}} for the display year.
    The denominators do not vary by result leg, so any run's rows serve."""
    out = {}
    for r in rows:
        out.setdefault(r["income_definition"], {})[r["group"]] = round(
            float(r["income_baseline"]), 3)
    return out


def flatten_etr(pack, groups):
    """{def: {year: {group: [comps]}}} -> flat def x group x comp vector."""
    out = []
    for d in X.ETR_INCOME_DEFS:
        by_group = pack.get(d, {}).get(str(ETR_YEAR), {})
        for g in groups:
            row = by_group.get(g)
            out.extend(row if row else [0.0] * len(X.ETR_COMPS))
    return out


def etr_groups_of(root, scen_id):
    """(group order, effective-rate rows) for the display year."""
    rows = X.etr_rows(root, scen_id, years=[ETR_YEAR])
    if not rows:
        sys.exit(f"{scen_id}: no effective-rate rows for {ETR_YEAR} — "
                 "refusing to fit")
    seen = []
    for r in rows:
        if r["group"] not in seen:
            seen.append(r["group"])
    return seen, rows


# One sentence stamped into the JSON when the run carries no distribution
# table, so a reader of the file knows why those quantities are missing.
NO_ETR_NOTE = ("This model run wrote no distribution table, so the "
               "atlas carries revenue quantities only.")


# --------------------------------------------------------------------------- #
def fit(root, out_path, want_etr=None):
    """Fit the surrogate and write the atlas JSON.

    want_etr True or False forces the effective-rate quantities in or out;
    None reads the baseline scenario and decides from what the run wrote.
    """
    if want_etr is None:
        want_etr = X.has_distribution_table(root, BASELINE_SCENARIO)
    qids = list(MONEY_QIDS) + list(IMPACT_YEAR_QIDS) if want_etr else list(MONEY_QIDS)
    legend = read_legend()
    by_kind = {}
    for scen_id, kind, state in legend:
        by_kind.setdefault(kind, []).append((scen_id, state))

    # Baseline levels: the baseline world of the run's baseline scenario. Every
    # world's deltas are measured against it, exactly as R measured every leg
    # against the one baseline run.
    if not os.path.isdir(os.path.join(root, "scenarios", BASELINE_SCENARIO)):
        sys.exit(f"{root}: no scenarios/{BASELINE_SCENARIO} folder — the dials "
                 "runscript must carry a baseline row")
    base_heads, base_cg = X.baseline_levels(root, BASELINE_SCENARIO)

    # Baseline effective rates and the group order, from the first solo run.
    if want_etr:
        first_id = by_kind["solo"][0][0]
        groups, rows0 = etr_groups_of(root, first_id)
        etr_base_pack = X.etr_pack(rows0, "baseline")
        etr_base_flat = flatten_etr(etr_base_pack, groups)
        # Baseline income LEVELS ($B) by definition and group — baseline-fixed,
        # so any run's rows serve. The dist card multiplies surrogate ETR deltas
        # by these to get dollar tax deltas; bar heights use them as the
        # resource denominator.
        income_levels = income_levels_of(rows0)
    else:
        groups, etr_base_pack, etr_base_flat, income_levels = [], {}, [], {}

    def read(scen_id):
        return read_scenario(root, scen_id, base_heads, base_cg, etr_base_flat,
                             groups, want_etr=want_etr)

    runs = {}          # state_key -> quantities (fitted scenarios only)
    id_of = {}
    for kind in ("solo", "pair", "triple", "pairco", "tripco", "pairexem"):
        for scen_id, state in by_kind.get(kind, []):
            runs[state_key(state)] = read(scen_id)
            id_of[state_key(state)] = scen_id

    m_of = {"ct": len(DECADES), "cy": len(SPAN), "ch": len(DECADES) * len(HEADS_ORDER),
            "mt": len(DECADES), "my": len(SPAN), "mh": len(DECADES) * len(HEADS_ORDER),
            "st": len(DECADES), "sy": len(SPAN), "sh": len(DECADES) * len(HEADS_ORDER)}
    if want_etr:
        m_etr = len(X.ETR_INCOME_DEFS) * len(groups) * len(X.ETR_COMPS)
        for qid in IMPACT_YEAR_QIDS:
            m_of[qid] = m_etr

    # ---- solo grids + g ---------------------------------------------------- #
    solo, gfun = {}, {}
    for key in L.LEVER_KEYS:
        cells = lever_grid_states(key)
        solo[key] = {qid: [] for qid in qids}
        st_vals = []
        for vals, is_zero in cells:
            if is_zero:
                q = {qid: [0.0] * m_of[qid] for qid in qids}
            else:
                sk = state_key({key: vals})
                if sk not in runs:
                    sys.exit(f"missing solo run for {key} at {vals}")
                q = runs[sk]
            for qid in qids:
                solo[key][qid].append([round(v, 3) for v in q[qid]])
            st_vals.append(list(q["st"]))   # per-decade first-order totals -> g
        ref_state = {key: L.REF[key]}
        ref_i = [i for i, (vals, _) in enumerate(cells)
                 if state_key({key: vals}) == state_key(ref_state)]
        assert len(ref_i) == 1, f"{key}: REF not on the knot grid"
        st_ref = st_vals[ref_i[0]]
        gfun[key] = [[0.0 if abs(st_ref[d]) < 1e-6            # g guard per decade
                      else round(v[d] / st_ref[d], 6)
                      for d in range(len(DECADES))]
                     for v in st_vals]

    data_stub = {"surrogate": {"solo": solo, "g": gfun, "pairs": {}, "triples": {}},
                 "meta": {"surrogate": {"m": m_of}}}

    def f_at_ref(key, qid):
        return eval_grid(key, solo[key][qid], L.REF[key])

    # ---- pairs ------------------------------------------------------------- #
    pairs = {}
    import itertools
    for a, b in itertools.combinations(L.LEVER_KEYS, 2):
        sk = state_key({a: L.REF[a], b: L.REF[b]})
        if sk not in runs:
            sys.exit(f"missing pair run {a}|{b}")
        entry = {}
        for qid in qids:
            fa, fb = f_at_ref(a, qid), f_at_ref(b, qid)
            vec = [round(v - x - y, 3) for v, x, y in zip(runs[sk][qid], fa, fb)]
            tol = PAIR_TOL["etr" if qid in ETR_QIDS else "money"]
            if any(abs(v) > tol for v in vec):
                entry[qid] = vec
        if entry:
            pairs[f"{a}|{b}"] = entry

    # Per-position pair vectors (discrete-lever escape hatch): pairco runs
    # measure the deemed pair AT a specific ladder position (carryover), so
    # the evaluator can bypass the g_deemed scaling for that pair. The
    # 'deemed' position vector is materialized from the ref pair (g == 1
    # there), so byPos covers every ON position and the g-scaled fallback
    # never fires for these pairs.
    for scen_id, state in by_kind.get("pairco", []):
        others = [k for k in state if k != "deemed"]
        assert len(others) == 1, f"{scen_id}: pairco must be lever+deemed"
        other = others[0]
        assert state[other] == L.REF[other], f"{scen_id}: pairco other lever off-ref"
        pos = state["deemed"]["pos"]
        a, b = sorted([other, "deemed"], key=L.LEVER_KEYS.index)
        pkey = f"{a}|{b}"
        entry = pairs.setdefault(pkey, {})
        by_pos = entry.setdefault("byPos", {})
        vec_pos, vec_ref = {}, {}
        for qid in qids:
            fo = f_at_ref(other, qid)
            fd = eval_grid("deemed", solo["deemed"][qid], state["deemed"])
            sk = state_key(state)
            if sk not in runs:
                sys.exit(f"missing pairco run {scen_id}")
            vec_pos[qid] = [round(v - x - y, 3)
                            for v, x, y in zip(runs[sk][qid], fo, fd)]
            ref_i = entry.get(qid)
            vec_ref[qid] = ref_i if ref_i else [0.0] * m_of[qid]
        by_pos[pos] = vec_pos
        by_pos["deemed"] = vec_ref

    # Exclusion-anchored pair vectors (patch 4): pairexem runs measure a
    # deemed pair at a nonzero exclusion anchor, so the evaluator can
    # interpolate between measured anchors instead of scaling the exem = 0
    # vector by the solo ratio (which understated cg x deemed by 30% at the
    # $5M exclusion corner). Keyed by position, then exclusion anchor.
    for scen_id, state in by_kind.get("pairexem", []):
        others = [k for k in state if k != "deemed"]
        assert len(others) == 1, f"{scen_id}: pairexem must be lever+deemed"
        other = others[0]
        assert state[other] == L.REF[other], f"{scen_id}: pairexem other lever off-ref"
        pos = state["deemed"]["pos"]
        exem = float(state["deemed"]["exem"])
        a, b = sorted([other, "deemed"], key=L.LEVER_KEYS.index)
        entry = pairs.setdefault(f"{a}|{b}", {})
        by_exem = entry.setdefault("byExem", {}).setdefault(pos, {})
        sk = state_key(state)
        if sk not in runs:
            sys.exit(f"missing pairexem run {scen_id}")
        vec_e = {}
        for qid in qids:
            fo = f_at_ref(other, qid)
            fd = eval_grid("deemed", solo["deemed"][qid], state["deemed"])
            vec_e[qid] = [round(v - x - y, 3)
                          for v, x, y in zip(runs[sk][qid], fo, fd)]
        by_exem[str(int(exem))] = vec_e
    data_stub["surrogate"]["pairs"] = pairs

    # ---- triples (every kind='triple' run in the legend; deemed at its ref
    # position — carryover comes in via the tripco byPos block below) -------- #
    triples = {}
    triple_trios = []
    for scen_id, state in by_kind.get("triple", []):
        trio = tuple(sorted(state, key=L.LEVER_KEYS.index))
        for k in trio:
            assert state[k] == L.REF[k], f"{scen_id}: triple lever {k} off-ref"
        triple_trios.append(trio)
    for a, b, c in triple_trios:
        sk = state_key({a: L.REF[a], b: L.REF[b], c: L.REF[c]})
        if sk not in runs:
            sys.exit(f"missing triple run {a}|{b}|{c}")
        entry = {}
        for qid in qids:
            base = [0.0] * m_of[qid]
            for k in (a, b, c):
                for i, v in enumerate(f_at_ref(k, qid)):
                    base[i] += v
            for x, y in ((a, b), (a, c), (b, c)):
                vec = pairs.get(f"{x}|{y}", {}).get(qid)
                if vec:
                    for i, v in enumerate(vec):
                        base[i] += v            # g == 1 at REF
            vec = [round(v - w, 3) for v, w in zip(runs[sk][qid], base)]
            tol = PAIR_TOL["etr" if qid in ETR_QIDS else "money"]
            if any(abs(v) > tol for v in vec):
                entry[qid] = vec
        if entry:
            triples[f"{a}|{b}|{c}"] = entry

    # Per-position triple vectors: tripco runs re-measure a cluster triple
    # with deemed at a specific ladder position (discrete positions are never
    # g-scaled). Baseline for the residual = solo terms at this state + the
    # pair terms as the EVALUATOR computes them there (byPos pair vectors for
    # the deemed pairs, ref I for the continuous pair) — so prediction at the
    # tripco state is exact by construction.
    for scen_id, state in by_kind.get("tripco", []):
        others = sorted([k for k in state if k != "deemed"], key=L.LEVER_KEYS.index)
        assert len(others) == 2, f"{scen_id}: tripco must be 2 levers + deemed"
        for k in others:
            assert state[k] == L.REF[k], f"{scen_id}: tripco lever {k} off-ref"
        pos = state["deemed"]["pos"]
        trip_key = "|".join(sorted(others + ["deemed"], key=L.LEVER_KEYS.index))
        entry = triples.setdefault(trip_key, {})
        by_pos = entry.setdefault("byPos", {})
        sk = state_key(state)
        if sk not in runs:
            sys.exit(f"missing tripco run {scen_id}")
        vec_pos, vec_ref = {}, {}
        for qid in qids:
            base = [0.0] * m_of[qid]
            for k in others:
                for i, v in enumerate(f_at_ref(k, qid)):
                    base[i] += v
            for i, v in enumerate(eval_grid("deemed", solo["deemed"][qid],
                                            state["deemed"])):
                base[i] += v
            o1, o2 = others
            vec = pairs.get(f"{o1}|{o2}", {}).get(qid)
            if vec:
                for i, v in enumerate(vec):
                    base[i] += v                      # g_o1 = g_o2 = 1 at REF
            for other in others:
                a2, b2 = sorted([other, "deemed"], key=L.LEVER_KEYS.index)
                pe = pairs.get(f"{a2}|{b2}", {})
                pvec = pe.get("byPos", {}).get(pos, {}).get(qid)
                if pvec:                              # scaled by g_other = 1
                    for i, v in enumerate(pvec):
                        base[i] += v
            vec_pos[qid] = [round(v - w, 3) for v, w in zip(runs[sk][qid], base)]
            ref_t = entry.get(qid)
            vec_ref[qid] = ref_t if ref_t else [0.0] * m_of[qid]
        by_pos[pos] = vec_pos
        by_pos["deemed"] = vec_ref
    data_stub["surrogate"]["triples"] = triples

    # ---- meta --------------------------------------------------------------- #
    gdp_decades = X.gdp_fy_decades(DECADES)
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""

    lever_meta = []
    for lv in L.LEVERS:
        params = []
        for p, (pk, knots, scale) in zip(lv["params"], lever_axes(lv["key"])):
            pm = dict(key=p["key"], label=p["label"], unit=p["unit"], fmt=p["fmt"],
                      off=p["off"], ref=p["ref"], knots=knots, scale=scale)
            if "positions" in p:
                pm["positions"] = p["positions"]
            else:
                pm["min"], pm["max"] = p["min"], p["max"]
                pm["anchors"] = p["anchors"]
            params.append(pm)
        kind = {"linear1d": "continuous", "bilinear": "continuous",
                "ladder": "discrete", "ladder_x": "discrete",
                "binary": "binary"}[lv["interp"]]
        lever_meta.append(dict(key=lv["key"], label=lv["label"], grp=lv["grp"],
                               kind=kind, interp=lv["interp"], cluster=lv["cluster"],
                               params=params))

    n_scen = sum(len(v) for v in by_kind.values())
    meta = dict(
        schema=3, window=[SPAN[0], SPAN[-1]],
        decades=[list(w) for w in DECADES], gdp_fy_decades=gdp_decades,
        dist_years=[ETR_YEAR], etr_slice=X.ETR_FILTER,
        etr_income_defs=X.ETR_INCOME_DEFS, etr_comps=X.ETR_COMPS, etr_groups=groups,
        levers=lever_meta,
        surrogate=dict(quantities=qids, m=m_of, heads_order=HEADS_ORDER,
                       ref={k: L.REF[k] for k in L.LEVER_KEYS},
                       cluster=L.CLUSTER,
                       validation=None, checks=None),
        provenance=dict(
            vintage=root, git_sha=sha,
            date=datetime.date.today().isoformat(),
            n_scenarios=n_scen,
            corp_note=(
                "The corporate rate is scored inside the model against the CBO "
                "corporate receipts baseline (years/corporate/scorer.py), "
                "and the dial runs piecewise-linear through the solo anchors "
                "like every other rate dial. The effective-rate chart shows "
                "the corporate income tax spread over households top-down "
                "(above-normal return, normal return and labor, with the "
                "foreign-owned share dropped), and the estate tax and the tax "
                "on gains deemed realized at death in separate slots."),
        ),
        units="$B; ETRs in percent (deltas in pp)",
        placeholder=None,
        etr_available=want_etr,
        etr_note=None if want_etr else NO_ETR_NOTE,
    )

    # etr_base: baseline LEVELS by def/group (2027), same comp order. Left out
    # along with the income levels when the run has no distribution table.
    payload = dict(meta=meta, surrogate=data_stub["surrogate"])
    if want_etr:
        payload["etr_base"] = etr_base_pack
        payload["income_levels"] = income_levels
    with open(out_path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e3:.0f} KB)")
    print(f"  solos {sum(len(v['ct']) for v in solo.values())} grid rows | "
          f"pairs kept {len(pairs)}/28 | "
          f"triples kept {len(triples)}/{len(triple_trios)} measured")
    return payload


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--without-etr"]
    want = False if "--without-etr" in sys.argv[1:] else None
    if not args:
        sys.exit("usage: fit_surrogate.py <run_dir> [out_json] [--without-etr]")
    root = args[0].rstrip("/")
    out = args[1] if len(args) > 1 else DEFAULT_OUT
    fit(root, out, want_etr=want)
