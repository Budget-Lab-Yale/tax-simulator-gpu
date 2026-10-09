#!/usr/bin/env python3
"""
Build the top-tax dial batch: one reform scenario per dial setting, plus the
runscript that runs them all.

Writes, under the repository root:
  - config/scenarios/tax_law/alternatives/top_tax/dials/{id}/  (one folder of
    reform law files per scenario)
  - config/runscripts/top_tax/dials.csv                        (baseline + 211
    scenarios, plus the rows of every patch in levers.LIVE_PATCHES)
  - projects/top_tax/dials_legend.csv                          (ID, kind, the
    dial settings as JSON, and a human label)

Scenario kinds: solo (each dial's own anchor curve), pair (all 28 two-dial
combinations at the reference settings), pairco (the seven pairs with gains at
death re-measured at carryover basis), triple (all 56 three-dial combinations at
the reference settings), tripco (the 21 gains-at-death triples at carryover),
corner (ten off-reference pair settings), quiz (20 seeded holdout packages), and
stack (all eight dials at once). The corners, quiz and stack are validation
only; the surrogate is fitted on everything else. Positions on the
gains-at-death ladder are never reached by scaling — each is a real run.

Patches: after a holdout failure, add targeted runs to levers.PATCHES and list
the patch in levers.LIVE_PATCHES. The main build appends every live patch's
rows to dials.csv. `--patch N` writes one patch on its own to
config/runscripts/top_tax/dials_patch{N}.csv, holding only the new rows, to run
into the same vintage before the patch is made live.

Pure file writing — no model runs, so this is safe on a login node.

The main build deletes the dials/ tree and rebuilds it, including the live
patches. To confirm an edit changed nothing, run

    python3 projects/top_tax/build_dial_runs.py
    git status --short config/

and check that the status is empty.
"""

import csv
import itertools
import json
import os
import random
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levers as L

REPO = L.REPO
TAXLAW_ROOT = os.path.join(REPO, "config", "scenarios", "tax_law", "alternatives", "top_tax", "dials")
RUNSCRIPT_DIR = os.path.join(REPO, "config", "runscripts", "top_tax")
RUNSCRIPT = os.path.join(RUNSCRIPT_DIR, "dials.csv")
LEGEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dials_legend.csv")
TAXLAW_REL = "top_tax/dials"

# The model's runscript columns (src/taxmodel/config/runscript.py). The model
# refuses any other column.
HEADER = ["ID", "tax_law", "economy", "behavior", "years"]


# --------------------------------------------------------------------------- #
# Scenario names and labels
# --------------------------------------------------------------------------- #
def num_slug(x):
    """39.6 -> '39p6'; 8.46 -> '8p46'; 50.0 -> '50'."""
    s = f"{float(x):g}"
    return s.replace(".", "p").replace("-", "m")


def vals_slug(key, vals, sep="_"):
    if key == "wealth":
        return f"r{num_slug(vals['rate'])}{sep}t{int(vals['thr'] / 1e6)}"
    if key == "estate":
        e = vals["exem"]
        es = "cur" if e == L.ESTATE_CUR else num_slug(e / 1e6)
        return f"r{num_slug(vals['rate'])}{sep}e{es}"
    if key == "deemed":
        exem = float(vals.get("exem", 0.0))
        return vals["pos"] + (f"{sep}e{num_slug(exem / 1e6)}" if exem else "")
    if key in ("qbi", "taxmax"):
        return "on"
    return f"r{num_slug(vals['rate'])}"


def compact_slug(key, vals):
    """The shorter form used in corner names: cg50, wealth3t500, estate60e5,
    deemed, deemede5, qbi."""
    if key == "deemed":
        base = "deemed" if vals["pos"] == "deemed" else "deemedco"
        exem = float(vals.get("exem", 0.0))
        return base + (f"e{num_slug(exem / 1e6)}" if exem else "")
    if key in ("qbi", "taxmax"):
        return key
    return key + vals_slug(key, vals, sep="")


def vals_label(key, vals):
    if key == "ord":
        return f"ord {vals['rate']:g}%"
    if key == "cg":
        return f"cg {vals['rate']:g}%"
    if key == "corp":
        return f"corp {vals['rate']:g}%"
    if key == "wealth":
        return f"wealth {vals['rate']:g}% > ${vals['thr'] / 1e6:g}M"
    if key == "estate":
        e = vals["exem"]
        es = "current" if e == L.ESTATE_CUR else f"${e / 1e6:g}M"
        return f"estate {vals['rate']:g}% / {es}"
    if key == "deemed":
        exem = float(vals.get("exem", 0.0))
        lbl = f"deemed:{vals['pos']}"
        return lbl + (f" excl ${exem / 1e6:g}M" if exem else "")
    return key


def scen_label(scenario):
    return " + ".join(vals_label(k, scenario[k]) for k in L.LEVER_KEYS if k in scenario)


# --------------------------------------------------------------------------- #
# Writing a scenario. A scenario is {dial key: parameter values}.
# --------------------------------------------------------------------------- #
def write_scenario_dir(scen_id, scenario):
    """Write the merged law files for one scenario; return its tax_law path."""
    cdir = os.path.join(TAXLAW_ROOT, scen_id)
    os.makedirs(cdir, exist_ok=True)
    per_file = {}
    for key in L.LEVER_KEYS:
        if key not in scenario:
            continue
        lv = L.BY_KEY[key]
        if lv["files"] is None:            # a dial that writes no law file
            continue
        for fname, blocks in lv["files"](scenario[key]).items():
            per_file.setdefault(fname, []).extend(blocks)

    # The lift of the ordinary-rate ceiling rides along with the capital-gains
    # dial and is appended last (see levers.NO_ORD_CAP_BLOCK for why it sits
    # where it sits).
    if "cg" in scenario:
        per_file.setdefault("pref.yaml", []).append(L.NO_ORD_CAP_BLOCK)

    for fname, blocks in per_file.items():
        parts = ["---\n"]
        for blk in blocks:
            parts.append(blk if blk.endswith("\n") else blk + "\n")
            parts.append("\n")
        with open(os.path.join(cdir, fname), "w") as fh:
            fh.write("".join(parts).rstrip("\n") + "\n")
    return f"{TAXLAW_REL}/{scen_id}"


def runscript_row(scen_id, scenario):
    corp_on = "corp" in scenario
    wealth_on = any(L.BY_KEY[k]["wealth_behavior"] for k in scenario)
    return {
        "ID": scen_id,
        "tax_law": write_scenario_dir(scen_id, scenario),
        "behavior": L.BEHAVIOR_WEALTH if wealth_on else L.BEHAVIOR_BASE,
        "years": L.YEARS,
        "economy": L.CORP_ON_ECONOMY if corp_on else L.CORP_OFF_ECONOMY,
    }


# --------------------------------------------------------------------------- #
# The quiz: seeded holdout packages, the same every time, never fitted on.
# --------------------------------------------------------------------------- #
def _presentable_rate(rng, lo, hi, anchors, step=0.5):
    """Draw a rate in [lo, hi] rounded to a presentable step, never exactly on
    one of the dial's anchors."""
    for _ in range(100):
        x = round(round(rng.uniform(lo, hi) / step) * step, 4)
        x = min(max(x, lo), hi)
        if all(abs(x - a) > 1e-9 for a in anchors):
            return x
    raise RuntimeError("could not draw off-anchor value")


def _quiz_vals(rng, key, force_pos=None):
    lv = L.BY_KEY[key]
    if key == "ord":
        return {"rate": _presentable_rate(rng, 40.0, 50.0, [39.6, 44.8, 50.0])}
    if key == "cg":
        return {"rate": _presentable_rate(rng, 22.0, 50.0, lv["params"][0]["anchors"])}
    if key == "corp":
        return {"rate": _presentable_rate(rng, 22.0, 35.0, lv["params"][0]["anchors"])}
    if key == "wealth":
        thr = rng.choice([75e6, 100e6, 150e6, 200e6, 300e6, 700e6, 800e6])
        return {"rate": _presentable_rate(rng, 0.5, 5.0, [1, 2, 3, 4, 5], step=0.25),
                "thr": thr}
    if key == "estate":
        exem = _presentable_rate(rng, 5.25, 15.0,
                                 [5.0, L.ESTATE_M2 / 1e6, L.ESTATE_M1 / 1e6,
                                  L.ESTATE_CUR / 1e6], step=0.25) * 1e6
        return {"rate": _presentable_rate(rng, 41.0, 59.0, [40.0, 50.0, 60.0], step=1.0),
                "exem": exem}
    if key == "deemed":
        pos = force_pos or ("carryover" if rng.random() < 0.3 else "deemed")
        return {"pos": pos, "exem": 0.0}
    return {"on": 1}


def make_quiz():
    cfg = L.QUIZ
    rng = random.Random(cfg["seed"])
    non_cluster = [k for k in L.LEVER_KEYS if k not in L.CLUSTER]
    packages = []
    for i in range(cfg["n"]):
        size = rng.randint(cfg["size_min"], cfg["size_max"])
        force_pos = None
        if i < cfg["min_cluster_heavy"]:
            # at least three of the four cluster dials
            n_cl = rng.randint(3, len(L.CLUSTER))
            keys = rng.sample(L.CLUSTER, n_cl)
            keys += rng.sample(non_cluster, max(0, min(size, 8) - n_cl))
        elif i < cfg["min_cluster_heavy"] + cfg["min_carryover"]:
            # carryover basis inside a stack of three or more
            keys = ["deemed"] + rng.sample([k for k in L.LEVER_KEYS if k != "deemed"],
                                           max(2, size - 1))
            force_pos = "carryover"
        else:
            keys = rng.sample(L.LEVER_KEYS, size)
        keys = [k for k in L.LEVER_KEYS if k in keys]     # back into canonical order
        packages.append({k: _quiz_vals(rng, k, force_pos if k == "deemed" else None)
                         for k in keys})

    # The exclusion packages come from their own seeded stream, so the packages
    # above never reshuffle: the gains-at-death dial at an off-anchor exclusion
    # inside a stack, alternating between the two treatment positions.
    rng2 = random.Random(cfg["seed_exem"])
    for i in range(cfg["n_exem"]):
        size = rng2.randint(3, 5)
        keys = ["deemed"] + rng2.sample([k for k in L.LEVER_KEYS if k != "deemed"],
                                        size - 1)
        keys = [k for k in L.LEVER_KEYS if k in keys]
        pos = "carryover" if i % 2 else "deemed"
        exem = _presentable_rate(rng2, 0.25, 4.75, [0.0, 1.0, 5.0], step=0.25) * 1e6
        pkg = {k: _quiz_vals(rng2, k, None) for k in keys}
        pkg["deemed"] = {"pos": pos, "exem": exem}
        packages.append(pkg)
    return packages


# --------------------------------------------------------------------------- #
# The whole batch, in order
# --------------------------------------------------------------------------- #
def build_scenarios():
    """Return the ordered [(id, kind, scenario)] list for the full dial batch."""
    out = []
    for key in L.LEVER_KEYS:                                        # each dial's own curve
        for vals, run in L.solo_cells(key):
            if not run:
                continue
            out.append((f"s_{key}_{vals_slug(key, vals)}".rstrip("_"), "solo", {key: vals}))
    for a, b in itertools.combinations(L.LEVER_KEYS, 2):            # 28 pairs at reference
        out.append((f"pr_{a}_{b}", "pair", {a: L.REF[a], b: L.REF[b]}))
    # Positions on the gains-at-death ladder are never reached by scaling:
    # every interaction touching the ladder is
    # measured by a real run at each position. The pr_*_deemed rows cover
    # deemed realization; these cover carryover basis.
    non_deemed = [k for k in L.LEVER_KEYS if k != "deemed"]
    for a in non_deemed:                                            # 7 carryover pairs
        out.append((f"prco_{a}_deemed", "pairco",
                    {a: L.REF[a], "deemed": {"pos": "carryover"}}))
    for a, b, c in itertools.combinations(L.LEVER_KEYS, 3):         # all 56 triples
        out.append((f"t_{a}_{b}_{c}", "triple", {a: L.REF[a], b: L.REF[b], c: L.REF[c]}))
    for a, b in itertools.combinations(non_deemed, 2):              # 21 carryover triples
        out.append((f"tco_{a}_{b}_deemed", "tripco",
                    {a: L.REF[a], b: L.REF[b], "deemed": {"pos": "carryover"}}))
    for a, va, b, vb in L.PAIR_CORNERS:                             # 10 corners
        out.append((f"pc_{compact_slug(a, va)}_{compact_slug(b, vb)}", "corner",
                    {a: va, b: vb}))
    for i, pkg in enumerate(make_quiz()):                           # 20 holdout packages
        out.append((f"q{i + 1:02d}", "quiz", pkg))
    out.append(("stack_ref", "stack", {k: L.REF[k] for k in L.LEVER_KEYS}))
    return out


# --------------------------------------------------------------------------- #
# Checks run after writing
# --------------------------------------------------------------------------- #
def self_check(scenarios):
    problems = []

    # scenario names are unique
    ids = [s[0] for s in scenarios]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"duplicate IDs: {dupes}")

    # every law file written parses as YAML
    n_files = 0
    for scen_id, _, _ in scenarios:
        cdir = os.path.join(TAXLAW_ROOT, scen_id)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            n_files += 1
            try:
                with open(os.path.join(cdir, fname)) as fh:
                    yaml.safe_load(fh)
            except yaml.YAMLError as e:
                problems.append(f"{scen_id}/{fname}: YAML parse error: {e}")

    # the two-parameter dials cover their full anchor grid
    for key in ("wealth", "estate"):
        lv = L.BY_KEY[key]
        p1, p2 = lv["params"]
        want = {(a1, a2) for a1 in p1["anchors"] for a2 in p2["anchors"]}
        got = {(v[p1["key"]], v[p2["key"]]) for v, _ in L.solo_cells(key)}
        if want != got:
            problems.append(f"{key}: bilinear grid incomplete: missing {want - got}")

    # the taxable-maximum template matches the shape of the baseline payroll file
    with open(os.path.join(REPO, "config", "scenarios", "tax_law", "default", "pr.yaml")) as fh:
        base_pr = yaml.safe_load(fh)
    tm = yaml.safe_load("".join(L.taxmax_files({"on": 1})["pr.yaml"]))
    for sub, block in tm.items():
        if sub not in base_pr:
            problems.append(f"taxmax: subparameter {sub!r} not in baseline pr.yaml")
            continue
        base_len = len(base_pr[sub]["value"])
        for yr, v in block["value"].items():
            if len(v) != base_len:
                problems.append(f"taxmax: {sub} value length {len(v)} != baseline {base_len}")

    # holdouts really are held out: no holdout repeats a fitted run's settings
    fitted = {json.dumps(s[2], sort_keys=True) for s in scenarios
              if s[1] in ("solo", "pair", "triple", "pairco", "tripco")}
    for scen_id, kind, scen in scenarios:
        if kind in ("quiz", "corner", "stack") and json.dumps(scen, sort_keys=True) in fitted:
            problems.append(f"{scen_id}: holdout scenario duplicates a fitted run")

    # the quiz composition rules (full builds only; a patch batch has no quiz)
    quiz = [(i, s) for i, (sid, kind, s) in enumerate(scenarios) if kind == "quiz"]
    if quiz:
        n_heavy = sum(1 for _, s in quiz if sum(k in L.CLUSTER for k in s) >= 3)
        n_carry = sum(1 for _, s in quiz
                      if s.get("deemed", {}).get("pos") == "carryover" and len(s) >= 3)
        if n_heavy < L.QUIZ["min_cluster_heavy"]:
            problems.append(f"quiz: only {n_heavy} cluster-heavy packages")
        if n_carry < L.QUIZ["min_carryover"]:
            problems.append(f"quiz: only {n_carry} carryover 3+ stacks")

    return problems, n_files


# --------------------------------------------------------------------------- #
def emit(scenarios, runscript_path, legend_path, include_baseline=True):
    rows = []
    if include_baseline:
        rows.append({
            "ID": "baseline", "tax_law": "default", "behavior": "",
            "years": L.YEARS, "economy": L.CORP_OFF_ECONOMY,
        })
    for scen_id, _, scen in scenarios:
        rows.append(runscript_row(scen_id, scen))
    with open(runscript_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(legend_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "kind", "levers_json", "label"])
        for scen_id, kind, scen in scenarios:
            w.writerow([scen_id, kind, json.dumps(scen, sort_keys=True),
                        scen_label(scen)])
    return len(rows)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--patch":
        n = sys.argv[2] if len(sys.argv) > 2 else "4"
        if n not in L.PATCHES or not L.PATCHES[n]:
            sys.exit(f"levers.PATCHES[{n!r}] is empty — nothing to patch")
        scenarios = [(p["id"], p.get("kind", "patch"), p["levers"]) for p in L.PATCHES[n]]
        path = os.path.join(RUNSCRIPT_DIR, f"dials_patch{n}.csv")
        legend = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              f"dials_patch{n}_legend.csv")
        # The baseline row is required: every scenario's revenue estimate is a
        # difference against the baseline run in the same runscript.
        n_rows = emit(scenarios, path, legend, include_baseline=True)
        problems, n_files = self_check(scenarios)
        report(problems, n_rows, n_files, path)
        return

    if os.path.isdir(TAXLAW_ROOT):
        shutil.rmtree(TAXLAW_ROOT)
    os.makedirs(TAXLAW_ROOT, exist_ok=True)
    os.makedirs(RUNSCRIPT_DIR, exist_ok=True)
    scenarios = build_scenarios()
    for n in L.LIVE_PATCHES:
        scenarios += [(p["id"], p.get("kind", "patch"), p["levers"])
                      for p in L.PATCHES[n]]
    n_rows = emit(scenarios, RUNSCRIPT, LEGEND)
    problems, n_files = self_check(scenarios)
    kinds = {}
    for _, kind, _ in scenarios:
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"Scenario mix: {kinds}")
    report(problems, n_rows, n_files, RUNSCRIPT)


def report(problems, n_rows, n_files, path):
    print(f"Wrote {n_rows} runscript rows -> {path}")
    print(f"Wrote legend; {n_files} YAML files parsed clean")
    if problems:
        print("SELF-CHECK FAILURES:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("Self-checks: ALL GREEN")


if __name__ == "__main__":
    main()
