"""Taxes for a file of records under one year's law: a parquet file in, a parquet file out.

For a caller that holds its own records and wants each record's computed taxes, such as the Tax-Data rebuild, which
reweights its records to published totals of computed items (taxable income, credits, tax). The command reads one
parquet file with one row per record: an `id` column and every column the tax chain and the frame steps read, in
Tax-Data's form (the overlapping wage columns as Tax-Data writes them). It writes one row per record, keyed by the
same id, with every column the chain computes.

    python -m taxmodel.run.batch --records in.parquet --year 2019 --out out.parquet

Steps, all the model's own code:
1. the records are checked for every column the chain and the frame steps read;
2. the overlapping wage columns are split and the random draws keyed by record id are attached, as the Tax-Data
   bridge does for a production run (with --zero-draws every draw is zero instead);
3. the year's law is compiled from the default law folder (or the alternative named) with the price indexes of the
   pinned Macro-Projections publication (with --bundled-macro, the copy of them in resources/macro_projections);
4. the frame steps (the law join, the state tax workaround, the capital gain price ratio, net worth) and the whole
   income and payroll tax chain run under that law; the estate tax is left out;
5. with --wage-increment, the chain runs a second time with the primary earner's wages raised by that many dollars,
   and the marginal rates on wages are the change in tax divided by the increment.

Beside the output it writes `<out>.manifest.json`: the commit the caller names, where the code was imported from,
the JAX version and devices, the law year, the draw seed (or that the draws were zero), the price index source, the input file's SHA-256, a few law
values by filing status, the columns supplied but not read, and any column with missing values.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

# Columns the frame steps compute from the records, so a caller does not supply them.
MADE_BY_THE_MODEL = {
    "kg_lt_cpi_ratio",
    "salt_workaround_part",
    "salt_workaround_scorp",
    "net_worth",
    "wages_base1",
    "wages_base2",
}
# Frame-step columns written to the output with the chain's own columns.
FRAME_STEP_OUTPUTS = ("salt_workaround_part", "salt_workaround_scorp", "kg_lt_cpi_ratio", "net_worth")
# Law values written to the manifest, one per filing status (single, joint, separate, head of household).
MANIFEST_LAW_VALUES = ("std.value", "item.salt_limit", "ctc.need_ssn", "ctc.value_young1", "amt.exempt")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Taxes for a parquet file of records under one year's law.")
    p.add_argument("--records", required=True, help="input parquet, one row per record, with an id column")
    p.add_argument("--year", type=int, required=True, help="the tax year whose law is applied")
    p.add_argument("--out", required=True, help="output parquet, one row per record, keyed by id")
    p.add_argument("--seed", type=int, default=None, help="seed of the random draws keyed by id (the model's default)")
    p.add_argument("--zero-draws", action="store_true",
                   help="set every random draw to zero, so that each random rule (care credit takeup, the excess "
                        "business loss limit, and so on) applies to every record it can")
    p.add_argument("--bundled-macro", action="store_true",
                   help="read the price indexes from the copy in resources/macro_projections, not from the cluster")
    p.add_argument("--alternative", default=None, help="an alternative law folder under config/scenarios/tax_law/alternatives")
    p.add_argument("--commit", default="", help="the commit of this code, recorded in the manifest")
    p.add_argument("--wage-increment", type=float, default=0.0,
                   help="dollars added to the primary earner's wages for the marginal rates on wages (0: not computed)")
    p.add_argument("--law-sections", nargs="*", default=[],
                   help="sections of the year's law (for example qbi_ded) whose values for each record are written as "
                        "columns law.<section>.<field>")
    return p.parse_args(argv)


def record_draws(ids: np.ndarray, seed: int, zero: bool) -> pd.DataFrame:
    """The random draws for each record id: keyed by id and seed, or all zero."""
    from taxmodel.frame import prepare

    draws = prepare.keyed_draws(ids, seed=seed)
    if zero:
        draws[list(prepare.DRAW_COLUMNS)] = 0.0
    return draws


def run_chain(frame: pd.DataFrame, year: int, annual, builder, double_run: bool, seed: int, zero_draws: bool):
    """The frame steps and the tax chain on one frame of records. Returns the computed columns, the frame-step
    columns, and the law read for each record."""
    import jax.numpy as jnp

    from taxmodel.frame import prepare
    from taxmodel.taxes import statutory_order
    from taxmodel.years import one_year

    frame = prepare.split_overlapping_wages(frame, year)
    draws = record_draws(frame["id"].to_numpy(), seed, zero_draws)
    frame = frame.merge(draws, on="id", how="left", validate="one_to_one")
    raw = {c: jnp.asarray(frame[c].to_numpy(np.float64)) for c in frame.columns if c != "id"}
    raw["id"] = jnp.asarray(frame["id"].to_numpy(np.int64))
    raw["valid"] = jnp.ones(len(frame), dtype=bool)
    law = one_year.law_read_for_each_household(annual, raw["filing_status"])
    built = builder(raw, law, jnp.asarray(float(year)), apply_adjusters=False)
    taxes = statutory_order.run_taxes(built, law.calculator, None, double_run=double_run)
    return taxes, built, law


def main(argv=None) -> None:
    a = parse_arguments(argv)
    t0 = time.time()
    import jax

    import taxmodel
    from taxmodel.config import check_toolchain
    from taxmodel.frame import build_frame, prepare
    from taxmodel.law import indexes
    from taxmodel.scenario import law_by_year, provenance
    from taxmodel.taxes import statutory_order

    check_toolchain()
    where = os.path.realpath(taxmodel.__file__)
    seed = prepare.DEFAULT_SEED if a.seed is None else a.seed
    print(f"taxmodel from {where}; jax {jax.__version__} on {jax.devices()}", flush=True)

    frame = pd.read_parquet(a.records)
    n = len(frame)
    if "id" not in frame or not frame["id"].is_unique:
        raise ValueError(f"{a.records}: the id column is missing or not unique")
    needed = (set(statutory_order.RECORD_COLUMNS) | set(statutory_order.ESTATE_RECORD_COLUMNS)
              | set(build_frame.FRAME_STEP_RECORD_COLUMNS))
    needed -= MADE_BY_THE_MODEL | set(prepare.DRAW_COLUMNS)
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"{a.records} lacks columns the chain reads: {missing}")
    unread = sorted(set(frame.columns) - needed - set(prepare.OVERLAPPING_COLUMNS) - {"id"})
    print(f"{n:,} records; {len(needed)} columns read; supplied and not read: {unread}", flush=True)

    if a.bundled_macro:
        growth = indexes.derive_index_table(indexes.BUNDLED_MACRO_PROJECTIONS)
        index_source = str(indexes.BUNDLED_MACRO_PROJECTIONS)
    else:
        growth, index_source = provenance.index_series()
    scenario = provenance.load_scenario([a.year], alternative=a.alternative, growth=growth)
    series = law_by_year.bind_annual_laws(scenario.values, scenario.years, scenario.law_table, [a.year])
    annual = series.by_year[0]
    double_run = statutory_order.charity_double_run_applies(annual.calculator)
    builder = build_frame.make_frame_builder(scenario.indexes, scenario.cluster_death_weight_cap)
    t1 = time.time()

    taxes, built, law = run_chain(frame, a.year, annual, builder, double_run, seed, a.zero_draws)
    out = {"id": frame["id"].to_numpy(np.int64)}
    for k, v in taxes.items():
        out[k] = np.asarray(v, dtype=np.float64)
    for k in FRAME_STEP_OUTPUTS:
        out[k] = np.asarray(built[k], dtype=np.float64)
    # The chain zeroes these Schedule A components on non-itemizers and
    # selects the winning return when it compares charitable deduction forms.
    # Keep the components before the overall Pease reduction, as on SOI's
    # category lines. The total-minus-four residual is separate: Pease and
    # later overall limits can make it differ from actual other deductions.
    for name, source in (("claimed_medical", "med_item_ded"),
                         ("claimed_taxes", "salt_item_ded"),
                         ("claimed_interest", "int_item_ded"),
                         ("claimed_charity", "char_item_ded")):
        out[name] = out[source].copy()
    out["claimed_other"] = (out["casualty_item_ded"] + out["misc_item_ded"]
                            + out["other_item_ded"])
    out["itemized_total_minus_four"] = (out["item_ded"] - out["claimed_medical"]
                                        - out["claimed_taxes"] - out["claimed_interest"]
                                        - out["claimed_charity"])
    for section in a.law_sections:
        values = getattr(law.calculator, section)
        for field in dataclasses.fields(values):
            out[f"law.{section}.{field.name}"] = np.broadcast_to(
                np.asarray(getattr(values, field.name), dtype=np.float64), (n,)).copy()

    if a.wage_increment > 0:
        raised = frame.copy()
        for c in ("wages", "wages1"):
            raised[c] = raised[c] + a.wage_increment
        taxes_raised, _, _ = run_chain(raised, a.year, annual, builder, double_run, seed, a.zero_draws)
        for name, column in (("mtr_wages_iit", "liab_iit_net"), ("mtr_wages_payroll", "liab_pr")):
            out[name] = (np.asarray(taxes_raised[column], np.float64) - out[column]) / a.wage_increment
    out = pd.DataFrame(out)
    t2 = time.time()

    nan = {k: int(out[k].isna().sum()) for k in out.columns if out[k].isna().any()}
    out.to_parquet(a.out, index=False)
    law_values = {k: [float(v) for v in np.asarray(annual.frame_columns[k]).ravel()]
                  for k in MANIFEST_LAW_VALUES if k in annual.frame_columns}
    manifest = {"commit": a.commit, "taxmodel_from": where, "jax": jax.__version__,
                "devices": [str(d) for d in jax.devices()], "law_year": a.year, "alternative": a.alternative,
                "draw_seed": "all draws zero" if a.zero_draws else seed, "records": n, "input": os.path.realpath(a.records),
                "input_sha256": file_sha256(a.records), "charity_double_run": bool(double_run),
                "index_table": index_source, "law_values_by_filing_status": law_values,
                "wage_increment": a.wage_increment, "law_sections": a.law_sections,
                "columns_supplied_not_read": unread, "missing_values_in_outputs": nan,
                "seconds": {"setup": round(t1 - t0, 1), "frame steps and taxes": round(t2 - t1, 1)}}
    with open(f"{a.out}.manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps({k: manifest[k] for k in ("law_year", "records", "devices", "seconds", "missing_values_in_outputs")}),
          flush=True)


if __name__ == "__main__":
    main()
