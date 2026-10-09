"""Every margin at its neutral values returns the columns it was given, exactly.

Every behavioral margin runs in every compiled program. A scenario that wants
no response gives the margin its neutral parameter values, so a margin at
neutral values must leave every column exactly as it was.

The test is bitwise, and it is run under a rate change big enough that a live
module would move plenty: the reform's marginal rates are ten points away from
the baseline's on every priced dollar.

One test per margin, ten in all, counting the capital-gains applier's slot.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior import loader, parameters
from taxmodel.taxes.calc.estate import ESTATE_ASSET_COLS

# How far apart the two laws' marginal rates are in this test: ten percentage
# points on every priced dollar, which is a large reform and makes a
# module that has not been silenced impossible to miss.
RATE_GAP = 0.10
BASELINE_RATE = 0.30

# The households the modules run on: two, with income of every kind the ten
# modules read, a real balance sheet, and both earners working.
HOUSEHOLD_COLUMNS: dict[str, float] = {
    "wages_base1": 180_000.0,
    "wages_base2": 90_000.0,
    "tips1": 12_000.0,
    "tips2": 4_000.0,
    "ot1": 9_000.0,
    "ot2": 3_000.0,
    "sole_prop1": 70_000.0,
    "sole_prop2": 20_000.0,
    "farm1": 15_000.0,
    "farm2": 5_000.0,
    "part_active": 250_000.0,
    "part_passive": 40_000.0,
    "part_se1": 150_000.0,
    "part_se2": 100_000.0,
    "part_active_loss": 5_000.0,
    "part_passive_loss": 2_000.0,
    "part_179": 1_000.0,
    "scorp_active": 120_000.0,
    "scorp_passive": 30_000.0,
    "scorp_active_loss": 4_000.0,
    "scorp_passive_loss": 1_500.0,
    "scorp_179": 800.0,
    "rent": 45_000.0,
    "txbl_int": 60_000.0,
    "div_ord": 25_000.0,
    "div_pref": 75_000.0,
    "kg_lt": 400_000.0,
    "kg_st": 20_000.0,
    "char_cash": 30_000.0,
    "char_noncash": 5_000.0,
    "auto_int_exp": 3_500.0,
    "value.equities": 3_000_000.0,
    "value.bonds": 1_500_000.0,
    "value.cash": 500_000.0,
    "value.pass_throughs": 2_000_000.0,
    "value.primary_home": 1_000_000.0,
    "value.primary_mortgage": 700_000.0,
    "value.credit_cards": 200_000.0,
    # The wealth-tax base the frame hands the chain. The wealth module writes
    # it, so at neutral it has to come back exactly as it went in.
    "net_worth": 7_100_000.0,
    # Both earners are 55, the household files jointly, and the year is one
    # in which every provision the modules look for is in law.
    "age1": 55.0,
    "age2": 55.0,
    "filing_status": 2.0,
    "year": 2030.0,
    "weight": 1_000.0,
    # The two spare per-household draws the tips module's extensive margin
    # uses to decide which untipped workers become tipped.
    "r.behavior1": 0.4,
    "r.behavior2": 0.7,
    # The three bound-law deduction limits the tips, overtime and
    # auto-interest modules look for, under the names the frame's law join
    # gives them. All three provisions are in law here, so a live module would
    # respond. The tips and overtime caps are OBBBA's $25,000 on deductible
    # tips and overtime pay.
    "below.tip_ded_limit": 25_000.0,
    "below.ot_ded_limit": 25_000.0,
    "agi.auto_int_ded_limit": 10_000.0,
    # The statutory corporate rate under each law, which entity shifting
    # prices the corporate side with.
    "corp.rate": 0.21,
    "corp.rate_baseline": 0.21,
}

# What each module's newly created columns hold when nothing happened. These
# columns are not in the frame the module was handed, so "returns what it was
# given" cannot be tested on them; what is tested is that they say nothing
# happened.
NOTHING_HAPPENED: dict[str, dict[str, float]] = {
    "tips": {"tips_wage_floor_binding": 0.0},
    "conversion": {"conversion_clamped_share": 0.0},
    "entity_shifting": {"corp_tax_change": 0.0},
    # The three retained-reporting factors the wealth and estate modules read
    # as "nothing was hidden".
    "evasion": {name: bs.NO_EVASION for name in bs.EVASION_FACTOR_COLS},
    "wealth": {"wealth_c_pub": 0.0, "wealth_c_priv": 0.0},
    "kg_dynamics": {
        "equivalent_tax_rate_baseline": 0.0,
        "equivalent_tax_rate_scenario": 0.0,
    },
}


def frame() -> dict[str, jnp.ndarray]:
    """Three households carrying every column the ten modules read.

    The third household has tips larger than wages, so its base wages are
    negative. An earner with negative base wages has
    nothing to relabel, so no module may move this household's pay at neutral
    values.
    """
    columns = {
        name: jnp.asarray([value, value * 0.5, value], dtype=jnp.float64)
        for name, value in HOUSEHOLD_COLUMNS.items()
    }
    columns["wages_base1"] = columns["wages_base1"].at[2].set(-500.0)
    # Every balance-sheet column the wealth and estate modules read, zero
    # where the household above does not hold that kind of asset.
    for name in (*ESTATE_ASSET_COLS, *bs.DEBT_COLS):
        columns.setdefault(name, jnp.zeros(3, dtype=jnp.float64))
    for name in bs.EVASION_FACTOR_COLS:
        columns[name] = jnp.full(3, bs.NO_EVASION, dtype=jnp.float64)
    # The two equivalent tax rates the gains applier's slot writes and entity
    # shifting reads. Each module is run on its own here, so the columns an
    # earlier one would have written are on the frame from the start, at the
    # no-gains-model values.
    columns["equivalent_tax_rate_baseline"] = jnp.zeros(3, dtype=jnp.float64)
    columns["equivalent_tax_rate_scenario"] = jnp.zeros(3, dtype=jnp.float64)
    # The two concealed shares the wealth module writes and the estate module
    # reads, at "nothing was concealed", for the same reason.
    columns["wealth_c_pub"] = jnp.zeros(3, dtype=jnp.float64)
    columns["wealth_c_priv"] = jnp.zeros(3, dtype=jnp.float64)
    return columns


def prices(rate: float) -> dict[str, jnp.ndarray]:
    """One marginal rate on every dollar and every asset any module prices."""
    priced = {}
    for family in loader.RESPONSE_FAMILIES:
        _function, manifest, _defaults = loader.import_family(family)
        for name in manifest.get("prices", ()):
            priced[name] = jnp.full(3, rate, dtype=jnp.float64)
    # The wealth module prices each balance-sheet column. The estate module's
    # one price, the tax on a dollar of the taxable estate, comes from its
    # manifest in the loop above.
    for name in (*ESTATE_ASSET_COLS, *bs.DEBT_COLS, "estate_base_bump"):
        priced[bs.wealth_tax_rate_key(name)] = jnp.full(3, rate, dtype=jnp.float64)
    return priced


def run_at_neutral(family: str) -> tuple[dict, dict]:
    """One margin at its neutral values, on the frame above, and that frame."""
    records = frame()
    reform = prices(BASELINE_RATE + RATE_GAP)
    baseline = prices(BASELINE_RATE)
    entries = [
        entry
        for entry in loader.chain_entries()
        if entry["manifest"].get("family") == family
    ]
    assert len(entries) == 1, f"{family} should be one module in the pinned chain"
    chain = loader.compose(entries, tuple(records), tuple(reform))
    after = chain(
        records,
        reform,
        baseline,
        {family: parameters.neutral_values(family)}
        if family in parameters.loader.FAMILY_MODULES
        else {},
    )
    return after, records


@pytest.mark.parametrize("family", loader.FAMILY_ORDER)
def test_the_margin_at_neutral_returns_what_it_was_given(family: str):
    after, before = run_at_neutral(family)

    expected_new = NOTHING_HAPPENED.get(family, {})
    for name, value in after.items():
        if name in before and name not in expected_new:
            # Bitwise, deliberately: a module that is off must not move a
            # household by the last bit of a dollar.
            np.testing.assert_array_equal(
                np.asarray(value),
                np.asarray(before[name]),
                err_msg=(
                    f"the {family} margin at its neutral values changed "
                    f"{name}, so switching it off is not switching it off"
                ),
            )
        elif name in expected_new:
            np.testing.assert_array_equal(
                np.asarray(value),
                np.full(3, expected_new[name]),
                err_msg=(
                    f"the {family} margin at its neutral values did not leave "
                    f"{name} at the value that says nothing happened"
                ),
            )
    # Nothing infinite or missing survived the neutral path, which is the
    # other half of the exactness promise: x + 0 * response(x) is bitwise x
    # only where response(x) is finite.
    for name, value in after.items():
        assert np.all(np.isfinite(np.asarray(value))), (
            f"the {family} margin at neutral produced a non-finite {name}"
        )


def test_the_whole_chain_at_neutral_returns_the_frame_it_was_given():
    """Ten margins in a row, all neutral, still change nothing."""
    records = frame()
    reform = prices(BASELINE_RATE + RATE_GAP)
    baseline = prices(BASELINE_RATE)

    chain = loader.compose(loader.chain_entries(), tuple(records), tuple(reform))
    after = chain(records, reform, baseline, parameters.all_neutral())

    for name, value in records.items():
        np.testing.assert_array_equal(
            np.asarray(after[name]),
            np.asarray(value),
            err_msg=f"the whole chain at neutral changed {name}",
        )


def test_two_elasticities_trace_to_the_same_program():
    """Two charitable elasticities compile to the same program.

    The traced program is written out for each and the two texts are compared.
    The elasticity reaches the module as an array the program reads, so a
    sensitivity run over many elasticities compiles once.
    """
    records = frame()
    reform = prices(BASELINE_RATE + RATE_GAP)
    baseline = prices(BASELINE_RATE)
    chain = loader.compose(loader.chain_entries(), tuple(records), tuple(reform))

    def recipe(elasticity: float) -> str:
        values = parameters.all_neutral()
        values["charity"] = dict(values["charity"])
        values["charity"]["elasticity"] = jnp.asarray(elasticity, dtype=jnp.float64)
        return str(
            jax.make_jaxpr(
                lambda numbers: chain(records, reform, baseline, numbers)
            )(values)
        )

    assert recipe(-0.5) == recipe(-1.0)


@pytest.mark.parametrize("family", loader.FAMILY_ORDER)
def test_a_margin_returns_only_the_columns_it_declares(family: str):
    """Each module, called on its own, returns columns its manifest writes."""
    records = frame()
    entry = next(
        entry
        for entry in loader.chain_entries()
        if entry["manifest"].get("family") == family
    )
    params = dict(entry.get("params", {}))
    if family in parameters.loader.FAMILY_MODULES:
        params.update(parameters.neutral_values(family))
    draws = {name: records[name] for name in loader.MODULE_DRAW_COLUMNS if name in records}
    returned = entry["function"](
        records,
        prices(BASELINE_RATE + RATE_GAP),
        prices(BASELINE_RATE),
        params,
        draws,
    )
    undeclared = set(returned) - set(entry["manifest"]["writes"])
    assert not undeclared, (
        f"the {family} margin returned {sorted(undeclared)}, which its manifest "
        "does not declare"
    )


def test_every_family_in_the_pinned_order_has_a_module():
    """The chain is the whole list: ten families, ten modules, in one order."""
    entries = loader.chain_entries()
    assert [entry["manifest"]["family"] for entry in entries] == list(
        loader.FAMILY_ORDER
    )


def test_a_missing_value_comes_back_missing():
    """A module gives back what it was given, blanks included.

    A household whose column holds no value comes out of the chain holding no
    value, not a zero. The response itself is worked out reading a blank as
    zero, because the arithmetic needs a number, and that reading does not
    reach the frame.

    Entity shifting is live here and this household shifts. It scales the
    columns it moves, and a scaled blank is still blank. `kg_lt` receives the
    gains offset, but the frame's equivalent tax rates are zero, so no offset
    moves and its blank survives.
    """
    records = frame()
    blank_first = {
        name: value.at[0].set(jnp.nan)
        for name, value in records.items()
        if name in ("part_active", "wages_base1", "kg_lt", "rent", "part_se1")
    }
    records = {**records, **blank_first}
    reform = prices(BASELINE_RATE + RATE_GAP)
    baseline = prices(BASELINE_RATE)

    chain = loader.compose(loader.chain_entries(), tuple(records), tuple(reform))
    # Every margin turned on, so that the columns are passed through by modules
    # that are actually computing a response rather than by modules at neutral.
    responding = parameters.all_neutral()
    responding["entity_shifting"] = {
        **responding["entity_shifting"],
        "semi_elasticity": jnp.asarray(0.6313333333333334),
    }
    responding["evasion"] = {
        **responding["evasion"],
        "elasticity_sole_prop_and_farm": jnp.asarray(0.046),
        "elasticity_pass_through": jnp.asarray(0.052),
        "elasticity_rent": jnp.asarray(0.040),
    }
    after = chain(records, reform, baseline, responding)

    for name in blank_first:
        assert np.isnan(np.asarray(after[name])[0]), (
            f"{name} came back with a value where the frame had none"
        )
        assert np.isfinite(np.asarray(after[name])[1]), (
            f"{name} lost the household that did have a value"
        )
