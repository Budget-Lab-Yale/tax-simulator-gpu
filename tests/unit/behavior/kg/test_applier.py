"""The gains applier, against numbers worked out by hand.

Three age cells and four households, with round numbers, so every figure below
can be checked on paper.  Ages 40, 41 and 42 stand in for the model's 18-to-80
grid; nothing in the applier depends on the grid's length.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from taxmodel.behavior.gains.settings import (
    ESTATE_VALUATION_FILE,
    read_applier_settings,
)
from taxmodel.behavior.kg.applier import (
    CORPORATE_MARKDOWN_GAINS_COLUMN,
    DEEMED_ASSET_CLASSES,
    DEEMED_EXPECTED_COLUMN,
    DEEMED_FULL_COLUMN,
    EQUIVALENT_TAX_RATE_BASELINE_COLUMN,
    EQUIVALENT_TAX_RATE_SCENARIO_COLUMN,
    LOCKIN_COLUMN,
    apply_cell_table,
)
from taxmodel.behavior.kg.stepper import CELL_TABLE_FIELD_INDEX, CELL_TABLE_FIELDS
from taxmodel.behavior.kg.stock_inputs import KG_AGE_MIN
from tests.tolerances import CENT


def make_cell_table(rows):
    """Build a [cell, field] table from a dict of column name to values."""

    return jnp.stack(
        [jnp.asarray(rows[name], dtype=jnp.float64) for name in CELL_TABLE_FIELDS],
        axis=-1,
    )


@pytest.fixture
def cell_table():
    """Three cells, all round.

    Cell 0 sells more under the reform and holds more gain than baseline; cell 1
    is untouched; cell 2 realized nothing at baseline, so its lock-in has to fall
    back to holdings.
    """

    zeros = [0.0, 0.0, 0.0]
    return make_cell_table(
        {
            "baseline_stock": [1_000_000.0, 500_000.0, 200_000.0],
            "baseline_realizations": [100_000.0, 50_000.0, 0.0],
            "baseline_realization_rate": [0.10, 0.10, 0.0],
            "scenario_realization_rate": [0.12, 0.10, 0.0],
            "scenario_gain_share": [0.40, 0.40, 0.40],
            "effective_mortality": [0.02, 0.02, 0.02],
            "stock_change_entering": zeros,
            "corporate_gain_debit": zeros,
            "charitable_bequest_share": [0.10, 0.0, 0.0],
            "forgiven_share": zeros,
            "routed_to_heirs_share": zeros,
            "realized_at_death_share": zeros,
            "decedent_stock": zeros,
            "charitable_death_stock": zeros,
            "taxable_death_stock": zeros,
            "stock_change_from_survivors": zeros,
            "stock_change_from_inheritance": zeros,
            "equivalent_tax_rate_baseline": [0.15, 0.16, 0.17],
            "equivalent_tax_rate_scenario": [0.20, 0.21, 0.22],
            "conversion_inflow": zeros,
            "rate_factor": [1.2, 1.0, 1.0],
            "extra_realizations": [24_000.0, 0.0, 5_000.0],
            # The stock at death is the cell's stock times the entering-stock
            # ratio (1.1 in cell 0) less the year's sales: 0.88 x 1.1 = 0.968
            # in cell 0, and 0.9 in cell 1, which sells a tenth of its gain.
            "deemed_factor": [0.968, 0.9, 1.0],
        }
    )


@pytest.fixture
def households():
    """Four households: two in cell 0, one in cell 1, one in cell 2."""

    frame = {
        "age1": jnp.asarray([KG_AGE_MIN, KG_AGE_MIN, KG_AGE_MIN + 1, KG_AGE_MIN + 2]),
        "age2": jnp.asarray([KG_AGE_MIN, KG_AGE_MIN, KG_AGE_MIN + 1, KG_AGE_MIN + 2]),
        # Single, single, married joint, single.
        "filing_status": jnp.asarray([1, 1, 2, 1]),
        "kg_lt": jnp.asarray([60_000.0, 40_000.0, 50_000.0, 0.0]),
        "unrealized_gain_stock": jnp.asarray(
            [400_000.0, 600_000.0, 500_000.0, 200_000.0]
        ),
        "m_household": jnp.asarray([0.02, 0.02, 0.01, 0.05]),
        "p_widow": jnp.zeros(4),
    }
    for name in DEEMED_ASSET_CLASSES:
        frame[f"value.{name}"] = jnp.zeros(4)
        frame[f"basis.{name}"] = jnp.zeros(4)
    # One household holds appreciated equities: $1,000,000 worth against a
    # $400,000 basis, so a $600,000 gain.
    frame["value.equities"] = jnp.asarray([1_000_000.0, 0.0, 0.0, 0.0])
    frame["basis.equities"] = jnp.asarray([400_000.0, 0.0, 0.0, 0.0])
    return frame


def test_rate_and_lockin_match_hand_computation(cell_table, households):
    """The rate factor and the half-and-half allocation, to the cent.

    Cell 0 realized $100,000 at baseline and holds $1,000,000 of gain, and its
    two households realized $60,000 and $40,000 and hold $400,000 and $600,000.
    So the first household takes 60 percent of the cell's extra realizations by
    the realizations measure and 40 percent by the holdings measure, and half of
    each is 50 percent — $12,000 of the cell's $24,000.  Its rate channel scales
    $60,000 by 1.2, to $72,000.

    Cell 2 realized nothing, so its household falls back entirely to holdings:
    it holds the cell's whole $200,000 and takes all $5,000.
    """

    changed = apply_cell_table(households, {}, {}, {"cell_table": cell_table})

    assert changed[LOCKIN_COLUMN][0] == pytest.approx(12_000.0, abs=CENT)
    assert changed[LOCKIN_COLUMN][1] == pytest.approx(12_000.0, abs=CENT)
    assert changed[LOCKIN_COLUMN][2] == pytest.approx(0.0, abs=CENT)
    assert changed[LOCKIN_COLUMN][3] == pytest.approx(5_000.0, abs=CENT)

    assert changed["kg_lt"][0] == pytest.approx(72_000.0 + 12_000.0, abs=CENT)
    assert changed["kg_lt"][1] == pytest.approx(48_000.0 + 12_000.0, abs=CENT)
    assert changed["kg_lt"][2] == pytest.approx(50_000.0, abs=CENT)
    assert changed["kg_lt"][3] == pytest.approx(5_000.0, abs=CENT)


def test_deemed_death_gains_are_expectations_and_stay_out_of_realized_gains(
    cell_table, households
):
    """The deemed gain is mortality-weighted, and never touches kg_lt.

    The first household's $600,000 equity gain is deemed realized in full under
    a regime that taxes equities at death.  Its cell leaves 10 percent to
    charity, and the gain it still holds when it dies is 0.968 times baseline:
    1.1 times baseline entering the year, less the 12 percent of it sold before
    the death.  So the gain on the final return is 0.9 x 0.968 x $600,000 =
    $522,720, and the expected amount at a 2 percent chance of dying is
    $10,454.40.
    """

    params = {
        "cell_table": cell_table,
        # Equities are deemed realized at death in full; nothing else is.
        "realized_at_death_share": (1.0, 0.0, 0.0, 0.0, 0.0),
        "regime_code": (2, 0, 0, 0, 0),
    }
    changed = apply_cell_table(households, {}, {}, params)

    assert changed[DEEMED_FULL_COLUMN][0] == pytest.approx(522_720.0, abs=CENT)
    assert changed[DEEMED_EXPECTED_COLUMN][0] == pytest.approx(10_454.40, abs=CENT)

    # The deemed gain is several times this household's realized gains. If any
    # of it leaked into kg_lt the number below would move.
    without_death = apply_cell_table(households, {}, {}, {"cell_table": cell_table})
    assert changed["kg_lt"] == pytest.approx(without_death["kg_lt"], abs=CENT)


def test_pass_through_interests_are_valued_below_market_at_death(
    cell_table, households
):
    """Pass-through value is scaled by the factor; equities are not.

    The first household also holds a pass-through interest worth $1,000,000
    against a $400,000 basis.  At a factor of 0.612 its gain at death is
    0.612 x $1,000,000 - $400,000 = $212,000.  The equity gain stays at the
    full $600,000.
    """

    households["value.pass_throughs"] = jnp.asarray([1_000_000.0, 0.0, 0.0, 0.0])
    households["basis.pass_throughs"] = jnp.asarray([400_000.0, 0.0, 0.0, 0.0])
    params = {
        "cell_table": cell_table,
        "realized_at_death_share": (1.0, 1.0, 0.0, 0.0, 0.0),
        "regime_code": (2, 2, 0, 0, 0),
        "pass_through_valuation_factor": 0.612,
    }
    changed = apply_cell_table(households, {}, {}, params)
    assert changed[DEEMED_FULL_COLUMN][0] == pytest.approx(
        0.9 * 0.968 * (600_000.0 + 212_000.0), abs=CENT
    )


def test_a_pass_through_interest_valued_below_its_basis_has_no_gain_at_death(
    cell_table, households
):
    """0.612 x $500,000 is less than the $400,000 basis, so the gain is zero."""

    households["value.pass_throughs"] = jnp.asarray([500_000.0, 0.0, 0.0, 0.0])
    households["basis.pass_throughs"] = jnp.asarray([400_000.0, 0.0, 0.0, 0.0])
    params = {
        "cell_table": cell_table,
        "realized_at_death_share": (1.0, 1.0, 0.0, 0.0, 0.0),
        "regime_code": (2, 2, 0, 0, 0),
        "pass_through_valuation_factor": 0.612,
    }
    changed = apply_cell_table(households, {}, {}, params)
    assert changed[DEEMED_FULL_COLUMN][0] == pytest.approx(
        0.9 * 0.968 * 600_000.0, abs=CENT
    )


def test_the_pass_through_factor_is_the_estate_taxs_fitted_value():
    """The applier reads rho_pt from the estate valuation fit, not a copy."""

    fit = yaml.safe_load(Path(ESTATE_VALUATION_FILE).read_text())
    settings = read_applier_settings()
    assert settings.pass_through_valuation_factor == fit["rho_pt"]["value"]


def test_death_gain_exclusion_takes_the_first_dollars_pro_rata(cell_table, households):
    """A $100,000 exclusion against a $600,000 pool leaves five sixths taxable."""

    params = {
        "cell_table": cell_table,
        "realized_at_death_share": (1.0, 0.0, 0.0, 0.0, 0.0),
        "regime_code": (2, 0, 0, 0, 0),
        "death_gain_exclusion": 100_000.0,
    }
    changed = apply_cell_table(households, {}, {}, params)
    assert changed[DEEMED_FULL_COLUMN][0] == pytest.approx(
        0.9 * 0.968 * 600_000.0 * (500_000.0 / 600_000.0), abs=CENT
    )


def test_the_two_deferred_dollar_tax_rates_come_through_by_cell(cell_table, households):
    """Entity shifting reads these, so they have to arrive on the household."""

    changed = apply_cell_table(households, {}, {}, {"cell_table": cell_table})
    assert changed[EQUIVALENT_TAX_RATE_BASELINE_COLUMN] == pytest.approx(
        jnp.asarray([0.15, 0.15, 0.16, 0.17])
    )
    assert changed[EQUIVALENT_TAX_RATE_SCENARIO_COLUMN] == pytest.approx(
        jnp.asarray([0.20, 0.20, 0.21, 0.22])
    )


def test_the_deemed_tax_is_the_difference_between_two_evaluations(households):
    """The tax on the death gain is the difference between two evaluations.

    The household as it is, and the same household with the deemed gain added
    to its realized gains (`years/worlds.py`). A flat 20 percent rate on
    realized gains makes the difference exactly 20 percent of the deemed gain,
    and the expected amount that times the household's chance of dying.
    """
    from taxmodel.years import worlds

    def flat_rate_calculator(records, law):
        del law
        return {"liab_iit_net": 0.20 * jnp.asarray(records["kg_lt"])}

    frame = dict(households)
    frame[DEEMED_FULL_COLUMN] = jnp.asarray([522_720.0, 0.0, 0.0, 0.0])

    added = worlds.deemed_death_gain_by_evaluation_point(frame)
    realized_gain = jnp.asarray(frame["kg_lt"], dtype=jnp.float64)
    with_the_deemed_gain = [
        flat_rate_calculator({**frame, "kg_lt": realized_gain + gain}, None)[
            "liab_iit_net"
        ]
        for gain in added
    ]
    expected_tax, estate_deduction = worlds.the_deemed_death_tax(
        flat_rate_calculator(frame, None), with_the_deemed_gain, frame
    )
    assert expected_tax[0] == pytest.approx(0.02 * 0.20 * 522_720.0, abs=CENT)
    assert estate_deduction[0] == pytest.approx(0.20 * 522_720.0, abs=CENT)
    assert expected_tax[1] == pytest.approx(0.0, abs=CENT)


def test_a_deemed_tax_below_zero_is_floored(households):
    """A household with no deemed gain owes exactly zero, not a shade below.

    The two calculator runs the difference is taken between do not have to
    agree in the last bit, so a household with a deemed gain of zero can come
    out with a tax of minus a fraction of a cent. The deemed gain is floored at
    zero per asset class, so a negative amount is rounding noise and both
    returned amounts are floored.
    """
    from taxmodel.years import worlds

    frame = dict(households)
    frame[DEEMED_FULL_COLUMN] = jnp.zeros(4)

    added = worlds.deemed_death_gain_by_evaluation_point(frame)
    living = {"liab_iit_net": jnp.zeros(4)}
    # What the second run would report for a household owing nothing: the same
    # number, one unit in the last place below.
    a_shade_below_zero = [jnp.full(4, -1e-9) for _ in added]
    expected_tax, estate_deduction = worlds.the_deemed_death_tax(
        living, a_shade_below_zero, frame
    )
    assert np.asarray(expected_tax) == pytest.approx(np.zeros(4))
    assert np.asarray(estate_deduction) == pytest.approx(np.zeros(4))
    assert (np.asarray(expected_tax) >= 0.0).all()
    assert (np.asarray(estate_deduction) >= 0.0).all()


def test_the_markdown_gains_change_is_the_lockin_the_debit_took_away(
    cell_table, households
):
    """The recorded change equals the realized gains lost to the debit.

    Cell 0 enters the year $50,000 above its baseline stock and the corporate
    markdown claims $20,000 of it, so at a 12 percent pace the cell's extra
    realizations are $3,600 where they would have been $6,000. Each of its two
    households takes half, so each loses $1,200.
    """

    turnover = 0.12
    entering, debit = 50_000.0, 20_000.0
    table = cell_table.at[0, CELL_TABLE_FIELD_INDEX["stock_change_entering"]].set(
        entering
    )
    table = table.at[0, CELL_TABLE_FIELD_INDEX["corporate_gain_debit"]].set(debit)
    with_debit = table.at[0, CELL_TABLE_FIELD_INDEX["extra_realizations"]].set(
        turnover * (entering - debit)
    )
    without_debit = table.at[0, CELL_TABLE_FIELD_INDEX["extra_realizations"]].set(
        turnover * entering
    )

    changed = apply_cell_table(households, {}, {}, {"cell_table": with_debit})
    no_debit = apply_cell_table(households, {}, {}, {"cell_table": without_debit})

    np.testing.assert_allclose(
        changed[CORPORATE_MARKDOWN_GAINS_COLUMN],
        changed["kg_lt"] - no_debit["kg_lt"],
        atol=CENT,
    )
    np.testing.assert_allclose(
        changed[CORPORATE_MARKDOWN_GAINS_COLUMN],
        [-1_200.0, -1_200.0, 0.0, 0.0],
        atol=CENT,
    )


def test_no_markdown_debit_records_no_gains_change(cell_table, households):
    changed = apply_cell_table(households, {}, {}, {"cell_table": cell_table})

    np.testing.assert_array_equal(changed[CORPORATE_MARKDOWN_GAINS_COLUMN], np.zeros(4))
