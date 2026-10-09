"""The add-backs equal what the corporate step's change is still worth in the
final conventional frame.

Two households run through the corporate step, then the three later steps
that scale the same income: the wealth haircut, income evasion and wealth-tax
concealment. The same households also run through the three steps without
the corporate step, with the same haircut fraction and the same prices. The
difference between the two final frames is what the corporate step changed,
as it stands at the end, and each add-back must equal it to the cent.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.evasion import debacker
from taxmodel.behavior.wealth import avoidance as wealth_avoidance
from taxmodel.frame.steps_jax import ESTATE_ASSET_COLS, ESTATE_DEBT_COLS
from taxmodel.reporting import corporate_spread
from taxmodel.years import income_adjustments, wealth_state
from taxmodel.years.income_adjustments import apply_corporate_adjustment
from tests.tolerances import assert_agrees_to_the_cent

# The share of each household's assets the haircut removes, and the capital
# share of pass-through income the haircut scales.
HAIRCUT_FRACTION = jnp.array([0.3, 0.1])
PASS_THROUGH_WEIGHT = 0.2

# Income columns, each signed as income, grouped as the add-backs group them.
FLOW_COLUMNS = {
    "div_ord": 1.0,
    "div_pref": 1.0,
    "txbl_int": 1.0,
    "exempt_int": 1.0,
    "rent": 1.0,
    "rent_loss": -1.0,
    "part_active": 1.0,
    "part_passive": 1.0,
    "part_active_loss": -1.0,
    "part_passive_loss": -1.0,
    "part_179": -1.0,
    "scorp_active": 1.0,
    "scorp_passive": 1.0,
    "scorp_active_loss": -1.0,
    "scorp_passive_loss": -1.0,
    "scorp_179": -1.0,
    "sole_prop1": 1.0,
    "sole_prop2": 1.0,
    "farm1": 1.0,
    "farm2": 1.0,
}
RETIREMENT_COLUMNS = ("txbl_ira_dist", "gross_pens_dist")
GAIN_COLUMNS = ("kg_st", "kg_lt")


def households() -> dict[str, jnp.ndarray]:
    """Two households with every kind of income the corporate step changes.

    The second has a Schedule C loss and no dividends, so the tests of
    positive income in the later steps go both ways.
    """
    values = {
        "weight": [100.0, 200.0],
        "div_ord": [100.0, 0.0],
        "div_pref": [40.0, 10.0],
        "txbl_int": [50.0, 20.0],
        "exempt_int": [30.0, 0.0],
        "rent": [50.0, 0.0],
        "rent_loss": [10.0, 5.0],
        "sole_prop1": [80.0, -30.0],
        "sole_prop2": [20.0, 10.0],
        "part_active": [30.0, 60.0],
        "part_passive": [15.0, 0.0],
        "part_active_loss": [10.0, 0.0],
        "part_passive_loss": [0.0, 4.0],
        "part_179": [2.0, 0.0],
        "scorp_active": [25.0, 5.0],
        "scorp_passive": [0.0, 12.0],
        "scorp_active_loss": [3.0, 0.0],
        "scorp_passive_loss": [0.0, 1.0],
        "scorp_179": [0.0, 2.0],
        "farm1": [12.0, 0.0],
        "farm2": [0.0, 0.0],
        "part_se1": [20.0, 40.0],
        "part_se2": [0.0, 0.0],
        "txbl_ira_dist": [100.0, 40.0],
        "txbl_pens_dist": [100.0, 30.0],
        "gross_pens_dist": [100.0, 30.0],
        "kg_lt": [100.0, 60.0],
        "kg_lt_basis": [100.0, 40.0],
        "kg_st": [50.0, 10.0],
    }
    records = {name: jnp.array(value) for name, value in values.items()}
    for name in ESTATE_ASSET_COLS:
        records[name] = jnp.array([500.0, 300.0])
    for name in ESTATE_DEBT_COLS:
        records[name] = jnp.array([20.0, 10.0])
    records["value.equities"] = jnp.array([1_000.0, 400.0])
    records["value.dc"] = jnp.array([200.0, 50.0])
    records["net_worth"] = sum(records[name] for name in ESTATE_ASSET_COLS) - sum(
        records[name] for name in ESTATE_DEBT_COLS
    )
    return records


def after_the_corporate_step(records):
    return apply_corporate_adjustment(
        records,
        dividend_factor=0.9,
        interest_factor=0.8,
        rent_factor=0.5,
        pass_through_factor=0.75,
        markdown_rate=0.1,
        flow_factor=-0.2,
        asset_exposure={"value.equities": 1.0, "value.dc": 0.5},
        corporate_gain_share=0.5,
        gain_dynamics_active=False,
    )


def after_the_later_steps(records):
    """The haircut, income evasion and wealth-tax concealment, in that order."""
    out = wealth_state.scale_by_haircut(records, HAIRCUT_FRACTION, PASS_THROUGH_WEIGHT)
    rates = {"reform": 0.45, "baseline": 0.30}
    evasion_prices = {
        world: {
            name: jnp.full(2, rate)
            for name in ("mtr_sole_prop1", "mtr_part_active", "mtr_rent")
        }
        for world, rate in rates.items()
    }
    out |= debacker.evasion(
        out,
        evasion_prices["reform"],
        evasion_prices["baseline"],
        {
            "elasticity_sole_prop_and_farm": 0.4,
            "elasticity_pass_through": 0.3,
            "elasticity_rent": 0.5,
            "top_end_multiplier": 1.0,
            "max_adjustment": 1.0,
        },
    )
    wealth_prices = {
        world: {bs.wealth_tax_rate_key(name): jnp.full(2, rate) for name in ESTATE_ASSET_COLS}
        for world, rate in (("reform", 0.02), ("baseline", 0.0))
    }
    out |= wealth_avoidance.wealth_avoidance(
        out,
        wealth_prices["reform"],
        wealth_prices["baseline"],
        {
            "public_elasticity": -8.0,
            "closely_held_elasticity": -12.0,
            "concealed_share_marketable": wealth_avoidance.CONCEALED_SHARE_MARKETABLE,
            "concealed_share_closely_held": wealth_avoidance.CONCEALED_SHARE_CLOSELY_HELD,
        },
    )
    return out


def change_between(with_step, without_step, columns) -> np.ndarray:
    return sum(
        sign * (np.asarray(with_step[name]) - np.asarray(without_step[name]))
        for name, sign in columns.items()
    )


def test_each_add_back_is_the_change_left_in_the_final_frame():
    records = households()
    with_step = after_the_later_steps(after_the_corporate_step(records))
    without_step = after_the_later_steps(records)
    # The later steps did scale these households' income.
    assert float(np.min(np.asarray(with_step[wealth_avoidance.CONCEALED_SHARE_MARKETABLE_COL]))) > 0.0
    assert float(np.min(np.abs(np.asarray(with_step["evasion_g_pt"]) - 1.0))) > 0.0

    add_backs = corporate_spread.add_backs(with_step, PASS_THROUGH_WEIGHT)

    assert_agrees_to_the_cent(
        add_backs[corporate_spread.CORPORATE_FLOWS_CHANGE],
        change_between(with_step, without_step, FLOW_COLUMNS),
        "the corporate flows add-back is the change left in dividends, "
        "interest, rent and pass-through income",
    )
    assert_agrees_to_the_cent(
        add_backs[corporate_spread.RETIREMENT_CHANGE],
        change_between(with_step, without_step, dict.fromkeys(RETIREMENT_COLUMNS, 1.0)),
        "the retirement add-back is the change left in IRA and pension distributions",
    )
    assert_agrees_to_the_cent(
        add_backs[corporate_spread.GAINS_CHANGE],
        change_between(with_step, without_step, dict.fromkeys(GAIN_COLUMNS, 1.0)),
        "the gains add-back is the change left in realized gains",
    )
    template = jnp.asarray(with_step["weight"])
    defined_contribution = bs.column(with_step, "value.dc", template)
    balances = defined_contribution + bs.column(with_step, "value.db", template)
    assert_agrees_to_the_cent(
        add_backs[corporate_spread.RETIREMENT_CHANGE_ACCRUAL],
        change_between(with_step, without_step, {"gross_pens_dist": 1.0})
        * (1.0 - np.asarray(defined_contribution / balances)),
        "the accrual retirement add-back is the defined-benefit share of the "
        "change left in gross pensions",
    )
    assert_agrees_to_the_cent(
        add_backs[corporate_spread.WAGE_CHANGE],
        np.zeros(2),
        "no employer-payroll step ran, so the wage add-back is zero",
    )


def test_the_recorded_pieces_add_up_to_the_corporate_step_s_totals():
    out = after_the_corporate_step(households())
    assert_agrees_to_the_cent(
        sum(np.asarray(out[piece]) for piece in income_adjustments.FLOW_CHANGE_PIECES),
        np.asarray(out["corp_dY_exog"]),
        "the flow pieces add up to the corporate step's change in outside income",
    )
    assert_agrees_to_the_cent(
        sum(np.asarray(out[piece]) for piece in income_adjustments.GAIN_CHANGE_PIECES),
        np.asarray(out[income_adjustments.GAINS_CHANGE_COLUMN]),
        "the gain pieces add up to the corporate step's change in realized gains",
    )
