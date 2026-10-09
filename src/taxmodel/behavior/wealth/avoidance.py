"""Wealth-tax avoidance: the share of wealth a household stops reporting.

A tax on wealth is paid every year on a stock the tax authority cannot see
directly, so the reported base falls when the rate rises. Reported wealth
falls by ``exp(change in the wealth tax rate * elasticity)``, at one
elasticity for marketable wealth (publicly traded and liquid) and a larger one
for closely held wealth, whose value is a matter of appraisal. At the default
values a 3% marginal rate erodes about 19% of reported marketable wealth and
about 40% of reported closely held wealth.

The response has two kinds, with different consequences elsewhere on the tax
return:

  * Concealment. The asset is no longer reported. The income it produces
    stops being reported too, and it is not in the estate at death.
  * Valuation gaming. The asset is still reported and its income is still
    reported; only the assessed value is lower. This shrinks the wealth-tax
    base and nothing else.

Marketable wealth can only be concealed, since it has an exchange price.
Closely held wealth, homes included, splits evenly between the two.

Two consistency rules:

  * Income that the evasion module hid removes the matching closely held
    assets from reported wealth.
  * Concealed wealth also escapes the reported estate. This module writes the
    two concealed shares on the record, and the estate avoidance module,
    which runs after it, folds them into the estate's concealed fraction.

The module rewrites the reported bases: the `net_worth` column the wealth tax
is charged on, and the reported income columns that feed the individual income
tax. The `value.*` asset columns, which the rest of the model reads as the
household's balance sheet, and the basis of capital gains are unchanged.

The response uses the change in the wealth tax rate between reform and
baseline law. The pricing pass gives a rate per asset column, and this module
collapses those into one rate for marketable wealth and one for closely held
wealth, each weighted by the dollars in that class.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.kg.applier import CORPORATE_MARKDOWN_GAINS_COLUMN

Array = jnp.ndarray
Records = Mapping[str, Array]

# The elasticities of reported wealth with respect to the marginal wealth tax
# rate: reported wealth scales by exp(rate change * elasticity). Both are
# judgment values from the standalone wealth tax model, not calibrated. The
# closely held value is the largest behavioral magnitude in the model and the
# first number a sensitivity check should vary.
PUBLIC_ELASTICITY = -7.0
CLOSELY_HELD_ELASTICITY = -17.0

# The share of the response that is concealment rather than valuation gaming.
# Both are judgment values. Full concealment for marketable wealth, which has
# a listed price. Half for closely held wealth, since private business
# valuation discounts are legal. Setting both to zero leaves reported wealth
# shrinking by the full response while income, gains and the estate are
# unchanged.
CONCEALED_SHARE_MARKETABLE = 1.0
CONCEALED_SHARE_CLOSELY_HELD = 0.5

DEFAULTS: dict[str, float] = {
    "public_elasticity": PUBLIC_ELASTICITY,
    "closely_held_elasticity": CLOSELY_HELD_ELASTICITY,
    "concealed_share_marketable": CONCEALED_SHARE_MARKETABLE,
    "concealed_share_closely_held": CONCEALED_SHARE_CLOSELY_HELD,
}

# Which reported income legs each class of concealed wealth takes with it,
# and which column decides whether the household has that income at all.
#
# Only a household reporting positive income of a kind conceals any of it; a
# loss is kept intact. The two earners' shares of self-employment income move
# with the business that produced them, which keeps the payroll tax base
# consistent with the income tax base.
#
# Retirement account balances are in the marketable class for the avoidance
# elasticity, but retirement distributions are not in this list because the
# plan reports them to the tax authority, so they cannot be concealed.
MARKETABLE = "marketable"
CLOSELY_HELD = "closely_held"
CONCEALED_INCOME_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    # (columns whose sum decides the household has this income,
    #  columns scaled down when it is concealed,
    #  which class of wealth produced it)
    (("txbl_int",), ("txbl_int",), MARKETABLE),
    (("div_ord",), ("div_ord",), MARKETABLE),
    (("div_pref",), ("div_pref",), MARKETABLE),
    (("kg_lt",), ("kg_lt",), MARKETABLE),
    (("part_active",), ("part_active", "part_se1", "part_se2"), CLOSELY_HELD),
    (("part_passive",), ("part_passive",), CLOSELY_HELD),
    (("scorp_active",), ("scorp_active",), CLOSELY_HELD),
    (("scorp_passive",), ("scorp_passive",), CLOSELY_HELD),
    (("sole_prop1", "sole_prop2"), ("sole_prop1", "sole_prop2"), CLOSELY_HELD),
    (("rent",), ("rent",), CLOSELY_HELD),
)

_SCALED_INCOME_COLS: tuple[str, ...] = tuple(
    dict.fromkeys(
        name for _gate, scaled, _kind in CONCEALED_INCOME_GROUPS for name in scaled
    )
)
_GATE_COLS: tuple[str, ...] = tuple(
    dict.fromkeys(
        name for gate, _scaled, _kind in CONCEALED_INCOME_GROUPS for name in gate
    )
)

# The two concealed shares this module writes on the record for the estate
# avoidance module.
CONCEALED_SHARE_MARKETABLE_COL = "wealth_c_pub"
CONCEALED_SHARE_CLOSELY_HELD_COL = "wealth_c_priv"
CONCEALED_SHARE_COLS: tuple[str, ...] = (
    CONCEALED_SHARE_MARKETABLE_COL,
    CONCEALED_SHARE_CLOSELY_HELD_COL,
)

# This module runs after the income-evasion module, whose hidden income
# removes assets from reported wealth, and after the capital-gains model,
# whose realized long-term gain it scales. The estate avoidance module runs
# after it and reads the two concealed shares. FAMILY_ORDER in
# behavior/loader.py sets the order.
MANIFEST: dict[str, object] = {
    "name": "wealth_avoidance",
    # The function in this file that is the response.
    "entry_point": "wealth_avoidance",
    # The reported share of each class of wealth is exp(rate change times
    # elasticity), so zero elasticities give reported shares of one and
    # concealed shares of zero.
    "zero_response": {"public_elasticity": 0.0, "closely_held_elasticity": 0.0},
    "family": "wealth",
    "reads": (
        *bs.MARKETABLE_ASSET_COLS,
        *bs.CLOSELY_HELD_ASSET_COLS,
        *bs.DEBT_COLS,
        *_SCALED_INCOME_COLS,
        *_GATE_COLS,
    ),
    # Columns the income-evasion module writes, if it ran. Where it did not,
    # each factor reads as one, the default
    # `balance_sheet.evaded_closely_held_income_share` supplies.
    # The gains applier's record of how much the corporate markdown lowered
    # realized long-term gains, present only when the gains model runs.
    "optional_reads": (*bs.EVASION_FACTOR_COLS, CORPORATE_MARKDOWN_GAINS_COLUMN),
    "writes": (
        "net_worth",
        *_SCALED_INCOME_COLS,
        *CONCEALED_SHARE_COLS,
        CORPORATE_MARKDOWN_GAINS_COLUMN,
    ),
    "prices": tuple(
        bs.wealth_tax_rate_key(name)
        for name in (*bs.MARKETABLE_ASSET_COLS, *bs.CLOSELY_HELD_ASSET_COLS)
    ),
    "discrete_prices": (),
    "draws": (),
}


def avoided_and_concealed_shares(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, float],
) -> dict[str, Array]:
    """The four shares the module is built from, per household.

    `avoided_marketable` and `avoided_closely_held` are the share of each
    class of wealth that stops being reported, counting both concealment and
    valuation gaming. `concealed_marketable` and `concealed_closely_held` are
    the concealment part, which also removes reported income and estate value.
    """
    marketable_rate_change = bs.rate_on_base(
        records, prices_reform, bs.MARKETABLE_ASSET_COLS
    ) - bs.rate_on_base(records, prices_baseline, bs.MARKETABLE_ASSET_COLS)
    closely_held_rate_change = bs.rate_on_base(
        records, prices_reform, bs.CLOSELY_HELD_ASSET_COLS
    ) - bs.rate_on_base(records, prices_baseline, bs.CLOSELY_HELD_ASSET_COLS)
    reported_marketable_share = jnp.exp(
        marketable_rate_change * params["public_elasticity"]
    )
    reported_closely_held_share = jnp.exp(
        closely_held_rate_change * params["closely_held_elasticity"]
    )
    avoided_marketable = 1.0 - reported_marketable_share
    avoided_closely_held = 1.0 - reported_closely_held_share
    return {
        "reported_share_marketable": reported_marketable_share,
        "reported_share_closely_held": reported_closely_held_share,
        "avoided_marketable": avoided_marketable,
        "avoided_closely_held": avoided_closely_held,
        "concealed_marketable": params["concealed_share_marketable"]
        * avoided_marketable,
        "concealed_closely_held": params["concealed_share_closely_held"]
        * avoided_closely_held,
    }


def hidden_ledger(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, float],
) -> dict[str, Array]:
    """Every dollar this module moves, on both sides of the ledger.

    The module writes reported amounts onto the frame. This function says, in
    dollars per household, where the difference went, so that a test can check
    that reported plus hidden equals the amount before the module ran.

    The wealth entries are:

        marketable wealth reported + concealed + gamed = marketable wealth
        closely held reported + concealed + gamed + pulled out by evasion
            = closely held wealth

    "Gamed" is wealth still reported but assessed lower; it leaves the wealth
    tax base and nothing else. "Pulled out by evasion" is the assets behind
    income the evasion module hid.

    The income entries are one pair per reported income column:

        `reported_<column>` + `hidden_<column>` = the column before this ran

    with the hidden amount equal to the column times the concealed share of
    the class of wealth that produced it, and zero for a household that had no
    positive income of that kind.
    """
    shares = avoided_and_concealed_shares(
        records, prices_reform, prices_baseline, params
    )
    marketable = bs.total(records, bs.MARKETABLE_ASSET_COLS)
    closely_held = bs.total(records, bs.CLOSELY_HELD_ASSET_COLS)
    debts = bs.total(records, bs.DEBT_COLS)
    evaded = bs.evaded_closely_held_income_share(records)

    reported_marketable = marketable * shares["reported_share_marketable"]
    # The evasion link applies to what is still reported after the wealth
    # response, so the same dollar is not removed twice.
    still_reported_closely_held = closely_held * shares["reported_share_closely_held"]
    pulled_out_by_evasion = still_reported_closely_held * evaded
    reported_closely_held = still_reported_closely_held - pulled_out_by_evasion

    concealed_marketable = marketable * shares["concealed_marketable"]
    concealed_closely_held = closely_held * shares["concealed_closely_held"]

    ledger: dict[str, Array] = {
        "marketable_wealth": marketable,
        "closely_held_wealth": closely_held,
        "debts": debts,
        "reported_marketable_wealth": reported_marketable,
        "reported_closely_held_wealth": reported_closely_held,
        "concealed_marketable_wealth": concealed_marketable,
        "concealed_closely_held_wealth": concealed_closely_held,
        # The response net of concealment: wealth still reported at a lower
        # assessed value.
        "gamed_marketable_wealth": marketable * shares["avoided_marketable"]
        - concealed_marketable,
        "gamed_closely_held_wealth": closely_held * shares["avoided_closely_held"]
        - concealed_closely_held,
        "closely_held_wealth_pulled_out_by_evasion": pulled_out_by_evasion,
        # Reported net worth is the frame's net worth less what this module
        # hid, so a household with nothing hidden keeps exactly the net worth
        # it arrived with.
        "reported_net_worth": bs.column(records, "net_worth")
        - (marketable - reported_marketable)
        - (closely_held - reported_closely_held)
        if "net_worth" in records
        else reported_marketable + reported_closely_held - debts,
        "evaded_income_share": evaded,
        CONCEALED_SHARE_MARKETABLE_COL: shares["concealed_marketable"],
        CONCEALED_SHARE_CLOSELY_HELD_COL: shares["concealed_closely_held"],
    }

    for gate_cols, scaled_cols, kind in CONCEALED_INCOME_GROUPS:
        concealed_share = (
            shares["concealed_marketable"]
            if kind == MARKETABLE
            else shares["concealed_closely_held"]
        )
        reports_this_income = bs.total(records, gate_cols) > 0.0
        for name in scaled_cols:
            before = bs.column(records, name)
            hidden = jnp.where(reports_this_income, before * concealed_share, 0.0)
            ledger[f"hidden_{name}"] = hidden
            # A missing entry goes back out missing. The concealed amount is
            # computed on a reading that treats a gap as zero, and that
            # reading does not reach the frame.
            ledger[f"reported_{name}"] = bs.keep_what_was_missing(
                bs.as_given(records, name), before - hidden
            )
    return ledger


def wealth_avoidance(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, float],
    draws: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """Reduce reported wealth, and the income and estate concealed wealth hides.

    Returns the changed columns:

      * `net_worth`, the base the wealth tax is charged on. Marketable wealth
        shrinks by its elasticity, closely held wealth by its own and again by
        the share of its income the evasion module hid, and debts are
        subtracted in full.
      * the reported income columns concealed wealth stops producing, with the
        two earners' self-employment shares following their business.
      * `wealth_c_pub` and `wealth_c_priv`, the concealed share of each class,
        which the estate avoidance module reads.

    The `value.*` asset columns are unchanged. `draws` is unused.
    """
    del draws
    ledger = hidden_ledger(records, prices_reform, prices_baseline, params)
    changed: dict[str, Array] = {"net_worth": ledger["reported_net_worth"]}
    for name in _SCALED_INCOME_COLS:
        changed[name] = ledger[f"reported_{name}"]
    for name in CONCEALED_SHARE_COLS:
        changed[name] = ledger[name]
    # The part of long-term gains the corporate markdown removed is part of
    # kg_lt, so the same share of it is concealed. Only the distribution table
    # reads this column.
    if CORPORATE_MARKDOWN_GAINS_COLUMN in records:
        reports_gains = bs.column(records, "kg_lt") > 0.0
        markdown_change = bs.column(records, CORPORATE_MARKDOWN_GAINS_COLUMN)
        changed[CORPORATE_MARKDOWN_GAINS_COLUMN] = jnp.where(
            reports_gains,
            markdown_change * (1.0 - ledger[CONCEALED_SHARE_MARKETABLE_COL]),
            markdown_change,
        )
    return changed
