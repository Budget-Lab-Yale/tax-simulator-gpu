"""Business income moving between corporate and pass-through form.

Pearce and Prisinzano (2018) estimate how the pass-through share of business
income responds to the gap between what business income pays in corporate form
and what it pays in pass-through form.  This module applies that response
household by household: when a reform widens the gap in pass-through's favor,
some of the household's business income moves to pass-through, and when it
narrows the gap the move goes the other way.

The corporate side of the comparison is two taxes stacked.  A dollar earned in
corporate form pays the corporate rate, and what is left is taxed again when it
reaches the shareholder:

    corporate price = corporate rate + (1 − corporate rate) × shareholder price

The shareholder price splits by what the corporation does with the money. The
share paid out this year is taxed at the shareholder's rate on capital gains,
on the simplification that dividends and gains face the same rate. The share
retained is taxed when the shareholder sells the appreciated stock, at the
capital-gains model's equivalent tax rate: the expected present value of tax
on a dollar of gain left unrealized, by age, year and policy. The gains
applier attaches those two rates, one per law, so this module runs after it.

    shareholder price = α × rate on gains + (1 − α) × equivalent tax rate

Where a household's rate on gains is zero, the retained share uses a flat
approximation: a quarter of the shareholder's rate on gains.

The pass-through side is the household's marginal rate on active partnership
income, taken to stand for every kind of pass-through income.

The amount that moves is the share shifted times the household's net
pass-through income, where that income is positive. It is spread across the
pass-through columns in proportion: every partnership, S corporation and sole
proprietorship column, losses and section 179 deductions included, is
multiplied by one plus the share shifted. The changes in those columns, with
losses and deductions counted negatively, add up to the amount shifted. The
partnership self-employment earnings columns are multiplied by the same
factor. Sole proprietorship income is itself the self-employment earnings base
and S corporation income carries no self-employment tax, so neither has a
separate column to move.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior import loader
from taxmodel.behavior.kg.applier import (
    EQUIVALENT_TAX_RATE_BASELINE_COLUMN,
    EQUIVALENT_TAX_RATE_SCENARIO_COLUMN,
)

# The two corporate rates, as the frame names them: the scored law's rate and
# baseline law's rate joined on by year.
from taxmodel.years.corporate.channel import (
    BASELINE_CORPORATE_RATE_COLUMN,
    CORPORATE_RATE_COLUMN,
)

Array = jnp.ndarray
Records = Mapping[str, Array]

# The change in corporate tax revenue this response implies, which the receipts
# code totals. Negative when income leaves the corporate base.
CORPORATE_TAX_CHANGE_COLUMN = "corp_tax_change"

# The household's pass-through income: every leg of partnership, S corporation
# and sole-proprietorship income that could change form, net of the losses and
# §179 deductions reported against them.
PASS_THROUGH_POSITIVE_COLUMNS = (
    "part_active",
    "part_passive",
    "scorp_active",
    "scorp_passive",
    "sole_prop1",
    "sole_prop2",
)
PASS_THROUGH_NEGATIVE_COLUMNS = (
    "part_active_loss",
    "part_passive_loss",
    "part_179",
    "scorp_active_loss",
    "scorp_passive_loss",
    "scorp_179",
)
# Each earner's net earnings from self-employment out of partnership income.
# They are part of the self-employment tax base, not of pass-through income.
PARTNERSHIP_SELF_EMPLOYMENT_COLUMNS = ("part_se1", "part_se2")

# The two marginal rates the response is measured from: the shareholder's rate
# on long-term capital gains, which prices the corporate side, and the rate on
# active partnership income, which prices the pass-through side.
GAINS_PRICE = "mtr_kg_lt"
PASS_THROUGH_PRICE = "mtr_part_active"

# A rate this small counts as zero. The retained-earnings fraction below is a
# fraction of the household's rate on gains, which is undefined at zero.
ZERO_RATE_TOLERANCE = 1e-9

DEFAULTS: dict[str, Any] = {
    # The semi-elasticity of the pass-through share of business income with
    # respect to the corporate-versus-pass-through tax differential. Pearce and
    # Prisinzano (2018), Table IV.B preferred results, divided by pass-through's
    # 0.6 share of business income to put it on a per-dollar basis.
    "semi_elasticity": 0.3788 / 0.6,
    # The share of corporate earnings paid out this year rather than retained.
    # A judgment value with no recorded derivation.
    "distributed_share": 0.45,
    # The fraction used for the tax on retained earnings where the household's
    # rate on gains is zero: a quarter of that rate. It has no tax effect
    # there.
    "retained_deferral_proxy": 0.25,
    # The statutory corporate rate under the law being scored and under
    # baseline law. A run reads both off the frame: the reform's rate is the
    # compiled law column, and the baseline's is joined on beside it before
    # the response runs (`years/worlds.py`). These entries are the fallback
    # for a caller running the module on a hand-built frame.
    CORPORATE_RATE_COLUMN: 0.0,
    BASELINE_CORPORATE_RATE_COLUMN: 0.0,
}

MANIFEST: dict[str, Any] = {
    "name": "entity_shifting_pearce_prisinzano",
    # The function in this file that is the response.
    "entry_point": "shift_entity_form",
    # The pass-through share moves by the semi-elasticity times the change in
    # the differential, so a zero semi-elasticity moves nothing.
    "zero_response": {"semi_elasticity": 0.0},
    "family": "entity_shifting",
    "reads": (
        CORPORATE_RATE_COLUMN,
        BASELINE_CORPORATE_RATE_COLUMN,
        *PASS_THROUGH_POSITIVE_COLUMNS,
        *PASS_THROUGH_NEGATIVE_COLUMNS,
        *PARTNERSHIP_SELF_EMPLOYMENT_COLUMNS,
        "kg_lt",
    ),
    "writes": (
        *PASS_THROUGH_POSITIVE_COLUMNS,
        *PASS_THROUGH_NEGATIVE_COLUMNS,
        *PARTNERSHIP_SELF_EMPLOYMENT_COLUMNS,
        "kg_lt",
        CORPORATE_TAX_CHANGE_COLUMN,
    ),
    # The two equivalent tax rates on a deferred dollar, which the
    # capital-gains applier attaches. In a scenario running no gains model
    # both read as zero, so the retained share is priced at zero.
    "optional_reads": (
        EQUIVALENT_TAX_RATE_BASELINE_COLUMN,
        EQUIVALENT_TAX_RATE_SCENARIO_COLUMN,
    ),
    "prices": (GAINS_PRICE, PASS_THROUGH_PRICE),
    "discrete_prices": (),
    "draws": (),
}


def shift_entity_form(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, Any],
    draws: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """Entity shifting, in the behavior module contract's signature.

    Returns the pass-through and capital-gain columns it moved, and the change
    in corporate tax revenue the move implies.
    """

    del draws

    kg_lt = jnp.asarray(records["kg_lt"], dtype=jnp.float64)
    template = kg_lt

    distributed = jnp.asarray(params["distributed_share"], dtype=jnp.float64)
    retained_proxy = jnp.asarray(params["retained_deferral_proxy"], dtype=jnp.float64)
    # A run reads both statutory rates off the frame; the params stand in on a
    # hand-built frame that carries neither.
    corporate_rate = jnp.asarray(
        loader.run_fact(records, params, CORPORATE_RATE_COLUMN, "entity shifting"),
        dtype=jnp.float64,
    )
    corporate_rate_baseline = jnp.asarray(
        loader.run_fact(
            records, params, BASELINE_CORPORATE_RATE_COLUMN, "entity shifting"
        ),
        dtype=jnp.float64,
    )

    gains_rate = bs.column(prices_reform, GAINS_PRICE, missing_like=template)
    gains_rate_baseline = bs.column(
        prices_baseline, GAINS_PRICE, missing_like=template
    )
    pass_through_rate = bs.column(
        prices_reform, PASS_THROUGH_PRICE, missing_like=template
    )
    pass_through_rate_baseline = bs.column(
        prices_baseline, PASS_THROUGH_PRICE, missing_like=template
    )

    # The retained share is priced at the capital-gains model's equivalent tax
    # rate. A scenario running no gains model reads both rates as zero.
    retained_rate = bs.column(
        records, EQUIVALENT_TAX_RATE_SCENARIO_COLUMN, missing_like=template
    )
    retained_rate_baseline = bs.column(
        records, EQUIVALENT_TAX_RATE_BASELINE_COLUMN, missing_like=template
    )
    shareholder_price = distributed * gains_rate + (1.0 - distributed) * retained_rate
    shareholder_price_baseline = (
        distributed * gains_rate_baseline
        + (1.0 - distributed) * retained_rate_baseline
    )
    # The offset to this year's realized gains below is expressed as a fraction
    # of this household's rate on gains. At a zero rate the flat approximation
    # stands in; it has no tax effect there.
    has_gains_rate = jnp.abs(gains_rate) > ZERO_RATE_TOLERANCE
    retained_fraction = jnp.where(
        has_gains_rate,
        jnp.clip(retained_rate / jnp.where(has_gains_rate, gains_rate, 1.0), 0.0, 1.0),
        retained_proxy,
    )

    distribution_offset_factor = distributed + (1.0 - distributed) * retained_fraction

    corporate_price = corporate_rate + (1.0 - corporate_rate) * shareholder_price
    corporate_price_baseline = (
        corporate_rate_baseline
        + (1.0 - corporate_rate_baseline) * shareholder_price_baseline
    )

    # The change in the tax advantage of corporate form. A reform that leaves
    # both sides' prices unchanged makes this zero, and every column below is
    # then returned unchanged.
    change_in_differential = (corporate_price - pass_through_rate) - (
        corporate_price_baseline - pass_through_rate_baseline
    )
    share_shifted = (
        jnp.asarray(params["semi_elasticity"], dtype=jnp.float64)
        * change_in_differential
    )

    business_income = jnp.zeros_like(template)
    for name in PASS_THROUGH_POSITIVE_COLUMNS:
        business_income = business_income + bs.column(
            records, name, missing_like=template
        )
    for name in PASS_THROUGH_NEGATIVE_COLUMNS:
        business_income = business_income - bs.column(
            records, name, missing_like=template
        )
    amount_shifted = jnp.maximum(0.0, business_income) * share_shifted

    # Every pass-through column, and the partnership self-employment earnings
    # columns, are multiplied by the same factor, so the dollar changes in the
    # pass-through columns add up to the amount shifted. A household whose net
    # pass-through income is not positive shifts nothing and keeps its columns.
    scale = jnp.where(business_income > 0, 1.0 + share_shifted, 1.0)

    # Shareholders receive the after-corporate-tax part of what shifted. The
    # reduction is taken out of capital gains so that dividends cannot be
    # driven negative, which makes the household-level gain figure an
    # accounting device.
    gains_offset = amount_shifted * (1.0 - corporate_rate) * distribution_offset_factor

    # A missing entry goes back out missing; the response is computed on a
    # reading that treats a gap as zero, and that reading does not reach the
    # frame.
    def as_given(name: str) -> Array:
        return bs.as_given(records, name, missing_like=template)

    # Scaling a blank gives no dollars, so every scaled column keeps its
    # blanks. kg_lt receives the gains offset, so a blank entry there is filled
    # where an offset was taken; otherwise the offset would vanish while the
    # corporate tax change stands.
    scaled = {
        name: bs.keep_what_was_missing(
            as_given(name), bs.column(records, name, missing_like=template) * scale
        )
        for name in (
            *PASS_THROUGH_POSITIVE_COLUMNS,
            *PASS_THROUGH_NEGATIVE_COLUMNS,
            *PARTNERSHIP_SELF_EMPLOYMENT_COLUMNS,
        )
    }
    return {
        **scaled,
        "kg_lt": bs.fill_when_written_into(
            as_given("kg_lt"), kg_lt - gains_offset, gains_offset
        ),
        CORPORATE_TAX_CHANGE_COLUMN: -amount_shifted * corporate_rate,
    }
