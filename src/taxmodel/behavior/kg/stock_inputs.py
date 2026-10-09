"""Inputs to the stock recurrence, and the two operators it steps with.

Two bundles of inputs are defined here:

``BaselineCells``
    The age cells as Tax-Data describes them under baseline law, rebuilt for
    every year: the stock of unrealized gain, realizations, turnover,
    mortality, and the share of a decedent's gains that goes to charity.
    These are levels, re-read each year: the recurrence carries only the
    *deviation* from them.

``ConversionPool``
    Per household, per year: the income legs that can be relabeled as capital
    gain, the marginal rate on each leg under both laws, and the fields that
    decide whether the household is in the pool at all.

The recurrence also reads the schedule bundle, which is defined where it is
produced, in ``schedule.py``.  Its arrays are age by year; the arrays here are
year by age, because the recurrence scans over years.
``stepper.schedule_by_year()`` does the transpose.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass

Array = jnp.ndarray

# The age grid the stock is carried on: one cell per age from 18 to 80, with 80
# pooling everyone 80 and older.
KG_AGE_MIN = 18
KG_AGE_MAX = 80
KG_N_AGES = KG_AGE_MAX - KG_AGE_MIN + 1


@dataclass(frozen=True)
class BaselineCells:
    """Baseline-law age cells for every year, each array shaped [year, age].

    ``mortality_gain_weighted`` is the numerator of the holdings-weighted
    mortality rate: the sum over the cell's households of mortality times
    holdings.  Dividing by the cell's stock gives the mortality rate a dollar
    of unrealized gain faces, and the flow of gains at death follows that rate.
    ``mortality_headcount`` is the plain per-person rate, used only where a
    cell holds nothing.

    ``turnover`` is realizations over the stock of gains, the quantity the data
    measures directly.  Selling is pro-rata out of a well-mixed portfolio, so
    it is also the share of the cell's asset value sold in the year.
    """

    stock: Array
    realizations: Array
    turnover: Array
    mortality_headcount: Array
    mortality_gain_weighted: Array
    charitable_bequest_share: Array


register_dataclass(
    BaselineCells,
    data_fields=[field.name for field in fields(BaselineCells)],
    meta_fields=[],
)


# The five income legs a household can relabel as capital gain, in the order
# the leg arrays use.  Wages come out of the two earners' wage boxes; the three
# pass-through legs are the active (not passive) business income the owner
# could pay themselves out of instead.
CONVERSION_LEGS = (
    "wages1",
    "wages2",
    "partnership_active",
    "s_corporation_active",
    "sole_proprietorship",
)

# Which legs count as active business income.  A household with none of it
# cannot convert: an employee with no business cannot decide to take their pay
# as equity.
CONVERSION_LEG_IS_ACTIVE_BUSINESS = (False, False, True, True, True)


def no_household_conversions(n_years: int) -> Array:
    """An empty conversion table, for a program that works in age cells alone.

    One entry per year and no households in it.  The stock recurrence takes the
    conversions household by household so that the in-year module can clamp and
    apply them; a caller with no households passes this and keeps the real
    table itself.
    """

    return jnp.zeros((n_years, 0, len(CONVERSION_LEGS)))


@dataclass(frozen=True)
class ConversionPool:
    """Per-household conversion inputs, arrays shaped [year, household].

    ``leg_amount``, ``marginal_rate_scenario`` and ``marginal_rate_baseline``
    carry a third axis over :data:`CONVERSION_LEGS`.  The two marginal-rate
    arrays are the mechanical world's prices on its own frame, once under reform
    law and once under baseline law (the same frame priced under two laws, so
    their difference is the rate change the household faces).

    ``gains_rate_scenario`` and ``gains_rate_baseline`` are the same
    household's marginal rate on one more dollar of realized long-term gain,
    on the same frame under the same two laws.

    ``age_index`` is the household's position on the 18-to-80 age grid.  A
    household outside that range, or with no age, gets a negative index and
    converts nothing, because there is no age cell in the stock of gain to
    receive the converted dollars.

    ``top_bracket_threshold`` is the taxable income at which the household
    reaches the top ordinary bracket; conversion is modeled only above it.
    """

    weight: Array
    age_index: Array
    taxable_income: Array
    top_bracket_threshold: Array
    leg_amount: Array
    marginal_rate_scenario: Array
    marginal_rate_baseline: Array
    gains_rate_scenario: Array
    gains_rate_baseline: Array


register_dataclass(
    ConversionPool,
    data_fields=[field.name for field in fields(ConversionPool)],
    meta_fields=[],
)


def aging_operator(n_ages: int = KG_N_AGES) -> Array:
    """The operator that moves each age cell to the next one.

    Applied as ``aging.T @ x``, so the cell for age 40 receives what age 39
    held.  The top-coded cell points at itself, so age 80 keeps its own stock
    and takes in age 79's.
    """

    operator = np.zeros((n_ages, n_ages))
    for i in range(n_ages - 1):
        operator[i, i + 1] = 1.0
    operator[n_ages - 1, n_ages - 1] = 1.0
    return jnp.asarray(operator)


def normalized_heir_row(heir_age_distribution: Array) -> np.ndarray:
    """The distribution of heir ages, checked and rescaled to sum to one."""

    row = np.asarray(heir_age_distribution, dtype=np.float64)
    total = row.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "heir age distribution must be positive and sum to more than zero"
        )
    if row.min() < 0:
        raise ValueError("heir age distribution must not be negative")
    return row / total


def heir_operator(heir_age_distribution: Array) -> Array:
    """The operator routing a decedent's gains to the ages of their heirs.

    Every row is the same: the observed distribution of heir ages, weighted by
    dollars inherited.  That assumes heir age does not depend on the decedent's
    age, and it matches the total flow to each heir age exactly.  Applied as
    ``heirs.T @ x``.
    """

    row = jnp.asarray(normalized_heir_row(heir_age_distribution))
    return jnp.broadcast_to(row, (row.shape[0], row.shape[0]))
