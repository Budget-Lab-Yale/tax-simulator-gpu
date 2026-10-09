"""Finished tax and outside-income change used by the wealth recurrence."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from taxmodel.reporting.estate_allocator_jax import DEEMED_GAINS_COLUMN

Array = jnp.ndarray
TaxResults = Mapping[str, Array]


def _required(results: TaxResults, name: str) -> Array:
    if name not in results:
        raise KeyError(f"tax results are missing {name}")
    return jnp.asarray(results[name])


def during_life_tax(results: TaxResults) -> Array:
    """Income, payroll, and wealth tax owed during the household's life.

    Tax on gains deemed realized at death is removed because the distribution
    table assigns estate-related tax through a separate inheritance measure.
    """
    income_tax = _required(results, "liab_iit_net")
    payroll_tax = _required(results, "liab_pr")
    wealth_tax = _required(results, "liab_wealth")
    deemed_tax = jnp.asarray(
        results.get(DEEMED_GAINS_COLUMN, jnp.zeros_like(income_tax))
    )
    return income_tax + payroll_tax - deemed_tax + wealth_tax


def burden(
    reform: TaxResults,
    baseline: TaxResults,
    corp_dY_exog: Array | None,
    pr_dY_exog: Array | None,
    weights: Array,
) -> Array:
    """Return each record's weighted finished tax and income change.

    Finished burden is the change in individual income tax net of refundable
    credits, plus payroll and wealth tax, less tax on deemed gains, less the
    corporate and employer-payroll changes in income from outside the tax
    system. Missing deemed tax and missing outside-income columns are zero.
    Employer payroll liability is already reflected in payroll liability and
    cash wages, so it is not added separately.
    """
    weights = jnp.asarray(weights)
    zero = jnp.zeros_like(weights, dtype=jnp.float64)
    corporate_income = zero if corp_dY_exog is None else jnp.asarray(corp_dY_exog)
    payroll_income = zero if pr_dY_exog is None else jnp.asarray(pr_dY_exog)
    return weights * (
        during_life_tax(reform)
        - during_life_tax(baseline)
        - corporate_income
        - payroll_income
    )
