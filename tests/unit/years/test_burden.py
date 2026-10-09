"""The wealth forcing formula, with every term visible in round dollars."""

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.years.burden import burden


def tax_results():
    baseline = {
        "liab_iit_net": jnp.array([10.0, 20.0, 30.0]),
        "liab_pr": jnp.array([5.0, 5.0, 5.0]),
        "liab_wealth": jnp.array([0.0, 0.0, 0.0]),
        # This employer-side detail is deliberately irrelevant to burden.
        "liab_pr_er": jnp.array([1_000.0, 1_000.0, 1_000.0]),
    }
    reform = {
        "liab_iit_net": jnp.array([20.0, 18.0, 35.0]),
        "liab_pr": jnp.array([5.0, 6.0, 4.0]),
        "liab_deemed": jnp.array([2.0, 0.0, 1.0]),
        "liab_wealth": jnp.array([1.0, 0.0, 2.0]),
        "liab_pr_er": jnp.array([9_000.0, 9_000.0, 9_000.0]),
    }
    return reform, baseline


def test_burden_matches_the_weighted_r_formula():
    reform, baseline = tax_results()
    actual = burden(
        reform,
        baseline,
        corp_dY_exog=jnp.array([-3.0, 1.0, 0.0]),
        pr_dY_exog=jnp.array([-2.0, 0.0, 3.0]),
        weights=jnp.array([1.0, 2.0, 0.5]),
    )

    np.testing.assert_allclose(actual, [14.0, -4.0, 1.0])


def test_employer_payroll_income_is_subtracted_once_not_twice():
    reform, baseline = tax_results()
    weights = jnp.array([1.0, 2.0, 0.5])
    payroll_income = jnp.array([-2.0, 0.0, 3.0])
    reference = burden(
        reform,
        baseline,
        corp_dY_exog=jnp.array([-3.0, 1.0, 0.0]),
        pr_dY_exog=payroll_income,
        weights=weights,
    )

    deliberately_double_subtracted = reference - weights * payroll_income
    np.testing.assert_allclose(reference, [14.0, -4.0, 1.0])
    assert not np.allclose(deliberately_double_subtracted, reference)


def test_omitting_corporate_income_fails_the_reference_value():
    reform, baseline = tax_results()
    reference = burden(
        reform,
        baseline,
        corp_dY_exog=jnp.array([-3.0, 1.0, 0.0]),
        pr_dY_exog=jnp.array([-2.0, 0.0, 3.0]),
        weights=jnp.array([1.0, 2.0, 0.5]),
    )
    without_corporate_income = burden(
        reform,
        baseline,
        corp_dY_exog=None,
        pr_dY_exog=jnp.array([-2.0, 0.0, 3.0]),
        weights=jnp.array([1.0, 2.0, 0.5]),
    )

    np.testing.assert_allclose(reference, [14.0, -4.0, 1.0])
    assert not np.allclose(without_corporate_income, reference)


def test_zero_reform_has_zero_burden_with_missing_optional_columns():
    results = {
        "liab_iit_net": jnp.array([10.0, -5.0]),
        "liab_pr": jnp.array([4.0, 2.0]),
        "liab_wealth": jnp.array([0.0, 0.0]),
    }

    actual = burden(results, results, None, None, jnp.array([1.0, 3.0]))
    np.testing.assert_allclose(actual, [0.0, 0.0])


def test_burden_is_a_jax_calculation():
    reform, baseline = tax_results()

    actual = jax.jit(burden)(
        reform,
        baseline,
        jnp.array([-3.0, 1.0, 0.0]),
        jnp.array([-2.0, 0.0, 3.0]),
        jnp.array([1.0, 2.0, 0.5]),
    )
    np.testing.assert_allclose(actual, [14.0, -4.0, 1.0])
