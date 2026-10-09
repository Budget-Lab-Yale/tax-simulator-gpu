"""Repealing head-of-household status has to reach the law, not only the frame.

Repealing the head-of-household filing status works entirely by moving those
filers onto the single-filer column of every filing-status-dependent
parameter: the single standard deduction, the single rate brackets, the single
phase-out thresholds. So the recode has to happen before the year's law is
read for each household.

The standard deduction stands in for every filing-status parameter here: it is
the largest one, and if the law is read for the wrong status it is wrong.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.scenario import law_by_year, provenance
from taxmodel.years import one_year
from tests.conftest import CONFIG_ROOT, INDEX_TABLE

# The reform repeals the status from 2026, so 2025 is the year nothing happens
# and 2026 is the first year it does.
YEARS = (2025, 2026)
REFORM = "tests/repeal_hoh_2026"

SINGLE = 1
HEAD_OF_HOUSEHOLD = 4


@pytest.fixture(scope="module")
def bound_laws():
    """The reform's and baseline's law, bound over both years."""
    reform = provenance.load_scenario(YEARS, CONFIG_ROOT, INDEX_TABLE, REFORM)
    baseline = provenance.load_scenario(YEARS, CONFIG_ROOT, INDEX_TABLE, None)
    reform_laws = law_by_year.bind_annual_laws(
        reform.values,
        reform.years,
        reform.law_table,
        YEARS,
        baseline.values,
        baseline.years,
    )
    baseline_laws = law_by_year.bind_annual_laws(
        baseline.values, baseline.years, baseline.law_table, YEARS
    )
    return reform_laws, baseline_laws


def standard_deduction(law, filing_status):
    """The standard deduction each household gets, with the law read as the runner reads it."""
    read_for = one_year.filing_status_the_law_is_read_for(law, filing_status)
    return np.asarray(law.for_records(read_for).calculator.std_ded.value), np.asarray(
        read_for
    )


def test_repealed_head_of_household_filers_get_the_single_deduction(bound_laws):
    reform_laws, _ = bound_laws
    filing_status = jnp.asarray([SINGLE, HEAD_OF_HOUSEHOLD])

    before, status_before = standard_deduction(reform_laws.by_year[0], filing_status)
    assert list(status_before) == [SINGLE, HEAD_OF_HOUSEHOLD]
    assert before[1] > before[0], "2025 still has head-of-household status"

    after, status_after = standard_deduction(reform_laws.by_year[1], filing_status)
    assert list(status_after) == [SINGLE, SINGLE], (
        "from 2026 the reform makes head-of-household filers single filers"
    )
    assert after[1] == after[0], (
        "a repealed head of household takes the single standard deduction"
    )


def test_baseline_law_still_reads_them_as_heads_of_household(bound_laws):
    """The baseline world the reform is priced against keeps the status."""
    _, baseline_laws = bound_laws
    filing_status = jnp.asarray([SINGLE, HEAD_OF_HOUSEHOLD])
    for year_index in range(len(YEARS)):
        amounts, status = standard_deduction(
            baseline_laws.by_year[year_index], filing_status
        )
        assert list(status) == [SINGLE, HEAD_OF_HOUSEHOLD]
        assert amounts[1] > amounts[0]
