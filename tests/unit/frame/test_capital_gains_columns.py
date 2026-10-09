"""The three household facts the capital-gains model reads off the frame.

Each case is a household written out in full, so the expected stock of
unrealized gain, chance of dying and chance of being a surviving spouse can
be worked out by hand from the numbers in the test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from taxmodel.behavior.kg import life_table
from taxmodel.frame import steps_jax, widowhood

from .conftest import make_frame

# The 2022 SCF shares in config/calibrations/estate/widowhood.yaml, quoted
# here so a test that expects one of them says which one it is.
SHARE_WIDOWED_FEMALE_70_TO_79 = 0.467756
SHARE_WIDOWED_MALE_80_AND_OVER = 0.651497
SHARE_WIDOWED_FEMALE_18_TO_49 = 0.012476


@pytest.fixture
def households() -> pd.DataFrame:
    """Four households covering the cases the applier can meet."""
    return make_frame(
        [
            # A joint return where the husband is 80 and the wife 78. Both
            # have to die for the couple's gains to be settled.
            {
                "id": 1,
                "filing_status": 2,
                "age1": 80,
                "age2": 78,
                "male1": 1,
                "q_death1": 0.06,
                "q_death2": 0.04,
                "value.equities": 1_000_000.0,
                "basis.equities": 400_000.0,
                "value.primary_home": 500_000.0,
                "basis.primary_home": 500_000.0,
            },
            # A single woman of 72, who might be a widow.
            {
                "id": 2,
                "filing_status": 1,
                "age1": 72,
                "male1": 0,
                "q_death1": 0.02,
                "value.pass_throughs": 250_000.0,
                "basis.pass_throughs": 100_000.0,
            },
            # A household holding one asset class under water and another
            # above it. The loss must not net against the gain.
            {
                "id": 3,
                "filing_status": 1,
                "age1": 45,
                "male1": 0,
                "q_death1": 0.001,
                "value.other_home": 300_000.0,
                "basis.other_home": 800_000.0,
                "value.re_fund": 200_000.0,
                "basis.re_fund": 50_000.0,
            },
            # A divorced head of household, and a class whose basis Tax-Data
            # did not record.
            {
                "id": 4,
                "filing_status": 4,
                "age1": 68,
                "male1": 1,
                "divorce_year": 2005,
                "q_death1": 0.01,
                "value.equities": 900_000.0,
                "basis.equities": np.nan,
            },
        ]
    )


def as_arrays(frame: pd.DataFrame) -> dict[str, jnp.ndarray]:
    """The numeric columns of a small test frame, as the compiled steps read them."""
    return {
        name: jnp.asarray(frame[name].to_numpy(dtype=np.float64))
        for name in frame.columns
        if pd.api.types.is_numeric_dtype(frame[name])
    }


def with_estate_mortality(
    arrays: dict[str, jnp.ndarray], cluster_cap: float = np.inf
) -> dict[str, jnp.ndarray]:
    """The frame after the estate mortality step, which the gains step follows.

    With the default cap of infinity no cluster of replicated donor records is
    scaled down.
    """
    valid = jnp.ones(len(arrays["id"]), dtype=bool)
    return steps_jax.calc_estate_mortality(
        arrays, jnp.asarray(cluster_cap), jnp.asarray(0.0), valid
    )


def gains_attributes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three household facts, keyed by the record's id."""
    out = steps_jax.add_capital_gains_record_attributes(
        with_estate_mortality(as_arrays(frame)), jnp.ones(len(frame), dtype=bool)
    )
    ids = frame["id"].to_numpy()
    return {
        name: {int(record): float(value) for record, value in zip(ids, np.asarray(out[name]))}
        for name in ("unrealized_gain_stock", "m_household", "p_widow")
    }


def test_the_stock_of_unrealized_gain_adds_up_the_classes_that_show_a_gain(households):
    stock = gains_attributes(households)["unrealized_gain_stock"]

    # 600,000 of appreciation on the equities, none on the home, which is
    # held at exactly its purchase price.
    assert stock[1] == 600_000.0
    assert stock[2] == 150_000.0
    # The 500,000 loss on the second home leaves the 150,000 gain on the real
    # estate fund alone.
    assert stock[3] == 150_000.0
    # No basis on record means the class is unusable, not free of basis.
    assert stock[4] == 0.0


def test_a_joint_return_dies_only_when_both_spouses_do(households):
    mortality = gains_attributes(households)["m_household"]

    assert mortality[1] == pytest.approx(0.06 * 0.04)
    assert mortality[2] == 0.02
    assert mortality[4] == 0.01


def test_the_gains_model_applies_the_estate_tax_cap_on_donor_clusters():
    """A replicated donor pair dies at the same capped rate in both models.

    Two records with the same gross estate to the dollar are a replicated
    donor pair. Each stands for 100,000 households with a 5 percent chance of
    dying, so together they carry 10,000 expected deaths. With a cap of 300
    both models scale each record's chance of dying by 300 / 10,000, to
    0.0015.
    """
    pair = make_frame(
        [
            {
                "id": 5,
                "filing_status": 1,
                "age1": 79,
                "q_death1": 0.05,
                "weight": 100_000.0,
                "value.equities": 50_000_000.0,
            },
            {
                "id": 6,
                "filing_status": 1,
                "age1": 79,
                "q_death1": 0.05,
                "weight": 100_000.0,
                "value.equities": 50_000_000.0,
            },
        ]
    )
    valid = jnp.ones(len(pair), dtype=bool)
    after_estate_step = with_estate_mortality(as_arrays(pair), cluster_cap=300.0)
    gains = steps_jax.add_capital_gains_record_attributes(after_estate_step, valid)

    assert np.asarray(after_estate_step["estate_m"]) == pytest.approx([0.0015] * 2)
    assert np.asarray(gains["m_household"]) == pytest.approx([0.0015] * 2)


def test_only_an_unmarried_filer_with_no_divorce_on_record_can_be_widowed(households):
    widowed = gains_attributes(households)["p_widow"]

    # A joint return has a living spouse by definition.
    assert widowed[1] == 0.0
    assert widowed[2] == pytest.approx(SHARE_WIDOWED_FEMALE_70_TO_79)
    assert widowed[3] == pytest.approx(SHARE_WIDOWED_FEMALE_18_TO_49)
    # Tax-Data records this filer as divorced, so he is not a widower.
    assert widowed[4] == 0.0


def test_the_widowhood_share_follows_age_and_sex():
    shares = widowhood.read_widowhood_shares()
    band = shares.for_age_and_sex(
        age=np.array([85.0, 85.0, 20.0]), is_male=np.array([True, False, False])
    )
    assert band[0] == pytest.approx(SHARE_WIDOWED_MALE_80_AND_OVER)
    assert band[1] > band[0]
    # Anyone below the first band takes the first band's share.
    assert band[2] == pytest.approx(SHARE_WIDOWED_FEMALE_18_TO_49)


def test_the_three_columns_survive_being_compiled(households):
    """The step runs inside the compiled year, so it has to trace."""
    frame = with_estate_mortality(as_arrays(households))
    compiled = jax.jit(steps_jax.add_capital_gains_record_attributes)(
        frame, jnp.ones(len(households), dtype=bool)
    )
    straight = steps_jax.add_capital_gains_record_attributes(
        frame, jnp.ones(len(households), dtype=bool)
    )

    for name in ("unrealized_gain_stock", "m_household", "p_widow"):
        np.testing.assert_allclose(
            np.asarray(compiled[name]), np.asarray(straight[name]), rtol=0, atol=0
        )


def test_mortality_past_the_top_code_averages_the_two_published_tables():
    """Checked by hand against the two Trustees Report CSVs.

    In 2030 the Alternative 2 chance of dying within the year at age 85 is
    0.088120 for a man and 0.068561 for a woman.
    """
    rates = life_table.mortality_above_the_top_code(years=[2029, 2030, 2031])
    age_85 = rates[85 - life_table.FIRST_AGE_ABOVE_TOP_CODE, 1]
    assert age_85 == pytest.approx(0.5 * 0.088120 + 0.5 * 0.068561)

    # The oldest age the tables report is 119, and mortality rises with age
    # all the way there.
    assert (np.diff(rates[:, 1]) > 0).all()
    assert rates.shape == (
        life_table.OLDEST_AGE - life_table.FIRST_AGE_ABOVE_TOP_CODE + 1,
        3,
    )
