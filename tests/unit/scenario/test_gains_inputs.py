"""The capital-gains model's household inputs, read off the population block."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.frame import population, prepare, steps_jax
from taxmodel.scenario import gains_inputs
from tests.unit.frame.conftest import make_frame

YEAR = 2027
# The estate bridge's cap on expected deaths in one cluster of replicated
# donor records, chosen so the pair below is capped.
CLUSTER_CAP = 300.0


def test_the_chance_of_dying_carries_the_donor_cluster_cap():
    """The host inputs cap a donor cluster exactly as the compiled frame steps do.

    Records 1 and 2 are a replicated donor pair: the same 50 million dollar
    estate, each standing for 100,000 households with a 5 percent chance of
    dying, 10,000 expected deaths together. With a cap of 300 each chance of
    dying is scaled by 300 / 10,000, to 0.0015. Record 3 has a smaller estate
    of its own and keeps its 2 percent. Record 4 is a joint return, where the
    chance of dying is the chance both spouses die.
    """
    frame = make_frame(
        [
            {"id": 1, "age1": 79, "q_death1": 0.05, "weight": 100_000.0,
             "value.equities": 50_000_000.0},
            {"id": 2, "age1": 79, "q_death1": 0.05, "weight": 100_000.0,
             "value.equities": 50_000_000.0},
            {"id": 3, "age1": 70, "q_death1": 0.02, "weight": 1_000.0,
             "value.equities": 2_000_000.0},
            {"id": 4, "filing_status": 2, "age1": 75, "age2": 73,
             "q_death1": 0.04, "q_death2": 0.03, "weight": 1_000.0,
             "value.equities": 1_000_000.0},
        ]
    )
    universe = prepare.build_universe({YEAR: frame["id"].to_numpy()})
    source = population.population_from_frames({YEAR: frame}, universe)

    [records] = gains_inputs.host_records_by_year(source, [YEAR], CLUSTER_CAP)

    assert records["m_household"] == pytest.approx([0.0015, 0.0015, 0.02, 0.0012])

    arrays = {
        name: jnp.asarray(frame[name].to_numpy(dtype=np.float64))
        for name in frame.columns
        if name != "id" and np.issubdtype(frame[name].dtype, np.number)
    }
    valid = jnp.ones(len(frame), dtype=bool)
    compiled = steps_jax.add_capital_gains_record_attributes(
        steps_jax.calc_estate_mortality(
            arrays,
            jnp.asarray(CLUSTER_CAP),
            jnp.asarray(gains_inputs.DONOR_CLUSTER_GROSS_ESTATE_FLOOR),
            valid,
        ),
        valid,
    )
    assert np.asarray(compiled["m_household"]) == pytest.approx(
        records["m_household"]
    )
