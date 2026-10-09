"""A batched solve of four scenarios against the same four solved one at a time.

Four reforms with different treatments of gains at death, different exclusion
amounts and different rates on gains, solved both ways on the same made-up
population: all four in one compiled program, as a run does, and each on its
own as a group of one (``whole_model.run_one_scenario``). The two run the same
functions on the same numbers, so the cell tables have to agree to within the
tolerance below.

The frozen pass and the behavioral pass are both checked, because they are two
different solvers reached through the same program.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.behavior.kg import assemble, many_scenarios, prepass
from taxmodel.behavior.kg.inputs import Calibration

from . import whole_model

YEARS = tuple(range(2026, 2031))
N_HOUSEHOLDS = 40

# Fixed response parameters, so that a refit does not change what these tests
# assert.
CALIBRATION = Calibration(
    elasticity=jnp.asarray(0.72),
    timeable_share=jnp.asarray(0.2),
    appreciation=0.0677,
)

# How far apart the two paths may be, as a share of the size of the numbers.
# Batching changes the order a cell's households are summed in, and bisection
# can amplify that last-bit difference to about one part in a billion.
RELATIVE_TOLERANCE = 1e-8

# Today's top rate on a long-term gain with the investment income tax.
BASELINE_RATE_ON_GAINS = 0.238

STEP_UP_EVERYTHING = (0, 0, 0, 0, 0)
CARRYOVER_EVERYTHING = (1, 1, 1, 1, 1)
DEEMED_EVERYTHING = (2, 2, 2, 2, 2)


def households() -> dict[str, np.ndarray]:
    """Forty filers spread over the age grid, half of them married.

    Two filing statuses, because the death-gain exclusions are stated by filing
    status and the batched path looks them up inside the compiled program
    instead of on the host — a lookup that has to land on the same row.
    """

    age = np.linspace(25.0, 79.0, N_HOUSEHOLDS)
    value = np.linspace(50_000.0, 5_000_000.0, N_HOUSEHOLDS)
    joint = np.arange(N_HOUSEHOLDS) % 2 == 0
    frame: dict[str, np.ndarray] = {
        "weight": np.full(N_HOUSEHOLDS, 1_000.0),
        "age1": age,
        "age2": np.where(joint, age - 2.0, np.nan),
        # Filing status 2 is married filing jointly, 1 is single.
        "filing_status": np.where(joint, 2.0, 1.0),
        "m_household": 0.001 + 0.0004 * (age - 25.0),
        # A single filer has some chance of being a surviving spouse at death,
        # so the widowed exclusion amount matters.
        "p_widow": np.where(joint, 0.0, 0.3),
        "estate_economic_gross": 2.0 * value,
        "kg_lt": 0.025 * value,
    }
    for name in prepass.DEEMED_ASSET_CLASSES:
        frame[f"value.{name}"] = np.zeros(N_HOUSEHOLDS)
        frame[f"basis.{name}"] = np.zeros(N_HOUSEHOLDS)
    frame["value.equities"] = value
    frame["basis.equities"] = 0.5 * value
    # A primary home too, so the section 121 exclusion applies.
    frame["value.primary_home"] = 0.4 * value
    frame["basis.primary_home"] = 0.1 * value
    return frame


class Reform:
    """One made-up scenario's law on gains at death, and its rate on gains.

    ``corporate_exposure`` and ``equity_markdown`` are the corporate channel's
    two facts. Both are zero for most reforms, which is the case that has to
    keep working, and nonzero for one, which is the case the batched program
    has to carry as data.
    """

    def __init__(
        self,
        *,
        regime_codes,
        section_121: float,
        death_gain_exclusion: float,
        joint_death_gain_exclusion: float,
        rate_on_gains: float,
        corporate_exposure: dict[str, float] | None = None,
        equity_markdown: float = 0.0,
    ):
        self.regime_codes = regime_codes
        self.section_121 = section_121
        self.death_gain_exclusion = death_gain_exclusion
        self.joint_death_gain_exclusion = joint_death_gain_exclusion
        self.rate_on_gains = rate_on_gains
        self.corporate_exposure = corporate_exposure
        self.equity_markdown = equity_markdown


REFORMS = (
    Reform(
        regime_codes=STEP_UP_EVERYTHING,
        section_121=250_000.0,
        death_gain_exclusion=0.0,
        joint_death_gain_exclusion=0.0,
        rate_on_gains=0.238,
    ),
    Reform(
        regime_codes=CARRYOVER_EVERYTHING,
        section_121=250_000.0,
        death_gain_exclusion=1_000_000.0,
        joint_death_gain_exclusion=2_000_000.0,
        rate_on_gains=0.338,
    ),
    Reform(
        regime_codes=DEEMED_EVERYTHING,
        section_121=500_000.0,
        death_gain_exclusion=5_000_000.0,
        joint_death_gain_exclusion=10_000_000.0,
        rate_on_gains=0.396,
    ),
    # A corporate rate rise: the corporate channel marks equity down and
    # debits the marked-down value from the cell's stock of gain.
    Reform(
        regime_codes=CARRYOVER_EVERYTHING,
        section_121=250_000.0,
        death_gain_exclusion=0.0,
        joint_death_gain_exclusion=0.0,
        rate_on_gains=0.288,
        corporate_exposure={"value.equities": 0.6, "value.pass_throughs": 0.35},
        equity_markdown=0.04,
    ),
)


def law_of(reform: Reform) -> many_scenarios.ScenarioLaw:
    """The reform's law on gains at death, in the form the program reads."""
    return whole_model.scenario_law(
        len(YEARS),
        regime_codes=reform.regime_codes,
        section_121=reform.section_121,
        death_gain_exclusion=reform.death_gain_exclusion,
        joint_death_gain_exclusion=reform.joint_death_gain_exclusion,
        corporate_exposure=reform.corporate_exposure,
        equity_markdown=(
            0.0 if reform.corporate_exposure is None else reform.equity_markdown
        ),
    )


def prices_of(reform: Reform) -> dict[str, jnp.ndarray]:
    """The reform's cell prices, the same in every cell and year."""
    return whole_model.cell_prices(
        prepass.KG_N_AGES,
        len(YEARS),
        baseline_rate_on_gains=BASELINE_RATE_ON_GAINS,
        reform_rate_on_gains=reform.rate_on_gains,
    )


def solve_one_at_a_time(macro_root: Path, reform: Reform, *, frozen: bool):
    """One scenario on its own, for the batched group to match."""
    return whole_model.run_one_scenario(
        [households() for _ in YEARS],
        YEARS,
        macro_root=macro_root,
        charity=prepass.read_charitable_bequest_settings(),
        calibration=CALIBRATION,
        law=law_of(reform),
        prices=prices_of(reform),
        frozen=frozen,
    )


def solve_all_at_once(macro_root: Path, *, frozen: bool):
    """The batched path: every reform in one compiled program."""
    records = [households() for _ in YEARS]
    shared = many_scenarios.shared_solve_inputs(
        records,
        YEARS,
        macro_root=macro_root,
        charity=prepass.read_charitable_bequest_settings(),
    )
    laws = jax.tree.map(
        lambda *leaves: jnp.stack(leaves), *[law_of(reform) for reform in REFORMS]
    )
    cells = many_scenarios.scenario_cells(shared, laws)

    no_prices = dict.fromkeys(
        whole_model.PRICE_NAMES, jnp.zeros((prepass.KG_N_AGES, len(YEARS)))
    )
    stacked_prices = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[
            whole_model.scenario_prices(
                no_prices if frozen else prices_of(reform), CALIBRATION
            )
            for reform in REFORMS
        ],
    )
    solved = many_scenarios.solve_schedules_for_scenarios(
        shared,
        laws,
        cells,
        stacked_prices,
        frozen=frozen,
        shaping=many_scenarios.ProgramShapingSettings.from_calibration(CALIBRATION),
    )
    no_conversions = jnp.zeros((len(REFORMS), len(YEARS), prepass.KG_N_AGES))
    # The corporate channel's markdown is a stock: the level of the price
    # decline that year, against the corporate equity standing behind the
    # cell's gains. A frozen pass marks nothing down.
    corporate_gain_debit = (
        no_conversions
        if frozen
        else laws.equity_markdown[:, :, None] * cells.corporate_exposed_value
    )
    return many_scenarios.step_stock_for_scenarios(
        shared, solved.bundle, no_conversions, corporate_gain_debit, frozen=frozen
    ).cell_tables


def worst_disagreement(batched: jnp.ndarray, one_at_a_time: jnp.ndarray):
    """The cell-table field the two paths disagree on most, and by how much.

    Measured as a share of the largest number in that field, so a field holding
    millions of dollars and one holding a rate between zero and one are judged
    on the same scale. A field that is essentially zero, such as the extra
    realizations of a reform identical to baseline law, is judged against a
    millionth of the table's largest dollar amount instead, so that
    floating-point noise on a zero does not count as a disagreement.
    """
    from taxmodel.behavior.kg.stepper import CELL_TABLE_FIELDS

    largest_dollar_amount = float(jnp.max(jnp.abs(one_at_a_time)))
    zero_field_scale = 1e-6 * largest_dollar_amount
    worst_field, worst_share, worst_absolute = CELL_TABLE_FIELDS[0], 0.0, 0.0
    for position, field in enumerate(CELL_TABLE_FIELDS):
        mine = batched[:, :, position]
        theirs = one_at_a_time[:, :, position]
        absolute = float(jnp.max(jnp.abs(mine - theirs)))
        scale = max(float(jnp.max(jnp.abs(theirs))), zero_field_scale)
        share = absolute / scale if scale > 0 else absolute
        if share > worst_share:
            worst_field, worst_share, worst_absolute = field, share, absolute
    return worst_field, worst_share, worst_absolute


@pytest.mark.parametrize("frozen", [True, False])
def test_the_batched_solve_gives_the_same_cell_tables(macro_root, frozen):
    """Four reforms in one program against each on its own, cell for cell."""

    batched = solve_all_at_once(macro_root, frozen=frozen)

    for position, _reform in enumerate(REFORMS):
        one_at_a_time = solve_one_at_a_time(macro_root, _reform, frozen=frozen)
        field, share, absolute = worst_disagreement(
            batched[position], one_at_a_time.cell_tables
        )
        assert share < RELATIVE_TOLERANCE, (
            f"reform {position} disagrees most on {field}: {absolute} in "
            f"absolute terms, {share} as a share of the field's own size"
        )


def test_the_population_cells_are_the_ones_the_pre_pass_builds(macro_root):
    """The nine fields no reform changes, split out and batched over years."""

    records = [households() for _ in YEARS]
    shared = many_scenarios.shared_solve_inputs(
        records,
        YEARS,
        macro_root=macro_root,
        charity=prepass.read_charitable_bequest_settings(),
    )
    deflator = assemble.price_deflator_to_base_year(
        macro_root, YEARS, prepass.read_charitable_bequest_settings().base_year
    )
    for year, (year_records, year_deflator) in enumerate(zip(records, deflator)):
        expected = prepass.population_age_cells(
            year_records,
            deflator_to_base_year=year_deflator,
            charity=prepass.read_charitable_bequest_settings(),
        )
        assert float(
            jnp.max(jnp.abs(shared.population_cells.stock[year] - expected.stock))
        ) < 1e-6
        assert float(
            jnp.max(
                jnp.abs(
                    shared.population_cells.charitable_bequest_share[year]
                    - expected.charitable_bequest_share
                )
            )
        ) < 1e-6

