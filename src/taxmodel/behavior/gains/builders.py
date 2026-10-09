"""The parts of the capital gains model that run inside the compiled year.

JAX traces everything here: the two appliers (the frozen one moves a
decedent's gain and the deemed tax, the behavioral one also sells the solved
amount), the pricers that measure the age cells' marginal tax rates and
average them into cells for the host solve, and the runtime tables passed to
the compiled program as data. The helpers that read the averaged rates back
out of an execution's exports are here too.

The settings these pieces read are in `behavior/gains/settings.py`. The
host-side solves that use the exported rates are in `years/gains_solve.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass

from taxmodel.behavior import loader
from taxmodel.behavior.gains.settings import (
    CONVERSION_LEG_RATE_NAMES,
    DEFAULT_SETTINGS_FILE,
    JOINT_FILING_STATUS_ROW,
    ApplierSettings,
    GainsLaw,
    _filing_status_row,
    exported_column,
    read_applier_settings,
)
from taxmodel.behavior.kg import (
    assemble,
    prepass,
    regime,
    stepper,
    stock_inputs,
)
from taxmodel.behavior.kg.stock_inputs import KG_N_AGES
from taxmodel.taxes import individual_income_tax

Array = jnp.ndarray

# ---------------------------------------------------------------------------
# Reading the age cells' tax rates back out of an execution's exports
# ---------------------------------------------------------------------------


def cell_rate_on_gains(rate_exports: Mapping[str, Array], leg: str) -> Array:
    """One leg's cell rate on realized gains, every year, [year, age cell]."""
    sums = {
        f"{leg}_{field}": exported_column(rate_exports, f"{leg}_{field}")
        for field in prepass.MARGINAL_RATE_CELL_SUM_FIELDS
    }
    return prepass.MarginalRateCellSums.from_named(leg, sums).average()


# ---------------------------------------------------------------------------
# Runtime tables passed to the compiled program
# ---------------------------------------------------------------------------

# Year-varying gains inputs, passed to the scan as data. Scenarios with the
# same array shapes share one compiled program.


@dataclass(frozen=True)
class GainsTables:
    """The capital-gains model's runtime inputs, one row per run year.

    Stacked on the year axis on the way into the scan, which slices each
    year's row out. The three cell tables are [year, age cell, field] over
    ``behavior/kg/stepper.CELL_TABLE_FIELDS``. ``frozen_cell_table`` feeds the
    first-order world and leaves out the corporate markdown of share prices;
    ``mechanical_frozen_cell_table`` feeds the mechanical worlds and includes
    it.
    The other four are the year's law on gains at death, in the shapes the
    applier reads: ``death_regime`` and ``realized_at_death_share`` are
    [year, asset class] in ``DEEMED_ASSET_CLASSES`` order, and the two
    exclusions are [year, filing status] in the compiled law's own order.

    The first execution of a gains scenario runs with the frozen tables solved
    and the behavioral tables at the do-nothing values
    (:func:`identity_cell_tables`); the second runs the same compiled program
    with both solved.
    """

    behavioral_cell_table: Array
    frozen_cell_table: Array
    mechanical_frozen_cell_table: Array
    death_regime: Array
    realized_at_death_share: Array
    section_121_exclusion: Array
    death_gain_exclusion: Array
    # The conversions the behavioral solve priced, [year, household, leg] over
    # `behavior/kg/stock_inputs.CONVERSION_LEGS`, read by the in-year
    # conversion module. A scenario whose behavior leg runs no conversion
    # module carries a table with zero households.
    conversions: Array
    # The base year's price level over this year's, one number per run year.
    # The applier deflates each decedent's gross estate by it before running
    # the charitable bequest logits, which were fit in the base year's dollars.
    charity_deflator: Array


register_dataclass(
    GainsTables,
    data_fields=[field.name for field in fields(GainsTables)],
    meta_fields=[],
)


def identity_cell_tables(n_years: int) -> Array:
    """Cell tables under which the applier changes no household.

    Every field is zero except the two ratios: the rate factor, which
    multiplies each household's realized gains and so must be one, and the
    deemed factor, which scales the gain a decedent is deemed to realize and
    is one when the stock is exactly its baseline. The deemed gain itself is
    still zero wherever the year's law deems nothing, because the
    realized-at-death share is zero there.
    """
    stack = np.zeros(
        (n_years, KG_N_AGES, len(stepper.CELL_TABLE_FIELDS)), dtype=np.float64
    )
    stack[:, :, stepper.CELL_TABLE_FIELD_INDEX["rate_factor"]] = 1.0
    stack[:, :, stepper.CELL_TABLE_FIELD_INDEX["deemed_factor"]] = 1.0
    return jnp.asarray(stack)


def runtime_tables(
    law: GainsLaw,
    frozen_cell_tables: Array,
    behavioral_cell_tables: Array | None = None,
    conversions: Array | None = None,
    charity_deflator: Array | None = None,
    mechanical_frozen_cell_tables: Array | None = None,
) -> GainsTables:
    """One scenario's runtime tables, from its law and its solved cell tables.

    ``behavioral_cell_tables`` left out means the behavioral solve has not
    happened yet (the first execution), and the conventional world runs on the
    do-nothing tables. The realized-at-death share is worked out from the
    year's regime codes here, on the host.

    ``charity_deflator`` is the base year's price level over each run year's,
    which the applier deflates a decedent's gross estate by before running the
    charitable bequest logits. Left out, it is one for every year, which means
    no deflation and only suits a caller that has no macro data.

    ``conversions`` is the behavioral solve's [year, household, leg] table for
    a scenario whose leg runs a conversion module. Left out, the table has
    zero households, and the two executions must agree on which: the caller
    passes a zero-filled table of the full shape for the first execution of a
    converting scenario, so both executions present the compiled program the
    same shapes.

    ``mechanical_frozen_cell_tables`` is the frozen pass with the corporate
    markdown subtracted from the stock. Left out, the mechanical worlds read
    ``frozen_cell_tables``, which is correct for a scenario with no corporate
    markdown.
    """
    frozen_cell_tables = jnp.asarray(frozen_cell_tables)
    mechanical_frozen_cell_tables = (
        frozen_cell_tables
        if mechanical_frozen_cell_tables is None
        else jnp.asarray(mechanical_frozen_cell_tables)
    )
    n_years = frozen_cell_tables.shape[0]
    behavioral = (
        identity_cell_tables(n_years)
        if behavioral_cell_tables is None
        else jnp.asarray(behavioral_cell_tables)
    )
    if conversions is None:
        conversions = jnp.zeros(
            (n_years, 0, len(stock_inputs.CONVERSION_LEGS))
        )
    realized_at_death = jnp.asarray(
        [
            regime.realized_at_death_share_by_class(
                tuple(int(code) for code in row)
            )
            for row in law.death_regime_by_class
        ],
        dtype=jnp.float64,
    )
    return GainsTables(
        behavioral_cell_table=behavioral,
        frozen_cell_table=frozen_cell_tables,
        mechanical_frozen_cell_table=mechanical_frozen_cell_tables,
        death_regime=jnp.asarray(law.death_regime_by_class, dtype=jnp.int32),
        realized_at_death_share=realized_at_death,
        section_121_exclusion=jnp.asarray(
            law.section_121_exclusion, dtype=jnp.float64
        ),
        death_gain_exclusion=jnp.asarray(
            law.death_gain_exclusion, dtype=jnp.float64
        ),
        conversions=jnp.asarray(conversions, dtype=jnp.float64),
        charity_deflator=(
            jnp.ones(n_years, dtype=jnp.float64)
            if charity_deflator is None
            else jnp.asarray(charity_deflator, dtype=jnp.float64)
        ),
    )


def charity_deflator_for_years(
    macro_root: Path | str,
    years: Sequence[int],
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
) -> np.ndarray:
    """The base year's price level over each run year's, for the charity logits.

    The charitable bequest logits were fit on estates measured in the dollars
    of one particular year, so an estate in any other year is brought back to
    that year's dollars before the coefficients mean anything. The price
    series is CPI-U.

    The pre-pass uses the same series and the same base year to build
    its cell averages. It is read here as well so that the applier can work the
    probability out for a single household.
    """
    base_year = prepass.read_charitable_bequest_settings(settings_file).base_year
    return assemble.price_deflator_to_base_year(macro_root, years, base_year)


# ---------------------------------------------------------------------------
# Behavior builders for the frozen and the behavioral slot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GainsChainSpec:
    """Everything needed to compose a gains scenario's behavior step, read once.

    Importing the module functions and reading their manifests is host work
    done when the scenario is prepared; what remains inside the traced year
    loop is only putting the year's numbers into the applier's parameters and
    calling the composed functions. ``module_entries`` is the rest of the
    pinned chain, every behavioral margin in the order they run, carrying each
    module's default values. A scenario's own numbers are passed to the
    compiled program as data (``behavior/parameters.py``), so two scenarios
    differing only in an elasticity share one compiled program and one chain
    spec.
    """

    applier_function: Any
    applier_manifest: Mapping[str, Any]
    applier_settings: ApplierSettings
    module_entries: tuple[Mapping[str, Any], ...]
    # The two logits that give a decedent their own chance of leaving
    # appreciated property to charity. Read here on the host and passed to the
    # applier as parameters, because the applier works the probability out for
    # each household rather than reading the age cell's average.
    charitable_settings: prepass.CharitableBequestSettings


def read_gains_chain_spec(
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
) -> GainsChainSpec:
    """Import the applier and every behavioral margin's module, on the host."""
    applier_function, applier_manifest, _defaults = loader.gains_applier_entry()
    return GainsChainSpec(
        applier_function=applier_function,
        applier_manifest=applier_manifest,
        applier_settings=read_applier_settings(settings_file),
        module_entries=tuple(loader.chain_entries()[1:]),
        charitable_settings=prepass.read_charitable_bequest_settings(settings_file),
    )


def _applier_params(
    spec: GainsChainSpec,
    tables: GainsTables,
    cell_table: Array,
    records: Mapping[str, Array],
) -> dict[str, Any]:
    """The applier's parameters for one year, from the year's runtime tables."""
    row = _filing_status_row(records["filing_status"])
    return {
        "allocation_weight_on_holdings": (
            spec.applier_settings.allocation_weight_on_holdings
        ),
        "pass_through_valuation_factor": (
            spec.applier_settings.pass_through_valuation_factor
        ),
        "cell_table": cell_table,
        "realized_at_death_share": tables.realized_at_death_share,
        "regime_code": tables.death_regime,
        "section_121_exclusion": tables.section_121_exclusion[row],
        "death_gain_exclusion": tables.death_gain_exclusion[row],
        "death_gain_exclusion_married": tables.death_gain_exclusion[
            JOINT_FILING_STATUS_ROW
        ],
        # This year's price level against the year the charitable bequest
        # logits were fit in, and the logits themselves.
        "charitable_deflator": tables.charity_deflator,
        "charitable_settings": spec.charitable_settings,
    }


def frozen_gains_builder_for(spec: GainsChainSpec):
    """The frozen slot's behavior step, which runs only the gains applier.

    The frozen pass is law arithmetic. The treatment of gains at death changes
    while selling behavior is held at what baseline law produced. The leg's
    behavior modules do not run here, because the worlds this builder feeds
    hold behavior fixed.

    ``with_corporate_markdown`` picks the table: false for the first-order
    world, which reads the frozen table without the corporate markdown of
    share prices, and true for the mechanical worlds, which read the table
    with it.
    """

    def build(
        records,
        gains_year,
        prices_reform,
        prices_baseline,
        behavior_parameters=None,
        *,
        with_corporate_markdown: bool,
    ):
        del prices_reform, prices_baseline, behavior_parameters
        cell_table = (
            gains_year.mechanical_frozen_cell_table
            if with_corporate_markdown
            else gains_year.frozen_cell_table
        )
        chain = loader.compose(
            [
                {
                    "function": spec.applier_function,
                    "manifest": spec.applier_manifest,
                    "params": _applier_params(
                        spec, gains_year, cell_table, records
                    ),
                }
            ],
            tuple(records),
            (),
        )
        return chain(records, {}, {})

    return build


def behavioral_gains_builder_for(spec: GainsChainSpec):
    """The conventional world's behavior step, the applier followed by the
    leg's other modules.

    The applier reads the behavioral cell table; the leg's other modules run
    after it in the loader's pinned family order, reading what it wrote.
    Composition happens inside the traced function, where the frame's own set
    of columns exists to check the manifests against. Every import and file
    read already happened on the host when the spec was built.
    """

    def build(
        records,
        gains_year,
        prices_reform,
        prices_baseline,
        behavior_parameters=None,
    ):
        entries = [
            {
                "function": spec.applier_function,
                "manifest": spec.applier_manifest,
                "params": _applier_params(
                    spec, gains_year, gains_year.behavioral_cell_table, records
                ),
            },
        ]
        # The conversion module reads the year's priced conversions out of the
        # runtime tables, the same way the applier reads its cell table. Every
        # gains scenario carries a table of the full shape, all zeros where
        # the solve priced no conversions, so converting and non-converting
        # scenarios share one compiled program.
        for entry in spec.module_entries:
            if entry["manifest"].get("family") == "conversion":
                entries.append(
                    {
                        **entry,
                        "params": {
                            **entry["params"],
                            "conversions": gains_year.conversions,
                        },
                    }
                )
            else:
                entries.append(entry)
        chain = loader.compose(entries, tuple(records), tuple(prices_reform))
        # This scenario's behavioral numbers arrive as data. The frame's
        # per-household random draws are frame columns, so the chain reads them
        # itself: the tips module decides which untipped workers start
        # receiving tips by comparing a draw against a probability.
        return chain(records, prices_reform, prices_baseline, behavior_parameters)

    return build


# ---------------------------------------------------------------------------
# The rates the program exports
# ---------------------------------------------------------------------------

# The three legs of the selling choice that the year loop prices, named
# for the law and the frame each was measured on.
#
# The selling choice compares the reform leg against the baseline leg. Both
# are measured on the mechanical frame, so the comparison isolates the change
# in law. The mechanical frame carries the frozen gains pass, so under a
# carryover reform the reform rate moves because heirs are carrying inherited
# gains. The law-only rate is the same reform rate on the same frame before
# that injection, and it feeds the retiming overlay alone.
#
# Every rate is the derivative of net income tax with respect to realized
# long-term gain, like every other rate the model stores.
REFORM_LEG = "reform_on_mechanical"
BASELINE_ON_MECHANICAL_LEG = "baseline_on_mechanical"
LAW_ONLY_LEG = "reform_law_only"

# Every leg's rate on realized long-term gains is averaged into age cells.
_EXPORTED_LEGS = (REFORM_LEG, BASELINE_ON_MECHANICAL_LEG, LAW_ONLY_LEG)


@dataclass(frozen=True)
class GainsRatePricers:
    """The code that measures and exports the rates the gains model reads.

    The reform and baseline legs are read off the price dictionaries the year
    already computed for the mechanical frame. ``rate_on_long_term_gains``
    prices a frame under a law and returns the marginal rate on a dollar of
    realized long-term gain; it measures the law-only leg, whose frame the
    year does not otherwise price.
    """

    rate_on_long_term_gains: Any

    def year_exports(
        self,
        *,
        cell_weights,
        mechanical_frame,
        mechanical_prices_reform,
        mechanical_prices_baseline,
        pre_injection_frame,
        reform_law,
        mechanical_results=None,
    ) -> dict[str, Array]:
        """The cell-level sums one year hands back to the host-side solve.

        The three legs' rates are measured on the households and added into
        their age cells here, inside the compiled program, so what crosses
        back to the host is a handful of sums per cell rather than a column
        per household.

        ``cell_weights`` is this year's slice of the dollars each household
        brings to its cell (``behavior/kg/prepass.CellWeights``), which the
        driver builds once from the population data.

        ``mechanical_results`` is the mechanical world's tax results, read for
        the taxable-income column the conversion pool tests against the
        top-bracket threshold; the pool's other quantities come off the frame.
        """
        if cell_weights is None:
            raise ValueError(
                "the rates a gains scenario measures are averaged into age "
                "cells inside the compiled program, so the run needs the "
                "cell weights the averaging reads; the driver builds them "
                "from the population block"
            )
        # Three rates per household, never leaving the compiled program: each
        # of the three legs' marginal rate on a dollar of realized long-term
        # gain.
        rates = {
            f"{REFORM_LEG}_mtr_kg_lt": mechanical_prices_reform["mtr_kg_lt"],
            f"{BASELINE_ON_MECHANICAL_LEG}_mtr_kg_lt": (
                mechanical_prices_baseline["mtr_kg_lt"]
            ),
            f"{LAW_ONLY_LEG}_mtr_kg_lt": self.rate_on_long_term_gains(
                pre_injection_frame, reform_law
            ),
        }
        exports = average_rates_into_cells(cell_weights, rates)
        # The conversion channel relabels one household's pay as its own
        # equity, so its rates and the frame columns below leave the device per
        # household and per year rather than as cell averages. The five
        # conversion-leg rates are read off the price dictionaries the year
        # already computed.
        for name in CONVERSION_LEG_RATE_NAMES:
            exports[f"reform_on_mechanical_mtr_{name}"] = mechanical_prices_reform[
                f"mtr_{name}"
            ]
            exports[f"baseline_on_mechanical_mtr_{name}"] = (
                mechanical_prices_baseline[f"mtr_{name}"]
            )
        # The household's own rate on one more dollar of realized long-term
        # gain, which the conversion wedge sets against the leg rates.
        exports["reform_on_mechanical_mtr_kg_lt"] = mechanical_prices_reform["mtr_kg_lt"]
        exports["baseline_on_mechanical_mtr_kg_lt"] = (
            mechanical_prices_baseline["mtr_kg_lt"]
        )
        # What the conversion pool reads off the mechanical frame, per
        # household per year: which filing status the household files under and
        # what each leg holds on the frame the rates were measured on. The
        # taxable income the pool tests against the top-bracket threshold is
        # a result column, added just below.
        for name in ("wages_base1", "wages_base2", "part_active", "scorp_active",
                     "sole_prop1", "sole_prop2", "filing_status"):
            exports[f"mechanical_{name}"] = jnp.asarray(mechanical_frame[name])
        if mechanical_results is not None:
            exports["mechanical_txbl_inc"] = jnp.asarray(
                mechanical_results["txbl_inc"]
            )
        return exports


def average_rates_into_cells(
    cell_weights: prepass.CellWeights,
    household_rates: Mapping[str, Array],
) -> dict[str, Array]:
    """Add one year's measured rates into the cells, leg by leg.

    Every rate the selling choice reads is a cell average over the dollars
    it applies to, so the compiled program forms the sums here rather than
    handing a column per household back to the host. The host divides them
    (:func:`solve_behavioral_tables_for_group`).
    """

    sums: dict[str, Array] = {}
    for leg in _EXPORTED_LEGS:
        sums.update(
            prepass.marginal_rate_cell_sums(
                cell_weights, household_rates[f"{leg}_mtr_kg_lt"]
            ).named(leg)
        )
    return sums


def _calculator_law(law: object) -> object:
    """The bound calculator law, off whichever wrapper the year loop carries."""
    return getattr(law, "calculator", law)


def make_gains_rate_pricers(double_run: bool) -> GainsRatePricers:
    """Build the pricer one group of gains scenarios shares.

    ``double_run`` is the charitable-deduction structural switch, settled for
    the whole projection when the calculators were built; it is passed as a
    plain boolean because these functions run inside the traced program.
    """

    def rate_on_long_term_gains(records, law):
        gradient = individual_income_tax.derivative(
            records,
            _calculator_law(law),
            outcome=individual_income_tax.net_income_tax,
            input_names=("kg_lt",),
            double_run=double_run,
        )
        return gradient["kg_lt"]

    return GainsRatePricers(rate_on_long_term_gains=rate_on_long_term_gains)
