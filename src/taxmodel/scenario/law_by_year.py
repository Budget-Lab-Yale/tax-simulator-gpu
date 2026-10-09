"""Binding every projection year of a scenario's law.

A scenario's law arrives as the law compiler's output: one value per parameter
per law year, plus the wide law table with one row per year and filing status.
:class:`AnnualLaw` is one year's law. :class:`BoundLawSeries` is every
projection year of it, stacked on a leading year axis for the compiled
program and also held year by year for host-side reads. The frame builder
that reads the bound law each year is in `frame/build_frame.py`.

An annual law carries three things. The bound calculator law is what the tax
calculators read. The wide law table's columns for the year are read by the
frame steps; `law/batch.FRAME_LAW_COLUMNS_READ_BY_NAME` lists them. Baseline
payroll law is what the employer-payroll wage adjustment is measured against.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.tree_util import register_dataclass

from taxmodel.frame import steps_jax
from taxmodel.taxes import law_for_calculators
from taxmodel.taxes.calc.payroll import PayrollLaw

Array = jnp.ndarray


@dataclass(frozen=True)
class CorporateYearFactors:
    """The corporate incidence path's per-year numbers, as runtime law data.

    One value per year in the stacked law, one scalar in a year's slice: the
    four proportional flow changes, the equity markdown, the payout cut, and
    the factor that scales gains from corporate share repurchases.

    They are passed with each year's law as data, so every scenario runs the
    same compiled program. A scenario with no corporate reform gets the
    identity values. The pieces of the
    corporate step that do not vary with the scenario, which balance-sheet
    columns are exposed and by how much and the corporate share of gains, come
    from the economy leg instead and are closed over by the frame builder, which
    every scenario of a group shares because a group shares one economy leg.

    ``years/corporate/channel.py`` builds the real values where the scenario's law
    changes the corporate rate, and every other scenario gets :func:`identity`.
    """

    dividend_factor: Array
    interest_factor: Array
    rent_factor: Array
    pass_through_factor: Array
    markdown_rate: Array
    flow_factor: Array
    gain_quantity_factor: Array

    @staticmethod
    def identity(n_years: int | None = None) -> CorporateYearFactors:
        """The values under which the corporate step changes no number."""
        one = jnp.ones(()) if n_years is None else jnp.ones(n_years)
        zero = jnp.zeros(()) if n_years is None else jnp.zeros(n_years)
        return CorporateYearFactors(
            dividend_factor=one,
            interest_factor=one,
            rent_factor=one,
            pass_through_factor=one,
            markdown_rate=zero,
            flow_factor=zero,
            gain_quantity_factor=one,
        )

    @staticmethod
    def from_arrays(factors: Mapping[str, object]) -> CorporateYearFactors:
        """The stacked factors from `years/corporate/channel.factor_arrays_for_years`."""
        return CorporateYearFactors(
            **{name: jnp.asarray(value) for name, value in factors.items()}
        )


register_dataclass(
    CorporateYearFactors,
    data_fields=[field.name for field in dataclasses.fields(CorporateYearFactors)],
    meta_fields=[],
)


# ---------------------------------------------------------------------------
# One year's law, in both the forms the model needs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnualLaw:
    """One year's law: what the calculators read, and what the frame steps join.

    ``calculator`` is the bound law object every calculator takes.
    ``frame_columns`` is the same year of the wide law table, one entry per law
    column, four rows in filing-status order. ``baseline_payroll`` is baseline
    payroll law, which the employer-payroll wage adjustment measures against;
    on a baseline law object it is that same law.

    Everything here is held per filing status, not per household, and
    ``for_records`` is what gives each household its own status's values. The
    frame columns are the exception: ``steps_jax.join_law`` does that lookup
    itself as it joins them onto records, so they travel as four rows the whole
    way.
    """

    calculator: law_for_calculators.YearLaw
    frame_columns: dict[str, Array]
    baseline_payroll: PayrollLaw
    # The corporate incidence path's per-year numbers, law-like runtime data,
    # at the identity values for every scenario without a corporate reform.
    # ``corporate`` is built from the corporate revenue change with the tax
    # base allowed to erode, and feeds the conventional world.
    # ``corporate_mechanical`` is built from the change with the base held at
    # its baseline level, and feeds the two mechanical worlds.
    corporate: CorporateYearFactors = dataclass_field(
        default_factory=CorporateYearFactors.identity
    )
    corporate_mechanical: CorporateYearFactors = dataclass_field(
        default_factory=CorporateYearFactors.identity
    )

    def for_records(self, filing_status: Array) -> AnnualLaw:
        """The calculator law and baseline payroll law, read for each household."""
        rows = law_for_calculators.filing_status_rows(filing_status)
        return AnnualLaw(
            calculator=self.calculator.for_records(filing_status),
            frame_columns=self.frame_columns,
            baseline_payroll=jax.tree.map(
                lambda table: jnp.asarray(table)[rows], self.baseline_payroll
            ),
            corporate=self.corporate,
            corporate_mechanical=self.corporate_mechanical,
        )


register_dataclass(
    AnnualLaw,
    data_fields=[field.name for field in dataclasses.fields(AnnualLaw)],
    meta_fields=[],
)


def annual_frame_law_columns_for_years(
    law_table: pd.DataFrame, years: Sequence[int]
) -> dict[str, np.ndarray]:
    """Every requested year of the wide law table, years on a leading axis.

    Each column comes over as its four rows in filing-status order for every
    year, shaped [year, filing status]; ``steps_jax.join_law`` gathers from
    that array once the year is picked. A column whose four values are the
    same is four copies of one number, and every law presents the same shapes
    for that reason. Each column is read out of the table once for the whole
    projection rather than once per year, which is only possible because a
    year's row does not depend on any other year's: the law compiler has
    already worked out each year's value.
    """
    wanted = [int(year) for year in years]
    expected = list(range(1, steps_jax.N_FILING_STATUSES + 1))
    table_years = law_table["year"].to_numpy()
    filing_statuses = law_table["filing_status"].to_numpy()

    # Which rows of the table this projection reads, in year order and within a
    # year in filing-status order, which is the order `steps_jax.join_law`
    # gathers with.
    reading_order: list[int] = []
    for year in wanted:
        in_year = np.flatnonzero(table_years == year)
        in_year = in_year[np.argsort(filing_statuses[in_year], kind="stable")]
        found = [int(status) for status in filing_statuses[in_year]]
        if found != expected:
            raise ValueError(
                f"the {year} law table covers filing statuses {found}, expected "
                f"{expected}"
            )
        reading_order.extend(int(row) for row in in_year)
    rows = np.asarray(reading_order)

    shape = (len(wanted), steps_jax.N_FILING_STATUSES)
    return {
        name: law_table[name].to_numpy(dtype=np.float64)[rows].reshape(shape)
        for name in law_table.columns
        if name not in ("year", "filing_status")
    }


@dataclass(frozen=True)
class BoundLawSeries:
    """One law bound for every year of a projection, in the two forms a run needs.

    ``stacked`` holds each parameter with the projection years on a leading
    axis; the year loop scans over it and it is the only form that reaches
    the compiled program. ``by_year`` is the same law year by year, for
    host-side reads of a single year's law. The year-by-year laws hold NumPy arrays rather than JAX ones,
    because nothing sends them to the device.
    """

    stacked: AnnualLaw
    by_year: tuple[AnnualLaw, ...]


def bind_annual_laws(
    values: Mapping[str, np.ndarray],
    law_years: Sequence[int],
    law_table: pd.DataFrame,
    years: Sequence[int],
    baseline_values: Mapping[str, np.ndarray] | None = None,
    baseline_law_years: Sequence[int] | None = None,
) -> BoundLawSeries:
    """Bind every year of a projection of one law.

    Each parameter is read across the year axis in a single array operation.

    ``baseline_values`` is baseline law, which the employer-payroll wage change
    is measured against. Left out, this law is its own baseline.

    Nothing here depends on the population: every law parameter comes over as
    its four filing-status values, and the lookup that gives each household its
    own status's value happens inside the compiled program.
    """
    wanted = [int(year) for year in years]
    parameters = law_for_calculators.resolve_years(values, law_years, wanted)
    frame_columns = annual_frame_law_columns_for_years(law_table, wanted)
    if baseline_values is None or baseline_values is values:
        baseline_parameters = parameters
    else:
        if baseline_law_years is None:
            raise ValueError(
                "binding against a separate baseline law needs that law's year axis"
            )
        baseline_parameters = law_for_calculators.resolve_years(
            baseline_values, baseline_law_years, wanted
        )
    # Payroll law is the only part of baseline law an annual law carries, and
    # `payroll` is the name `taxes/law_for_calculators.py` files it under.
    baseline_payroll = baseline_parameters["payroll"]

    # The whole projection, as JAX arrays with the years on a leading axis.
    # This is the object the year loop scans and the only one that reaches the
    # device.
    stacked = AnnualLaw(
        calculator=law_for_calculators.assemble_year_law(
            {
                group: {
                    attribute: jnp.asarray(array)
                    for attribute, array in table.items()
                }
                for group, table in parameters.items()
            },
            jnp.asarray(np.asarray(wanted, dtype=np.float64)),
        ),
        frame_columns={
            name: jnp.asarray(column) for name, column in frame_columns.items()
        },
        baseline_payroll=PayrollLaw(
            **{
                attribute: jnp.asarray(array)
                for attribute, array in baseline_payroll.items()
            }
        ),
        # The stacked law enters the year scan, so every leaf needs the years
        # on a leading axis (the identity factors included).
        corporate=CorporateYearFactors.identity(len(wanted)),
        corporate_mechanical=CorporateYearFactors.identity(len(wanted)),
    )

    # The same law one year at a time. Each array is a view into the arrays
    # above rather than a copy, so building all thirty years costs almost
    # nothing, and they stay NumPy because nothing sends them to the device.
    def law_for_one_year(index: int, year: int) -> AnnualLaw:
        return AnnualLaw(
            calculator=law_for_calculators.assemble_year_law(
                {
                    group: {
                        attribute: array[index] for attribute, array in table.items()
                    }
                    for group, table in parameters.items()
                },
                np.asarray(year, dtype=np.float64),
            ),
            frame_columns={
                name: column[index] for name, column in frame_columns.items()
            },
            baseline_payroll=PayrollLaw(
                **{
                    attribute: array[index]
                    for attribute, array in baseline_payroll.items()
                }
            ),
        )

    return BoundLawSeries(
        stacked=stacked,
        by_year=tuple(
            law_for_one_year(index, year) for index, year in enumerate(wanted)
        ),
    )


def with_corporate_factors(
    series: BoundLawSeries,
    conventional_factors: Mapping[str, object],
    mechanical_factors: Mapping[str, object],
) -> BoundLawSeries:
    """A bound law series with a scenario's real corporate factors attached.

    Each set of factors is `years/corporate/channel.factor_arrays_for_years`'s
    answer: one array per factor, one value per simulation year.
    ``conventional_factors`` come from the paths built with the corporate tax
    base eroding, and ``mechanical_factors`` from the paths built with it
    fixed. The stacked law gets the arrays whole; each year's law gets its own
    scalar slice.
    """
    conventional = CorporateYearFactors.from_arrays(conventional_factors)
    mechanical = CorporateYearFactors.from_arrays(mechanical_factors)

    def year_slice(stacked_factors, index):
        return jax.tree.map(lambda value: value[index], stacked_factors)

    return BoundLawSeries(
        stacked=dataclasses.replace(
            series.stacked, corporate=conventional, corporate_mechanical=mechanical
        ),
        by_year=tuple(
            dataclasses.replace(
                law,
                corporate=year_slice(conventional, index),
                corporate_mechanical=year_slice(mechanical, index),
            )
            for index, law in enumerate(series.by_year)
        ),
    )


def for_the_mechanical_worlds(law: Any) -> Any:
    """The same law with the fixed-base corporate factors in ``corporate``.

    The frame builder reads ``law.corporate``, so the two mechanical worlds'
    frame is built from this law. A law without mechanical factors, such as a
    unit test's stand-in, is returned unchanged.
    """
    mechanical = getattr(law, "corporate_mechanical", None)
    if mechanical is None:
        return law
    return dataclasses.replace(law, corporate=mechanical)


# ---------------------------------------------------------------------------
# The CPI level series the capital-gain basis ratio reads
# ---------------------------------------------------------------------------


def cpi_level_series(indexes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The spliced price level by year: CPI-U, then chained CPI-U from 2017.

    This is the same splice tax parameter indexation uses. It is built once on
    the host from the run's growth-rate table and handed to the compiled frame
    steps as two arrays. Only ratios of the level are ever read, so the level
    the series starts from does not matter.
    """
    early = indexes[
        (indexes["series"] == "cpi")
        & (indexes["year"] < steps_jax.CHAINED_CPI_FIRST_YEAR)
    ]
    late = indexes[
        (indexes["series"] == "chained_cpi")
        & (indexes["year"] >= steps_jax.CHAINED_CPI_FIRST_YEAR)
    ]
    blended = pd.concat([early, late]).sort_values("year")
    growth = blended["growth"].fillna(0.0).to_numpy()
    return (
        blended["year"].to_numpy(dtype=np.float64),
        np.cumprod(1.0 + growth).astype(np.float64),
    )
