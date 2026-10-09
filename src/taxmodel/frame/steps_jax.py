"""The annual frame-preparation steps, written as JAX array operations.

Each function accepts a dictionary of record arrays and returns a new
dictionary. The input dictionary is never changed. An annual law value shared
by filing statuses may be a scalar, while a filing-status-dependent value has
four rows and is gathered after the head-of-household recode.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from . import widowhood

Array = jnp.ndarray

# The household balance sheet: every asset column and every debt column the
# population carries. The estate calculator and the behavior modules import
# these from here.
ESTATE_ASSET_COLS: tuple[str, ...] = (
    "value.cash",
    "value.equities",
    "value.bonds",
    "value.dc",
    "value.db",
    "value.life_ins",
    "value.annuities",
    "value.trusts",
    "value.other_fin",
    "value.pass_throughs",
    "value.primary_home",
    "value.other_home",
    "value.re_fund",
    "value.other_nonfin",
)

ESTATE_DEBT_COLS: tuple[str, ...] = (
    "value.primary_mortgage",
    "value.other_mortgage",
    "value.credit_lines",
    "value.credit_cards",
    "value.installment_debt",
    "value.other_debt",
)

# The five asset classes whose unrealized gain the capital-gains model tracks,
# each with a market value and a cost basis.
CAPITAL_GAINS_VALUE_COLS: tuple[str, ...] = (
    "value.equities",
    "value.pass_throughs",
    "value.primary_home",
    "value.other_home",
    "value.re_fund",
)
CAPITAL_GAINS_BASIS_COLS: tuple[str, ...] = (
    "basis.equities",
    "basis.pass_throughs",
    "basis.primary_home",
    "basis.other_home",
    "basis.re_fund",
)

# Filing status codes as Tax-Data writes them.
FILING_SINGLE = 1
FILING_JOINT = 2
FILING_HEAD_OF_HOUSEHOLD = 4
N_FILING_STATUSES = 4

# A repeated estate value is a donor-clone cluster only when at least two
# records share it.
MIN_DONOR_CLUSTER_SIZE = 2

# Chained CPI-U replaced CPI-U for tax parameter indexation in 2017.
CHAINED_CPI_FIRST_YEAR = 2017

# Income sources used to apportion state income tax in the SALT-cap
# workaround step.
SALT_OTHER_INCOME_COLUMNS: tuple[str, ...] = (
    "wages_base1",
    "tips1",
    "ot1",
    "wages_base2",
    "tips2",
    "ot2",
    "trad_contr_er1",
    "trad_contr_er2",
    "txbl_int",
    "exempt_int",
    "div_ord",
    "div_pref",
    "state_ref",
    "txbl_ira_dist",
    "gross_pens_dist",
    "kg_st",
    "kg_lt",
    "other_gains",
    "alimony",
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
    "gross_ss",
    "ui",
    "other_inc",
)


REPEAL_HEAD_OF_HOUSEHOLD_COLUMN = "filing.repeal_hoh"


def head_of_household_repeal_switch(annual_law: Mapping[str, Array]) -> Array:
    """Whether this year's law repeals head-of-household filing status.

    The wide law table holds the switch as one number repeated across the four
    filing statuses, so the first row is taken. A law that never mentions the
    switch does not repeal the status.
    """
    switch = jnp.asarray(annual_law.get(REPEAL_HEAD_OF_HOUSEHOLD_COLUMN, 0))
    return switch[0] if switch.ndim else switch


def filing_status_under_law(filing_status: Array, repeal_hoh: Array) -> Array:
    """The filing status a household actually files under, given the repeal.

    Where the law repeals head-of-household status, those filers become single
    filers, and every filing-status-dependent parameter (the standard
    deduction, the rate brackets, every phase-out) applies its single-filer
    value.
    """
    filing_status = jnp.asarray(filing_status)
    becomes_single = (jnp.asarray(repeal_hoh) == 1) & (
        filing_status == FILING_HEAD_OF_HOUSEHOLD
    )
    return jnp.where(becomes_single, FILING_SINGLE, filing_status)


def recode_head_of_household(
    frame: Mapping[str, Array], repeal_hoh: Array, valid: Array
) -> dict[str, Array]:
    """The head-of-household recode.

    Head-of-household returns become single returns when the annual law
    repeals that filing status. Invalid slots retain the harmless Single code
    that population preparation gives them.
    """
    out = dict(frame)
    filing_status = jnp.asarray(frame["filing_status"])
    valid = jnp.asarray(valid, dtype=bool)
    out["filing_status"] = jnp.where(
        valid, filing_status_under_law(filing_status, repeal_hoh), filing_status
    )
    return out


def join_law(
    frame: Mapping[str, Array], annual_law: Mapping[str, Array]
) -> dict[str, Array]:
    """The annual law join.

    A scalar annual law value is kept as a scalar and broadcasts in later
    calculations. A filing-status-dependent value must have four rows, one
    for each Tax-Data filing status, and is gathered for every record. The
    host-side law compiler is responsible for ensuring that all four statuses
    are covered. This step needs no validity mask, because the filing status
    in every padded slot is already Single.
    """
    out = dict(frame)
    filing_index = jnp.asarray(frame["filing_status"], dtype=jnp.int32) - 1
    for name, value in annual_law.items():
        if name in ("year", "filing_status"):
            continue
        if name in out:
            raise ValueError(
                f"the frame already carries law column {name}, so the "
                "array join would overwrite it"
            )
        value = jnp.asarray(value)
        if value.ndim == 0:
            out[name] = value
        else:
            if value.shape[0] != N_FILING_STATUSES:
                raise ValueError(
                    f"annual law value {name} has {value.shape[0]} filing-status "
                    f"rows, expected {N_FILING_STATUSES}"
                )
            out[name] = value[filing_index]
    return out


def apply_salt_workaround(frame: Mapping[str, Array], valid: Array) -> dict[str, Array]:
    """The state SALT-cap workaround.

    The random participation draw is read from ``r.salt_workaround`` in the
    prepared population. Invalid slots move no state income tax.
    """
    out = dict(frame)
    valid = jnp.asarray(valid, dtype=bool)

    part = (
        frame["part_active"]
        + frame["part_passive"]
        - frame["part_active_loss"]
        - frame["part_passive_loss"]
        - frame["part_179"]
    )
    scorp = (
        frame["scorp_active"]
        + frame["scorp_passive"]
        - frame["scorp_active_loss"]
        - frame["scorp_passive_loss"]
        - frame["scorp_179"]
    )
    income = part + scorp
    for name in SALT_OTHER_INCOME_COLUMNS:
        income = income + frame[name]

    positive_income = income > 0
    denominator = jnp.where(positive_income, income, 1.0)
    part_share = jnp.where(positive_income, jnp.clip(part / denominator, 0.0, 1.0), 0.0)
    scorp_share = jnp.where(
        positive_income, jnp.clip(scorp / denominator, 0.0, 1.0), 0.0
    )

    salt = frame["salt_inc_sales"]
    salt_part = salt * part_share
    salt_scorp = jnp.minimum(salt - salt_part, salt * scorp_share)
    incentive = ~(jnp.isinf(frame["item.salt_limit"]) & jnp.isinf(frame["amt.exempt"]))
    draw = frame["r.salt_workaround"]
    sstb_rate = frame["item.salt_workaround_allowed_sstb"]
    non_sstb_rate = frame["item.salt_workaround_allowed_non_sstb"]

    def elected(sstb: Array) -> Array:
        """Whether the entity elects at its business-type participation rate."""
        rate = jnp.where(sstb == 1, sstb_rate, non_sstb_rate)
        return (draw < rate).astype(salt.dtype)

    sstb_part = frame["sstb_part"]
    sstb_scorp = frame["sstb_scorp"]
    workaround_part = jnp.where(
        valid
        & incentive
        & (frame["item.salt_workaround_allowed_part"] != 0)
        & ~jnp.isnan(sstb_part)
        & (salt_part != 0),
        salt_part * elected(sstb_part),
        0.0,
    )
    # There is no separate S corporation switch, so this leg does not test the
    # partnership switch the way the partnership leg above does.
    workaround_scorp = jnp.where(
        valid & incentive & ~jnp.isnan(sstb_scorp) & (salt_scorp != 0),
        salt_scorp * elected(sstb_scorp),
        0.0,
    )

    out["salt_workaround_part"] = workaround_part
    out["salt_workaround_scorp"] = workaround_scorp
    out["salt_inc_sales"] = jnp.where(
        valid, salt - workaround_part - workaround_scorp, 0.0
    )
    out["part_active_loss"] = jnp.where(
        valid, frame["part_active_loss"] + workaround_part, 0.0
    )
    out["scorp_active_loss"] = jnp.where(
        valid, frame["scorp_active_loss"] + workaround_scorp, 0.0
    )
    return out


def zero_social_security_in_padded_rows(
    frame: Mapping[str, Array], valid: Array
) -> dict[str, Array]:
    """Social Security benefits are set to zero in slots that hold no household."""
    return {
        **frame,
        "gross_ss": jnp.where(
            jnp.asarray(valid, dtype=bool), frame["gross_ss"], 0.0
        ),
    }


def calc_kg_cpi_ratio(
    frame: Mapping[str, Array],
    year: Array,
    index_years: Array,
    cpi_levels: Array,
    valid: Array,
) -> dict[str, Array]:
    """The capital-gains basis price ratio.

    Each long-term gain receives today's price level divided by the price
    level in its rounded purchase year. A purchase before the price series is
    assigned the first available year. Invalid slots and gains that do not
    qualify for indexation receive a ratio of one.
    """
    out = dict(frame)
    index_years = jnp.asarray(index_years)
    cpi_levels = jnp.asarray(cpi_levels)
    if index_years.ndim != 1 or cpi_levels.ndim != 1:
        raise ValueError("CPI years and levels must be vectors")
    if index_years.shape != cpi_levels.shape:
        raise ValueError("CPI years and levels must have equal length")

    years_held = frame["kg_lt_years_held"]
    safe_years_held = jnp.nan_to_num(years_held, nan=0.0)
    purchase_year = jnp.maximum(
        jnp.asarray(year) - jnp.round(safe_years_held), index_years[0]
    )
    purchase_row = jnp.searchsorted(index_years, purchase_year)
    purchase_row = jnp.clip(purchase_row, 0, index_years.shape[0] - 1)
    current_row = jnp.searchsorted(index_years, jnp.asarray(year))
    current_row = jnp.clip(current_row, 0, index_years.shape[0] - 1)
    ratio = cpi_levels[current_row] / cpi_levels[purchase_row]
    no_indexation = (
        ~jnp.asarray(valid, dtype=bool)
        | (frame["kg_lt"] == 0)
        | jnp.isnan(years_held)
        | (years_held == 0)
    )
    out["kg_lt_cpi_ratio"] = jnp.where(no_indexation, 1.0, ratio)
    missing_holding = (
        jnp.asarray(valid, dtype=bool) & (frame["kg_lt"] != 0) & jnp.isnan(years_held)
    )
    # A gain with no holding period is an invalid input. Traced code cannot
    # raise a data-dependent Python exception, so the ratio is set to NaN to
    # make the defect visible in the annual result.
    out["kg_lt_cpi_ratio"] = jnp.where(missing_holding, jnp.nan, out["kg_lt_cpi_ratio"])
    return out


def sum_missing_as_zero(frame: Mapping[str, Array], names: tuple[str, ...]) -> Array:
    """Add up the named columns, counting a missing entry as zero.

    Estate microdata leave some balance-sheet fields missing, and those
    missing asset and debt amounts count as zero in the totals. The source
    column itself is left alone.
    """
    total = jnp.zeros_like(frame[names[0]])
    for name in names:
        total = total + jnp.nan_to_num(frame[name], nan=0.0)
    return total


def calc_estate_mortality(
    frame: Mapping[str, Array],
    cluster_cap: Array,
    cluster_floor: Array,
    valid: Array,
) -> dict[str, Array]:
    """Estate mortality.

    Candidate estates are sorted by gross estate rounded to the dollar.
    Segment sums then measure each donor-clone cluster's expected deaths and
    size. A cluster of at least two records is scaled when its expected deaths
    exceed the supplied cap. Invalid slots never enter a cluster and receive
    zero mortality.

    Two columns come back: ``estate_m``, the estate tax's chance of dying, and
    ``donor_cluster_scale``, the factor each household's chance of dying was
    multiplied by (below one in a capped cluster and one everywhere else). The
    capital-gains model applies the same factor to its own chance of dying.
    """
    out = dict(frame)
    valid = jnp.asarray(valid, dtype=bool)
    economic_gross = sum_missing_as_zero(frame, ESTATE_ASSET_COLS)

    q1 = jnp.nan_to_num(frame["q_death1"], nan=0.0)
    q2 = jnp.nan_to_num(frame["q_death2"], nan=0.0)
    joint = (frame["filing_status"] == FILING_JOINT) & (q2 > 0)
    mortality = jnp.where(joint, q1 * q2, q1)
    mortality = jnp.where(valid, mortality, 0.0)
    weight = jnp.where(valid, jnp.nan_to_num(frame["weight"], nan=0.0), 0.0)

    candidate = valid & (economic_gross > cluster_floor)
    cluster_key = jnp.where(candidate, jnp.round(economic_gross), jnp.nan)
    sort_key = jnp.where(candidate, cluster_key, jnp.inf)
    order = jnp.argsort(sort_key, stable=True)
    sorted_key = sort_key[order]
    sorted_candidate = candidate[order]

    previous_key = jnp.concatenate((jnp.asarray([jnp.nan]), sorted_key[:-1]))
    previous_candidate = jnp.concatenate((jnp.asarray([False]), sorted_candidate[:-1]))
    starts_group = sorted_candidate & (
        ~previous_candidate | (sorted_key != previous_key)
    )
    segment = jnp.cumsum(starts_group.astype(jnp.int32)) - 1
    n_records = economic_gross.shape[0]
    # If any slot is not a candidate, the last segment number is unused by a
    # real cluster and safely holds the excluded rows' zero contributions.
    safe_segment = jnp.where(sorted_candidate, segment, n_records - 1)
    sorted_death_weight = jnp.where(sorted_candidate, (weight * mortality)[order], 0.0)
    cluster_deaths = jax.ops.segment_sum(
        sorted_death_weight, safe_segment, num_segments=n_records
    )
    cluster_size = jax.ops.segment_sum(
        sorted_candidate.astype(jnp.int32), safe_segment, num_segments=n_records
    )
    row_deaths = cluster_deaths[safe_segment]
    row_size = cluster_size[safe_segment]
    capped = (
        sorted_candidate
        & (row_size >= MIN_DONOR_CLUSTER_SIZE)
        & (row_deaths > cluster_cap)
    )
    capped_deaths = jnp.where(capped, row_deaths, 1.0)
    scale = jnp.where(capped, cluster_cap / capped_deaths, 1.0)
    sorted_mortality = mortality[order] * scale
    mortality = jnp.zeros_like(mortality).at[order].set(sorted_mortality)

    out["estate_m"] = mortality
    out["donor_cluster_scale"] = jnp.ones_like(mortality).at[order].set(scale)
    out["estate_economic_gross"] = economic_gross
    return out


def add_capital_gains_record_attributes(
    frame: Mapping[str, Array],
    valid: Array,
    shares: widowhood.WidowhoodShares | None = None,
) -> dict[str, Array]:
    """Three household attributes the capital-gains applier reads.

    The applier spreads an age cell's answers over the households in it using
    each household's stock of unrealized gain, its chance of dying this year,
    and, if it is unmarried, its chance of being a surviving spouse. All three
    are arithmetic on Tax-Data columns. An empty population slot carries no
    gain and no chance of dying.

    The stock of unrealized gain is market value less cost basis, added up
    over the five tracked asset classes. A class whose basis exceeds its value
    contributes zero, since an unsold loss is not a gain waiting to be
    realized. A class with a missing value or a missing basis also contributes
    zero.

    The chance of dying this year is the chance both spouses die on a joint
    return, since that is when the couple's gains are settled, and the filer's
    own chance otherwise, multiplied by the ``donor_cluster_scale`` that
    `calc_estate_mortality` writes. The frame must therefore have been through
    `calc_estate_mortality` first. The factor applies the estate tax's cap on
    the expected deaths in a cluster of replicated donor records. The result
    still differs from `estate_m` for a joint return on which the second spouse's
    chance of dying is recorded as zero: here the couple never dies, while the
    estate tax uses the first spouse's chance.

    The chance of being widowed comes from the age-and-sex table in
    `widowhood.py` and is zero for anyone who is not a single filer or head of
    household, and for anyone Tax-Data records as divorced. It is computed
    under every law; where the law does not tax gains at death with an
    exclusion, the two branches it blends between are identical.
    """
    out = dict(frame)
    if shares is None:
        shares = widowhood.read_widowhood_shares()
    valid = jnp.asarray(valid, dtype=bool)

    stock = jnp.zeros_like(frame[CAPITAL_GAINS_VALUE_COLS[0]])
    for value_column, basis_column in zip(
        CAPITAL_GAINS_VALUE_COLS, CAPITAL_GAINS_BASIS_COLS, strict=True
    ):
        gain = jnp.nan_to_num(frame[value_column] - frame[basis_column], nan=0.0)
        stock = stock + jnp.maximum(gain, 0.0)
    out["unrealized_gain_stock"] = jnp.where(valid, stock, 0.0)

    own_death = frame["q_death1"]
    spouse_death = frame["q_death2"]
    both_die = (frame["filing_status"] == FILING_JOINT) & ~jnp.isnan(spouse_death)
    mortality = jnp.nan_to_num(
        jnp.where(both_die, own_death * spouse_death, own_death), nan=0.0
    )
    out["m_household"] = jnp.where(
        valid, mortality * frame["donor_cluster_scale"], 0.0
    )

    bounds = jnp.asarray(shares.band_lower_bounds)
    age = jnp.where(jnp.isnan(frame["age1"]), bounds[0], frame["age1"])
    band = jnp.clip(
        jnp.searchsorted(bounds, age, side="right") - 1, 0, bounds.shape[0] - 1
    )
    is_male = jnp.nan_to_num(frame["male1"], nan=0.0) == 1
    share = jnp.where(
        is_male,
        jnp.asarray(shares.share_widowed_male)[band],
        jnp.asarray(shares.share_widowed_female)[band],
    )
    unmarried = jnp.isin(
        frame["filing_status"], jnp.asarray(widowhood.UNMARRIED_FILING_STATUSES)
    )
    never_divorced = jnp.isnan(frame["divorce_year"])
    out["p_widow"] = jnp.where(valid & unmarried & never_divorced, share, 0.0)
    return out


def add_net_worth(frame: Mapping[str, Array], valid: Array) -> dict[str, Array]:
    """Net worth.

    Net worth is the raw value of assets less debts. A missing balance-sheet
    entry counts as zero for this total, while the source entry remains
    unchanged. Invalid slots receive zero net worth.
    """
    out = dict(frame)
    assets = sum_missing_as_zero(frame, ESTATE_ASSET_COLS)
    debts = sum_missing_as_zero(frame, ESTATE_DEBT_COLS)
    out["net_worth"] = jnp.where(jnp.asarray(valid, dtype=bool), assets - debts, 0.0)
    return out


def build_world_frame(
    frame: Mapping[str, Array],
    year: Array,
    annual_law: Mapping[str, Array],
    index_years: Array,
    cpi_levels: Array,
    cluster_death_weight_cap: Array,
    cluster_floor: Array,
    valid: Array,
    widowhood_shares: widowhood.WidowhoodShares | None = None,
) -> dict[str, Array]:
    """Every annual frame step, in order.

    The steps are the head-of-household recode, the law join, the state
    SALT-cap workaround, zeroing Social Security in empty slots, the
    capital-gains basis price ratio, estate mortality, net worth, and the
    three household attributes the capital-gains model reads. The function
    accepts and returns only array dictionaries, so it can run inside the
    compiled annual calculation.
    """
    repeal_hoh = head_of_household_repeal_switch(annual_law)
    out = recode_head_of_household(frame, repeal_hoh, valid)
    out = join_law(out, annual_law)
    out = apply_salt_workaround(out, valid)
    out = zero_social_security_in_padded_rows(out, valid)
    out = calc_kg_cpi_ratio(out, year, index_years, cpi_levels, valid)
    out = calc_estate_mortality(out, cluster_death_weight_cap, cluster_floor, valid)
    out = add_net_worth(out, valid)
    return add_capital_gains_record_attributes(out, valid, widowhood_shares)
