#-------------------------------------------------------------------------------
# test_generator.py
#
# Checks on the synthetic battery itself: it is the same every time it is
# built, it covers every kink class it claims to, its kinks come in
# below/at/above triples, and the records are shaped like filing units. These
# run on CPU in public CI — nothing here reads microdata.
#-------------------------------------------------------------------------------

import math

import numpy as np
import pandas as pd
import pytest
from generator import (
    CASE_CLASSES,
    CENT,
    COLUMNS,
    FILING_STATUSES,
    NUMERIC_FIELDS,
    YEARS,
    generate,
    generate_all,
    placeholder_law,
)


@pytest.fixture(scope='module')
def battery():
    return generate_all()


def test_generation_is_deterministic():
    # Two builds of the battery are the same battery. Fixtures that drifted
    # between runs would turn every downstream parity failure into a question
    # about which run it came from.
    first = generate_all()
    second = generate_all()
    pd.testing.assert_frame_equal(first, second)


def test_year_generation_is_deterministic():
    for year in YEARS:
        pd.testing.assert_frame_equal(generate(year), generate(year))


def test_every_case_class_is_populated(battery):
    counts = battery['case_class'].value_counts()
    for case_class in CASE_CLASSES:
        assert counts.get(case_class, 0) > 0, \
            f'no synthetic units for case class {case_class}'


def test_named_kinks_are_present(battery):
    # The provisions the battery exists for. If one of these disappears the
    # generator has quietly stopped probing a schedule.
    kinks = set(battery['kink'])
    required = {
        'ord_bracket1_fs1',              # bottom ordinary bracket edge, single
        'pref_bracket1_fs2',             # 0-to-15 preferred edge, joint
        'amt_exempt_po_start_fs1',       # AMT exemption phase-out start
        'ss_wage_base_single_earner',    # Social Security wage base
        'seca_stacks_onto_wage_base',    # SECA stacked on wages at the base
        'eitc_phasein_end_0kids_fs1',    # childless EITC phase-in end
        'eitc_phaseout_end_2kids_fs4',   # two-child EITC phase-out end
        'ctc_po_step1_fs2',              # second riser of the CTC staircase
        'ctc_refundable_cap_2kids',      # per-child refundable cap binding
        'itemize_tie_fs1',               # itemized exactly equal to standard
        'ss_taxability_tier2_fs2',       # 85 percent benefit taxability tier
        'estate_exemption_single',       # one decedent's exemption
        'estate_soi_bin1_edge',          # first SOI measurement bin edge
        'panel_enters_in_last_year',     # unit absent from the earlier year
    }
    assert required <= kinks, sorted(required - kinks)


def test_eitc_covers_every_child_count(battery):
    eitc = battery[battery['case_class'] == 'eitc']
    for n_kids in (0, 1, 2, 3):
        for kink in ('phasein_end', 'plateau_end', 'phaseout_end'):
            match = eitc['kink'].str.contains(f'eitc_{kink}_{n_kids}kids')
            assert match.any(), f'EITC {kink} missing at {n_kids} children'


def test_kinks_come_in_below_at_above_triples(battery):
    # Every kink either has all three positions or is a standalone case.
    one_year = battery[battery['year'] == YEARS[0]]
    for kink, group in one_year.groupby('kink'):
        positions = set(group['position'])
        assert positions in ({'below', 'at', 'above'}, {'single'}), \
            f'{kink} has positions {sorted(positions)}'


def test_triples_straddle_by_one_cent(battery):
    # The three units of a triple differ by a cent somewhere. A triple whose
    # members were identical would pass every calculator test without ever
    # testing the boundary.
    one_year = battery[battery['year'] == YEARS[0]]
    triples = one_year[one_year['position'] != 'single']
    for kink, group in triples.groupby('kink'):
        by_position = group.set_index('position')[list(NUMERIC_FIELDS)]
        below = by_position.loc['below']
        above = by_position.loc['above']
        gap = (above - below).abs().replace([np.inf, np.nan], 0.0).max()
        # Two cents apart on the dial, or more where the dial feeds an input
        # through a rate: self-employment earnings are grossed up by the
        # 92.35 percent taxable share before they reach the wage base. The
        # slack is for a cent placed on a fifty-million-dollar estate, which
        # loses a few billionths to binary floating point.
        assert gap >= 2 * CENT * (1 - 1e-6), \
            f'{kink} below and above do not straddle the boundary'


def test_itemization_tie_takes_the_standard_deduction(battery):
    # The R model itemizes on a strict greater-than, so a unit whose itemized
    # deductions exactly equal the standard deduction takes the standard one.
    tie = battery[(battery['case_class'] == 'itemization') &
                  (battery['position'] == 'at')]
    assert len(tie) > 0
    assert (tie['item_ded'] == tie['std_ded']).all()
    assert (tie['itemizing'] == 0.0).all()
    above = battery[(battery['case_class'] == 'itemization') &
                    (battery['position'] == 'above')]
    assert (above['itemizing'] == 1.0).all()


def test_donor_clones_are_identical_apart_from_id_and_weight(battery):
    clones = battery[(battery['year'] == YEARS[0]) &
                     battery['kink'].str.startswith('estate_donor_clone')]
    assert len(clones) >= 2
    values = clones[list(NUMERIC_FIELDS)].drop_duplicates()
    assert len(values) == 1, 'donor clones differ in a tax input'
    assert clones['id'].nunique() == len(clones)
    assert clones['weight'].nunique() > 1


def test_panel_units_appear_and_disappear(battery):
    panel = battery[battery['case_class'] == 'panel']
    years_by_id = panel.groupby('id')['year'].nunique()
    assert (years_by_id == len(YEARS)).any(), 'no unit present in every year'
    assert (years_by_id < len(YEARS)).any(), \
        'no unit missing from a year; panel-shape tests have nothing to bite on'


def test_ids_are_unique_within_a_year_and_stable_across_years(battery):
    for year in YEARS:
        one_year = battery[battery['year'] == year]
        assert one_year['id'].is_unique, f'duplicate ids in {year}'

    # A unit that exists in both years keeps its id and its case.
    both = battery.groupby('id')['case'].nunique()
    assert (both == 1).all(), 'an id is reused for a different case'


def test_zero_and_negative_agi_present(battery):
    extreme = battery[battery['case_class'] == 'extreme_agi']
    assert (extreme['agi'] == 0.0).any()
    assert (extreme['agi'] < 0.0).any()
    # Negative AGI with positive earnings: the EITC and refundable CTC run on
    # earnings, not AGI, so this unit must be in the battery.
    loss = extreme[extreme['kink'] == 'agi_negative_business_loss']
    assert (loss['wages_base1'] > 0).all() and (loss['agi'] < 0).all()


def test_dependent_corners_span_zero_one_and_many(battery):
    deps = battery[battery['case_class'] == 'dependents']
    counts = set(deps['n_dep'])
    assert {0.0, 1.0} <= counts
    assert max(counts) >= 4.0, 'no unit with more dependents than CTC slots'
    # Dependent count and EITC qualifying-child count come apart somewhere.
    assert (deps['n_dep'] != deps['n_dep_eitc']).any()


def test_schema_is_sane(battery):
    assert list(battery.columns) == list(COLUMNS)
    assert battery['weight'].gt(0).all()
    assert battery['filing_status'].isin(FILING_STATUSES).all()
    assert battery['year'].isin(YEARS).all()
    assert battery['case'].notna().all()

    # Only the empty dependent slots carry infinity — that is the model's
    # encoding for "no dependent here", which fails every child age test.
    finite = [f for f in NUMERIC_FIELDS
              if f not in ('dep_age1', 'dep_age2', 'dep_age3')]
    assert np.isfinite(battery[finite].to_numpy()).all()
    for slot in (1, 2, 3):
        column = battery[f'dep_age{slot}']
        assert column.notna().all()
        assert ((column >= 0) | np.isinf(column)).all()


def test_law_override_moves_the_kink():
    # The `law` argument of generate() is how a caller replaces one of this
    # file's placeholder thresholds: pass a threshold and the cases built on it
    # move with it, with nothing else disturbed.
    moved_base = 176_400.0
    default = generate(YEARS[0])
    override = generate(YEARS[0], law={'ss_wage_base': moved_base})

    def wage_base_case(frame):
        row = frame[(frame['kink'] == 'ss_wage_base_single_earner') &
                    (frame['position'] == 'at')]
        return row['wages_base1'].iloc[0]

    assert wage_base_case(override) == moved_base
    assert wage_base_case(default) != moved_base

    untouched = default['kink'].str.startswith('eitc_')
    pd.testing.assert_frame_equal(default[untouched].reset_index(drop=True),
                                  override[untouched.values]
                                  .reset_index(drop=True))


def test_indexed_thresholds_rise_and_statutory_ones_do_not():
    first, last = placeholder_law(YEARS[0]), placeholder_law(YEARS[-1])
    # The standard deduction and the wage base are indexed for inflation.
    assert last['std_ded'][1] > first['std_ded'][1]
    assert last['ss_wage_base'] > first['ss_wage_base']
    # The Social Security taxability thresholds and the Additional Medicare
    # Tax threshold are fixed in statute and have never been indexed.
    assert last['ss_tier1_thresh'] == first['ss_tier1_thresh']
    assert last['add_med_thresh'] == first['add_med_thresh']


def test_indexed_thresholds_land_on_the_rounding_increment():
    law = placeholder_law(YEARS[-1])
    for edge in law['ord_brackets'][1]:
        assert math.isclose(edge % 50.0, 0.0, abs_tol=1e-6) or edge == 0.0
