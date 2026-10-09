#-------------------------------------------------------------------------------
# generator.py
#
# Deterministic synthetic tax units built to sit on the kinks of federal tax
# law: bracket edges of every schedule (ordinary, preferred, AMT, payroll,
# estate), the EITC's three kinks at each child count, the CTC's discretized
# phase-out and its refundability caps, the itemize-or-not tie, mixed
# wage/self-employment earners at the Social Security wage base, dependent
# corners, zero and negative AGI, the Social Security benefit taxability
# tiers, the estate measurement bins, donor clones, and units that enter or
# leave the panel between years.
#
# Nothing here comes from the PUF or any IRS microdata: these fixtures are
# public and run in GitHub CI. There is no randomness — every unit is written
# down on purpose, so a failing test names the provision it broke.
#
# The dollar thresholds below are placeholders written into this file by hand,
# chosen to look like 2026-2027 law. They are not authoritative and no parity
# claim rests on them. What this file owns is the case structure — where the
# kinks are and what a unit standing on one looks like — and that structure is
# what the tests check. An earlier plan was to read the real thresholds out of
# compiled law instead; that reader was never wired up and has been deleted, so
# the placeholders here are the thresholds these fixtures use.
#
# Variable names match what the calculators consume today (poc/tax_functions*.py,
# which mirror src/calc in the R model), so the fixtures plug in without a
# rename layer.
#-------------------------------------------------------------------------------

from __future__ import annotations

import copy
import math

import pandas as pd

# Filing status codes as the R model encodes them: single, married filing
# jointly, married filing separately, head of household. The calculators test
# these numerically (MFS returns are shut out of the CTC and, in baseline law,
# the EITC), so the codes are part of the fixture contract.
SINGLE = 1
MARRIED_JOINT = 2
MARRIED_SEPARATE = 3
HEAD_OF_HOUSEHOLD = 4
FILING_STATUSES = (SINGLE, MARRIED_JOINT, MARRIED_SEPARATE, HEAD_OF_HOUSEHOLD)

# One cent. Kinks are probed a cent below, exactly on, and a cent above the
# threshold: that is the smallest amount that can appear on a return, and it
# is what separates "the credit is still at its maximum" from "the phase-out
# has started."
CENT = 0.01

# Years the battery is generated for. Two adjacent years is enough to exercise
# indexation and panel entry/exit; the run years of the model proper are set
# by the runscript, not here.
YEARS = (2026, 2027)

# Every synthetic unit carries the same weight unless the case is about
# weights (the donor clones). A round number keeps hand-checked revenue
# arithmetic legible.
DEFAULT_WEIGHT = 100.0


#-------------------------------------------------------------------------------
# Placeholder law
#-------------------------------------------------------------------------------

# Nominal 2026 levels, roughly post-OBBBA. Marked placeholder throughout: the
# structure (which parameters exist, which vary by filing status, which are
# indexed) is meant to be right; the dollars are not authoritative.
_LAW_2026 = {
    # Standard deduction by filing status.
    'std_ded': {SINGLE: 16_100.0, MARRIED_JOINT: 32_200.0,
                MARRIED_SEPARATE: 16_100.0, HEAD_OF_HOUSEHOLD: 24_150.0},

    # Lower edge of each ordinary bracket. Seven brackets at 10/12/22/24/32/
    # 35/37 percent. The married-joint edges here are simply twice the single
    # edges, which is true of every bracket but the top one in actual law —
    # a placeholder simplification, not a modeling claim.
    'ord_brackets': {
        SINGLE: [0.0, 12_400.0, 50_400.0, 105_700.0, 201_775.0, 256_225.0,
                 640_600.0],
        MARRIED_JOINT: [0.0, 24_800.0, 100_800.0, 211_400.0, 403_550.0,
                        512_450.0, 1_281_200.0],
        MARRIED_SEPARATE: [0.0, 12_400.0, 50_400.0, 105_700.0, 201_775.0,
                           256_225.0, 640_600.0],
        HEAD_OF_HOUSEHOLD: [0.0, 17_700.0, 67_450.0, 105_700.0, 201_775.0,
                            256_225.0, 640_600.0],
    },
    'ord_rates': [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],

    # Lower edge of each preferred bracket: qualified dividends and long-term
    # gains at 0/15/20 percent. The preferred rate is applied to the top slice
    # of taxable income, so the edges are read against total taxable income,
    # not against the gains alone.
    'pref_brackets': {
        SINGLE: [0.0, 49_450.0, 545_500.0],
        MARRIED_JOINT: [0.0, 98_900.0, 613_700.0],
        MARRIED_SEPARATE: [0.0, 49_450.0, 306_850.0],
        HEAD_OF_HOUSEHOLD: [0.0, 66_200.0, 579_600.0],
    },

    # AMT exemption, the AMTI level where the exemption starts phasing out at
    # 50 cents per dollar, and the lower edge of the 28 percent AMT bracket.
    'amt_exempt': {SINGLE: 90_100.0, MARRIED_JOINT: 140_200.0,
                   MARRIED_SEPARATE: 70_100.0, HEAD_OF_HOUSEHOLD: 90_100.0},
    'amt_exempt_po_thresh': {SINGLE: 500_000.0, MARRIED_JOINT: 1_000_000.0,
                             MARRIED_SEPARATE: 500_000.0,
                             HEAD_OF_HOUSEHOLD: 500_000.0},
    'amt_exempt_po_rate': 0.50,
    'amt_brackets': {SINGLE: [0.0, 244_500.0], MARRIED_JOINT: [0.0, 244_500.0],
                     MARRIED_SEPARATE: [0.0, 122_250.0],
                     HEAD_OF_HOUSEHOLD: [0.0, 244_500.0]},

    # Payroll. Wages up to the wage base carry the 12.4 percent OASDI tax
    # (both sides); Medicare has no cap. The Additional Medicare Tax threshold
    # is fixed in statute and never indexed. Self-employment earnings below
    # the reporting threshold owe no SECA at all, and 92.35 percent of the
    # remainder is the taxable base.
    'ss_wage_base': 184_500.0,
    'add_med_thresh': {SINGLE: 200_000.0, MARRIED_JOINT: 250_000.0,
                       MARRIED_SEPARATE: 125_000.0,
                       HEAD_OF_HOUSEHOLD: 200_000.0},
    'se_thresh': 400.0,
    'seca_taxable_rate': 0.9235,

    # EITC by number of qualifying children, capped at three-or-more. The
    # credit rises at pi_rate on earnings up to pi_end, sits flat until
    # po_thresh, then falls at po_rate. Married-joint filers get the phase-out
    # threshold pushed out by a fixed amount.
    'eitc_pi_rate': {0: 0.0765, 1: 0.3400, 2: 0.4000, 3: 0.4500},
    'eitc_pi_end': {0: 8_490.0, 1: 12_730.0, 2: 17_880.0, 3: 17_880.0},
    'eitc_po_thresh': {0: 10_620.0, 1: 23_350.0, 2: 23_350.0, 3: 23_350.0},
    'eitc_po_rate': {0: 0.0765, 1: 0.1598, 2: 0.2106, 3: 0.2106},
    'eitc_po_thresh_mfj_bump': 7_120.0,

    # CTC. The credit falls $50 for each $1,000 (or fraction of $1,000) of AGI
    # above the threshold — that "or fraction" is what makes the phase-out a
    # staircase rather than a ramp, and it is the discretization the fixtures
    # probe. The refundable portion is 15 cents per dollar of earnings above
    # the earnings threshold, capped per child.
    'ctc_value': 2_200.0,
    'ctc_po_thresh': {SINGLE: 200_000.0, MARRIED_JOINT: 400_000.0,
                      MARRIED_SEPARATE: 200_000.0,
                      HEAD_OF_HOUSEHOLD: 200_000.0},
    'ctc_po_discrete_step': 1_000.0,
    'ctc_pi_thresh': 2_500.0,
    'ctc_pi_rate': 0.15,
    'ctc_max_refund': 1_700.0,

    # Social Security benefit taxability. Up to half of benefits enter AGI
    # once "combined income" (other income, plus tax-exempt interest, plus
    # half of benefits) passes the first threshold; up to 85 percent once it
    # passes the second. Neither threshold has ever been indexed.
    'ss_tier1_thresh': {SINGLE: 25_000.0, MARRIED_JOINT: 32_000.0,
                        MARRIED_SEPARATE: 0.0, HEAD_OF_HOUSEHOLD: 25_000.0},
    'ss_tier2_thresh': {SINGLE: 34_000.0, MARRIED_JOINT: 44_000.0,
                        MARRIED_SEPARATE: 0.0, HEAD_OF_HOUSEHOLD: 34_000.0},

    # Estate. One exemption per decedent, doubled for a surviving-spouse
    # (joint) death in the model's latent states. The graduated schedule below
    # the exemption is absorbed by the unified credit for ordinary estates but
    # still drives the tentative-tax arithmetic, so its edges are probed.
    'estate_exemption': 15_000_000.0,
    'estate_brackets': [0.0, 10_000.0, 20_000.0, 40_000.0, 60_000.0, 80_000.0,
                        100_000.0, 150_000.0, 250_000.0, 500_000.0, 750_000.0,
                        1_000_000.0],

    # Edges of the SOI size bins the estate measurement parameters are looked
    # up in (deduction share, portability election probability, DSUE amount).
    # A unit sitting a cent either side of an edge gets a different set of
    # measurement parameters.
    'estate_soi_bins': [0.0, 5_000_000.0, 10_000_000.0, 20_000_000.0,
                        50_000_000.0],

    # Ratio of reported gross estate to economic wealth in the valuation
    # bridge. Held at one here so a fixture's economic_gross is also its
    # reported gross and the bin edges land where they are written; the real
    # ratio is a frozen calibrated parameter (bridge.yaml).
    'estate_valuation_r': 1.0,
}

# Placeholder chained-CPI growth between the two fixture years, and the
# statutory rounding increment for indexed dollar amounts.
_PLACEHOLDER_INDEX_FACTOR = 1.023
_INDEX_INCREMENT = 50.0

# Parameters that are indexed for inflation. Everything not listed is fixed in
# statute (the Additional Medicare threshold, the Social Security taxability
# thresholds, the CTC phase-out and earnings thresholds, the estate rate
# schedule's dollar edges) or is a rate, and is carried forward unchanged.
_INDEXED_PARAMS = ('std_ded', 'ord_brackets', 'pref_brackets', 'amt_exempt',
                   'amt_exempt_po_thresh', 'amt_brackets', 'ss_wage_base',
                   'eitc_pi_end', 'eitc_po_thresh', 'eitc_po_thresh_mfj_bump',
                   'ctc_max_refund', 'estate_exemption')


def _index_value(value, factor):
    # Indexation applied the way the R model does it: scale, then round to the
    # statutory increment. Zero and non-dollar leaves pass through untouched.
    if isinstance(value, dict):
        return {k: _index_value(v, factor) for k, v in value.items()}
    if isinstance(value, list):
        return [_index_value(v, factor) for v in value]
    if value == 0.0:
        return 0.0
    return round(value * factor / _INDEX_INCREMENT) * _INDEX_INCREMENT


def placeholder_law(year: int) -> dict:
    """Placeholder thresholds for `year`, indexed off the 2026 table.

    Standing in for compiled law until the law compiler exists. Overriding
    this through generate(law=...) replaces whole parameters at once, the way
    a reform delta replaces a whole subparameter in the YAML law files.
    """
    if year < 2026:
        raise ValueError(
            f'the placeholder law table starts in 2026; got {year}')
    law = copy.deepcopy(_LAW_2026)
    for _ in range(year - 2026):
        for name in _INDEXED_PARAMS:
            law[name] = _index_value(law[name], _PLACEHOLDER_INDEX_FACTOR)
    return law


#-------------------------------------------------------------------------------
# Record schema
#-------------------------------------------------------------------------------

# The inputs the calculators read, in the R model's own names. A fixture sets
# the handful its case is about; everything else stays at zero, which is a
# valid return (a unit with no income of that kind).
# Wage income is carried the way the model carries it: an earner's base pay
# (everything that is neither tips nor overtime) is its own field, so nothing
# here counts a dollar twice. An earner's W-2 wages are wages_base + tips + ot.
_INCOME_FIELDS = (
    'wages_base1', 'wages_base2', 'tips1', 'tips2', 'ot1', 'ot2',
    'trad_contr_er1', 'trad_contr_er2',
    'sole_prop1', 'sole_prop2', 'farm1', 'farm2', 'part_se1', 'part_se2',
    'txbl_int', 'exempt_int', 'div_ord', 'div_pref', 'txbl_kg', 'sch_e',
    'part_scorp', 'kg_pref', 'kg_1250', 'kg_collect', 'gross_ss',
    'part_active', 'part_active_loss', 'part_179',
    'scorp_active', 'scorp_active_loss', 'scorp_179',
    'ei_prior_yr',
)

_DEDUCTION_FIELDS = (
    'agi', 'txbl_inc', 'pe_ded', 'std_ded', 'item_ded', 'item_ded_ex_limits',
    'salt_item_ded', 'misc_item_ded', 'inv_int_item_ded', 'qbi_ded',
    'tip_ded', 'ot_ded', 'senior_ded', 'state_ref', 'amt_nols',
)

_CREDIT_FIELDS = (
    'ftc', 'amt_ftc', 'excess_ptc', 'old_cred', 'savers_cred',
    'res_energy_cred', 'gbc', 'prior_yr_cred', 'other_nonref', 'net_ptc',
    'recapture_tax', 'ira_penalty', 'caregiver_cred_nonref',
    'caregiver_cred_ref', 'wage_subsidy1', 'wage_subsidy2',
    'cdctc_nonref', 'cdctc_ref', 'ed_nonref', 'ed_ref', 'rebate',
)

_UNIT_FIELDS = (
    'filing_status', 'filer', 'dep_status', 'itemizing',
    'n_dep', 'n_dep_eitc', 'age1', 'age2', 'has_age2',
    'dep_age1', 'dep_age2', 'dep_age3',
    'dep_ssn1', 'dep_ssn2', 'dep_ssn3', 'dep_ctc1', 'dep_ctc2', 'dep_ctc3',
)

# Estate inputs. economic_gross and estate_debts are the balance sheet summed
# to a single number; q_death1/q_death2 are the mortality probabilities that
# select the joint / surviving-spouse latent states.
_ESTATE_FIELDS = (
    'economic_gross', 'estate_debts', 'value_pass_throughs',
    'q_death1', 'q_death2', 'estate_income_tax_ded', 'estate_concealed_frac',
)

NUMERIC_FIELDS = (_INCOME_FIELDS + _DEDUCTION_FIELDS + _CREDIT_FIELDS +
                  _UNIT_FIELDS + _ESTATE_FIELDS)

# Labels: case_class groups the battery, kink names the boundary, position
# says which side of it the unit sits on, and case is the three glued together
# so a failing assertion prints the provision it broke.
LABEL_COLUMNS = ('id', 'year', 'case', 'case_class', 'kink', 'position',
                 'weight')

COLUMNS = LABEL_COLUMNS + NUMERIC_FIELDS


def _blank(filing_status: int = SINGLE) -> dict:
    # A filing unit with no income, no dependents, and no deductions. The
    # missing-value encodings are the ones the calculators expect: a second
    # filer's age is -1 with has_age2 off, and an absent dependent slot has
    # age infinity so it fails every child age test.
    rec = {field: 0.0 for field in NUMERIC_FIELDS}
    rec['filing_status'] = float(filing_status)
    rec['filer'] = 1.0
    rec['age1'] = 40.0
    rec['age2'] = 45.0 if filing_status == MARRIED_JOINT else -1.0
    rec['has_age2'] = 1.0 if filing_status == MARRIED_JOINT else 0.0
    for slot in (1, 2, 3):
        rec[f'dep_age{slot}'] = math.inf
    return rec


def _add_children(rec: dict, n_children: int, age: float = 8.0,
                  ctc_qualifying: bool = True, has_ssn: bool = True) -> None:
    # Put children in the return. n_dep and n_dep_eitc are unit-level counts;
    # the per-dependent attributes the CTC reads only exist for three slots in
    # the R model, so a fourth child is counted for the EITC and for personal
    # exemptions but is invisible to the CTC's age and SSN tests. That gap is
    # in the source model, not introduced here.
    rec['n_dep'] = float(n_children)
    rec['n_dep_eitc'] = float(n_children)
    for slot in (1, 2, 3):
        if slot <= n_children:
            rec[f'dep_age{slot}'] = age
            rec[f'dep_ctc{slot}'] = 1.0 if ctc_qualifying else 0.0
            rec[f'dep_ssn{slot}'] = 1.0 if has_ssn else 0.0


def _label(rec: dict, case_class: str, kink: str, position: str) -> dict:
    rec['case_class'] = case_class
    rec['kink'] = kink
    rec['position'] = position
    rec['case'] = f'{kink}_{position}' if position != 'single' else kink
    return rec


def _triple(case_class: str, kink: str, threshold: float, build) -> list[dict]:
    # The same unit a cent below the threshold, exactly on it, and a cent
    # above. Which side of a threshold a return falls on is the whole content
    # of a kink test, and "exactly on" is where tie rules (the itemization
    # strict greater-than, a bracket's lower edge) actually bite.
    out = []
    for position, offset in (('below', -CENT), ('at', 0.0), ('above', CENT)):
        out.append(_label(build(threshold + offset), case_class, kink,
                          position))
    return out


#-------------------------------------------------------------------------------
# Case builders, one per kink class
#-------------------------------------------------------------------------------

def _ordinary_bracket_edges(law: dict) -> list[dict]:
    # Taxable income sitting on each ordinary bracket edge, for every filing
    # status. The bottom edge (zero) is skipped: a cent of negative taxable
    # income is not a return, it is a floor the calculator already applies.
    out = []
    for fs in FILING_STATUSES:
        for i, edge in enumerate(law['ord_brackets'][fs]):
            if edge == 0.0:
                continue

            def build(value, fs=fs):
                rec = _blank(fs)
                # All ordinary income, so the whole schedule is exercised and
                # no preferred-rate slice sits on top.
                rec['txbl_inc'] = value
                rec['wages_base1'] = value + law['std_ded'][fs]
                rec['agi'] = rec['wages_base1']
                rec['std_ded'] = law['std_ded'][fs]
                return rec

            out += _triple('ordinary', f'ord_bracket{i}_fs{fs}', edge, build)
    return out


def _preferred_bracket_edges(law: dict) -> list[dict]:
    # Qualified dividends stacked on top of taxable income at the 0-to-15 and
    # 15-to-20 percent edges. The preferred schedule is read against total
    # taxable income, so the unit's whole taxable income is the dividend.
    out = []
    for fs in FILING_STATUSES:
        for i, edge in enumerate(law['pref_brackets'][fs]):
            if edge == 0.0:
                continue

            def build(value, fs=fs):
                rec = _blank(fs)
                rec['div_pref'] = value
                rec['txbl_inc'] = value
                rec['agi'] = value + law['std_ded'][fs]
                rec['std_ded'] = law['std_ded'][fs]
                return rec

            out += _triple('preferred', f'pref_bracket{i}_fs{fs}', edge, build)
    return out


def _capital_gain_character_edges(law: dict) -> list[dict]:
    # Unrecaptured section 1250 gain and collectibles gain are taxed at
    # ordinary rates capped at 25 and 28 percent, stacked above ordinary
    # income. These probe the rate cap binding and the stacking order: 1250
    # fills the remaining taxable income first, collectibles after it.
    out = []
    fs = SINGLE
    edge = law['ord_brackets'][fs][3]  # where the ordinary rate passes 25 pct

    def build_1250(value):
        rec = _blank(fs)
        rec['txbl_inc'] = value + 40_000.0
        rec['kg_pref'] = 40_000.0
        rec['kg_1250'] = 40_000.0
        rec['wages_base1'] = value + law['std_ded'][fs]
        rec['agi'] = rec['wages_base1'] + 40_000.0
        rec['std_ded'] = law['std_ded'][fs]
        return rec

    out += _triple('preferred_character', 'unrecaptured_1250_rate_cap', edge,
                   build_1250)

    def build_collect(value):
        rec = build_1250(value)
        rec['kg_collect'] = 25_000.0
        rec['kg_pref'] = 65_000.0
        rec['txbl_inc'] = value + 65_000.0
        rec['agi'] = value + law['std_ded'][fs] + 65_000.0
        return rec

    out += _triple('preferred_character', 'collectibles_rate_cap', edge,
                   build_collect)
    return out


def _amt_kinks(law: dict) -> list[dict]:
    # Three AMT boundaries. For a unit that takes the standard deduction and
    # has no other preferences, AMT income equals AGI (the calculator adds the
    # standard deduction back after subtracting it), so AGI is the dial.
    out = []
    for fs in (SINGLE, MARRIED_JOINT):
        exempt = law['amt_exempt'][fs]
        po_thresh = law['amt_exempt_po_thresh'][fs]
        po_rate = law['amt_exempt_po_rate']

        def build(value, fs=fs):
            rec = _blank(fs)
            rec['wages_base1'] = value
            rec['agi'] = value
            rec['std_ded'] = law['std_ded'][fs]
            rec['txbl_inc'] = max(0.0, value - law['std_ded'][fs])
            return rec

        # Where the exemption starts phasing out at 50 cents per dollar.
        out += _triple('amt', f'amt_exempt_po_start_fs{fs}', po_thresh, build)

        # Where the exemption has been fully phased out, so a further dollar
        # of AMT income no longer carries the phase-out's rate add-on.
        out += _triple('amt', f'amt_exempt_exhausted_fs{fs}',
                       po_thresh + exempt / po_rate, build)

        # The 26-to-28 percent AMT bracket edge, read against AMT income after
        # the exemption. Below the phase-out threshold the exemption is full,
        # so AGI at the edge plus the exemption puts AMT taxable income
        # exactly on it.
        out += _triple('amt', f'amt_bracket1_fs{fs}',
                       law['amt_brackets'][fs][1] + exempt, build)
    return out


def _payroll_kinks(law: dict) -> list[dict]:
    # The Social Security wage base, the Additional Medicare Tax threshold,
    # and the self-employment reporting floor.
    out = []
    base = law['ss_wage_base']

    def wage_only(value, fs=SINGLE):
        rec = _blank(fs)
        rec['wages_base1'] = value
        rec['agi'] = value
        rec['std_ded'] = law['std_ded'][fs]
        rec['txbl_inc'] = max(0.0, value - law['std_ded'][fs])
        return rec

    # A dollar of wages above the base owes Medicare but no OASDI.
    out += _triple('payroll', 'ss_wage_base_single_earner', base, wage_only)

    for fs in (SINGLE, MARRIED_JOINT):
        def build_add_med(value, fs=fs):
            rec = _blank(fs)
            # The Additional Medicare Tax is assessed on combined earnings of
            # both spouses, employee side only; splitting the earnings across
            # the couple checks that the combination happens.
            if fs == MARRIED_JOINT:
                rec['wages_base1'] = value * 0.6
                rec['wages_base2'] = value - rec['wages_base1']
            else:
                rec['wages_base1'] = value
            rec['agi'] = value
            rec['std_ded'] = law['std_ded'][fs]
            rec['txbl_inc'] = max(0.0, value - law['std_ded'][fs])
            return rec

        out += _triple('payroll', f'add_med_thresh_fs{fs}',
                       law['add_med_thresh'][fs], build_add_med)

    # Below the reporting threshold self-employment earnings owe no SECA at
    # all; a cent above it the whole amount becomes taxable, not just the
    # excess. This is the one genuine cliff in the payroll code.
    def build_se_floor(value):
        rec = _blank(SINGLE)
        rec['sole_prop1'] = value
        rec['agi'] = value
        rec['std_ded'] = law['std_ded'][SINGLE]
        return rec

    out += _triple('payroll', 'seca_reporting_floor', law['se_thresh'],
                   build_se_floor)
    return out


def _seca_fica_mixed(law: dict) -> list[dict]:
    # Earners with both wages and self-employment income. SECA stacks on top
    # of wages: the wage base is filled by wages first, and only what is left
    # of the base is available for taxable self-employment earnings.
    out = []
    base = law['ss_wage_base']
    seca_rate = law['seca_taxable_rate']
    wages = base - 40_000.0

    def build_straddle(total_oasdi_base):
        # Self-employment earnings scaled so wages plus taxable SE earnings
        # land exactly on the wage base.
        rec = _blank(SINGLE)
        rec['wages_base1'] = wages
        rec['sole_prop1'] = (total_oasdi_base - wages) / seca_rate
        rec['agi'] = rec['wages_base1'] + rec['sole_prop1']
        rec['std_ded'] = law['std_ded'][SINGLE]
        rec['txbl_inc'] = max(0.0, rec['agi'] - rec['std_ded'])
        return rec

    out += _triple('mixed_earner', 'seca_stacks_onto_wage_base', base,
                   build_straddle)

    # Wages already over the base: the whole SECA OASDI band is used up, so
    # self-employment earnings owe Medicare only.
    rec = _blank(SINGLE)
    rec['wages_base1'] = base + 25_000.0
    rec['sole_prop1'] = 60_000.0
    rec['agi'] = rec['wages_base1'] + rec['sole_prop1']
    rec['std_ded'] = law['std_ded'][SINGLE]
    rec['txbl_inc'] = rec['agi'] - rec['std_ded']
    out.append(_label(rec, 'mixed_earner', 'wages_over_base_plus_se',
                      'single'))

    # Both spouses mixed, each individually below the base but jointly over
    # it: the wage base is a per-person limit, the Additional Medicare Tax
    # threshold is not.
    rec = _blank(MARRIED_JOINT)
    rec['wages_base1'] = base - 20_000.0
    rec['wages_base2'] = base - 30_000.0
    rec['part_se1'] = 50_000.0
    rec['farm2'] = 40_000.0
    rec['agi'] = (rec['wages_base1'] + rec['wages_base2'] + rec['part_se1'] +
                  rec['farm2'])
    rec['std_ded'] = law['std_ded'][MARRIED_JOINT]
    rec['txbl_inc'] = rec['agi'] - rec['std_ded']
    out.append(_label(rec, 'mixed_earner', 'both_spouses_mixed_earners',
                      'single'))
    return out


def _eitc_kinks(law: dict) -> list[dict]:
    # The EITC's three kinks at each child count. Earnings are the dial and
    # AGI is held equal to earnings, because the phase-out runs on whichever
    # of the two is larger.
    out = []
    for n_kids in (0, 1, 2, 3):
        pi_rate = law['eitc_pi_rate'][n_kids]
        pi_end = law['eitc_pi_end'][n_kids]
        po_rate = law['eitc_po_rate'][n_kids]
        for fs in (SINGLE if n_kids == 0 else HEAD_OF_HOUSEHOLD,
                   MARRIED_JOINT):
            po_thresh = law['eitc_po_thresh'][n_kids]
            if fs == MARRIED_JOINT:
                po_thresh += law['eitc_po_thresh_mfj_bump']

            def build(value, fs=fs, n_kids=n_kids):
                rec = _blank(fs)
                rec['wages_base1'] = value
                rec['agi'] = value
                rec['std_ded'] = law['std_ded'][fs]
                rec['txbl_inc'] = max(0.0, value - rec['std_ded'])
                # Childless claimants must be within the age range; 40 is.
                _add_children(rec, n_kids)
                return rec

            tag = f'{n_kids}kids_fs{fs}'
            # Phase-in end: the last dollar of earnings that raises the credit.
            out += _triple('eitc', f'eitc_phasein_end_{tag}', pi_end, build)
            # Plateau end: a dollar more of earnings starts reducing the
            # credit at the phase-out rate.
            out += _triple('eitc', f'eitc_plateau_end_{tag}', po_thresh, build)
            # Phase-out end: the credit reaches zero and stays there.
            out += _triple('eitc', f'eitc_phaseout_end_{tag}',
                           po_thresh + pi_end * pi_rate / po_rate, build)

    # The investment income test is a cliff, not a phase-out: a cent of
    # interest over the limit disqualifies the whole credit. The limit itself
    # is a placeholder here since it is not in the threshold table above.
    inv_limit = 11_950.0  # placeholder EITC investment income limit

    def build_inv(value):
        rec = _blank(HEAD_OF_HOUSEHOLD)
        rec['wages_base1'] = 20_000.0
        rec['txbl_int'] = value
        rec['agi'] = rec['wages_base1'] + value
        rec['std_ded'] = law['std_ded'][HEAD_OF_HOUSEHOLD]
        rec['txbl_inc'] = max(0.0, rec['agi'] - rec['std_ded'])
        _add_children(rec, 2)
        return rec

    out += _triple('eitc', 'eitc_investment_income_limit', inv_limit,
                   build_inv)
    return out


def _ctc_kinks(law: dict) -> list[dict]:
    # The CTC phase-out staircase and the refundable credit's two boundaries.
    out = []
    step = law['ctc_po_discrete_step']
    for fs in (HEAD_OF_HOUSEHOLD, MARRIED_JOINT):
        po_thresh = law['ctc_po_thresh'][fs]

        def build(value, fs=fs):
            rec = _blank(fs)
            rec['wages_base1'] = value
            rec['agi'] = value
            rec['std_ded'] = law['std_ded'][fs]
            rec['txbl_inc'] = max(0.0, value - rec['std_ded'])
            _add_children(rec, 2)
            return rec

        # Each $1,000 of AGI above the threshold takes another $50 off the
        # credit, and a single cent into a new $1,000 band takes the whole $50.
        # Three consecutive steps show the staircase, not just its first riser.
        for n_step in (0, 1, 2):
            out += _triple('ctc', f'ctc_po_step{n_step}_fs{fs}',
                           po_thresh + n_step * step, build)

    # The refundable credit starts at the earnings threshold: below it a
    # family with no tax liability gets nothing.
    def build_ei(value, n_kids=2):
        rec = _blank(HEAD_OF_HOUSEHOLD)
        rec['wages_base1'] = value
        rec['agi'] = value
        rec['std_ded'] = law['std_ded'][HEAD_OF_HOUSEHOLD]
        rec['txbl_inc'] = max(0.0, value - rec['std_ded'])
        _add_children(rec, n_kids)
        return rec

    out += _triple('ctc', 'ctc_refundable_earnings_thresh',
                   law['ctc_pi_thresh'], build_ei)

    # Where 15 cents per dollar of earnings above the threshold reaches the
    # per-child refundable cap: above this point more earnings no longer buy
    # more refundable credit. Two child counts, because the cap is per child.
    for n_kids in (1, 2):
        def build_cap(value, n_kids=n_kids):
            return build_ei(value, n_kids)

        cap_at = (law['ctc_pi_thresh'] +
                  n_kids * law['ctc_max_refund'] / law['ctc_pi_rate'])
        out += _triple('ctc', f'ctc_refundable_cap_{n_kids}kids', cap_at,
                       build_cap)

    # A child with no SSN is not CTC-qualifying under current law, though the
    # family still claims the dependent. Paired with an otherwise identical
    # unit so the difference is attributable to the SSN alone.
    for has_ssn in (True, False):
        rec = _blank(HEAD_OF_HOUSEHOLD)
        rec['wages_base1'] = 45_000.0
        rec['agi'] = 45_000.0
        rec['std_ded'] = law['std_ded'][HEAD_OF_HOUSEHOLD]
        rec['txbl_inc'] = rec['agi'] - rec['std_ded']
        _add_children(rec, 2, has_ssn=has_ssn)
        out.append(_label(rec, 'ctc',
                          'ctc_child_ssn_' + ('present' if has_ssn else
                                              'absent'), 'single'))
    return out


def _itemization_tie(law: dict) -> list[dict]:
    # Itemized deductions exactly equal to the standard deduction. The R model
    # itemizes only on a strict greater-than, so the tie must take the
    # standard deduction; the two neighbours a cent away pin the comparison
    # down on both sides.
    out = []
    for fs in (SINGLE, MARRIED_JOINT):
        std = law['std_ded'][fs]

        def build(value, fs=fs, std=std):
            rec = _blank(fs)
            rec['wages_base1'] = 120_000.0
            rec['agi'] = 120_000.0
            rec['std_ded'] = std
            rec['item_ded'] = value
            rec['item_ded_ex_limits'] = value
            # All of it state and local tax, the component the OBBBA
            # clawback stacks first.
            rec['salt_item_ded'] = value
            # The choice the R rule implies: standard unless itemized are
            # strictly larger. The calculator must reproduce this.
            rec['itemizing'] = 1.0 if value > std else 0.0
            rec['txbl_inc'] = rec['agi'] - max(std, value)
            return rec

        out += _triple('itemization', f'itemize_tie_fs{fs}', std, build)
    return out


def _dependent_corners(law: dict) -> list[dict]:
    # Dependent counts at their corners, and the places where "dependent",
    # "EITC qualifying child" and "CTC qualifying child" come apart.
    out = []

    def base(fs=HEAD_OF_HOUSEHOLD, wages=48_000.0):
        rec = _blank(fs)
        rec['wages_base1'] = wages
        rec['agi'] = wages
        rec['std_ded'] = law['std_ded'][fs]
        rec['txbl_inc'] = max(0.0, wages - rec['std_ded'])
        return rec

    # No dependents.
    out.append(_label(base(SINGLE), 'dependents', 'dep_count_zero', 'single'))

    # One dependent, then the three-slot count, then more dependents than the
    # model carries attributes for.
    for n_kids, name in ((1, 'dep_count_one'), (3, 'dep_count_three'),
                         (5, 'dep_count_many')):
        rec = base()
        _add_children(rec, n_kids)
        out.append(_label(rec, 'dependents', name, 'single'))

    # A 19-year-old dependent: claimed on the return, too old for the CTC,
    # and too old to be an EITC qualifying child absent student status.
    rec = base()
    _add_children(rec, 1, age=19.0, ctc_qualifying=False)
    rec['n_dep_eitc'] = 0.0
    out.append(_label(rec, 'dependents', 'dep_older_than_ctc_age', 'single'))

    # A child who is CTC-qualifying but whom the EITC does not count (say a
    # child who fails the residency test): the two counts differ on the same
    # return.
    rec = base()
    _add_children(rec, 2)
    rec['n_dep_eitc'] = 1.0
    out.append(_label(rec, 'dependents', 'dep_ctc_and_eitc_counts_differ',
                      'single'))

    # A dependent filing her own return: no EITC, no CTC of her own.
    rec = base(SINGLE, wages=9_000.0)
    rec['dep_status'] = 1.0
    out.append(_label(rec, 'dependents', 'dep_status_filer', 'single'))

    # A child under one, which reform proposals single out for a bonus.
    rec = base()
    _add_children(rec, 1, age=0.0)
    out.append(_label(rec, 'dependents', 'dep_age_zero', 'single'))
    return out


def _zero_and_negative_agi(law: dict) -> list[dict]:
    # Returns at and below zero AGI. Business losses can put AGI well below
    # zero while wages and payroll tax stay positive, and several provisions
    # compare against AGI without flooring it first.
    out = []

    def build(value):
        rec = _blank(SINGLE)
        rec['agi'] = value
        rec['std_ded'] = law['std_ded'][SINGLE]
        rec['txbl_inc'] = max(0.0, value - rec['std_ded'])
        return rec

    out += _triple('extreme_agi', 'agi_zero', 0.0, build)

    # Wages against a large sole proprietorship loss: earnings for the EITC
    # and the refundable CTC are positive while AGI is negative.
    rec = _blank(HEAD_OF_HOUSEHOLD)
    rec['wages_base1'] = 30_000.0
    rec['sole_prop1'] = -95_000.0
    rec['agi'] = rec['wages_base1'] + rec['sole_prop1']
    rec['std_ded'] = law['std_ded'][HEAD_OF_HOUSEHOLD]
    rec['txbl_inc'] = 0.0
    _add_children(rec, 2)
    out.append(_label(rec, 'extreme_agi', 'agi_negative_business_loss',
                      'single'))

    # A capital loss at the annual limit, with no carryforward modeled.
    rec = _blank(SINGLE)
    rec['wages_base1'] = 2_000.0
    rec['txbl_kg'] = -3_000.0  # placeholder annual capital loss limit
    rec['agi'] = rec['wages_base1'] + rec['txbl_kg']
    rec['std_ded'] = law['std_ded'][SINGLE]
    out.append(_label(rec, 'extreme_agi', 'agi_negative_capital_loss',
                      'single'))

    # Nothing at all on the return: a filer with zero of everything.
    out.append(_label(_blank(SINGLE), 'extreme_agi', 'agi_all_zero_return',
                      'single'))
    return out


def _ss_taxability_kinks(law: dict) -> list[dict]:
    # Combined income — other income plus tax-exempt interest plus half of
    # benefits — at the two thresholds. Below the first, none of the benefit
    # is taxed; between them, up to half; above the second, up to 85 percent.
    out = []
    benefits = 24_000.0
    for fs in (SINGLE, MARRIED_JOINT):
        for tier, key in ((1, 'ss_tier1_thresh'), (2, 'ss_tier2_thresh')):
            def build(value, fs=fs, benefits=benefits):
                rec = _blank(fs)
                rec['gross_ss'] = benefits
                # Other income set so combined income lands on the threshold.
                rec['wages_base1'] = max(0.0, value - 0.5 * benefits)
                rec['agi'] = rec['wages_base1']
                rec['std_ded'] = law['std_ded'][fs]
                rec['txbl_inc'] = max(0.0, rec['agi'] - rec['std_ded'])
                return rec

            out += _triple('ss_taxability',
                           f'ss_taxability_tier{tier}_fs{fs}', law[key][fs],
                           build)

    # Tax-exempt interest counts toward combined income even though it never
    # enters AGI: the same unit with the same AGI, one holding municipal
    # bonds, crosses the first threshold and the other does not.
    for exempt in (0.0, 15_000.0):
        rec = _blank(SINGLE)
        rec['gross_ss'] = benefits
        rec['wages_base1'] = 12_000.0
        rec['exempt_int'] = exempt
        rec['agi'] = rec['wages_base1']
        rec['std_ded'] = law['std_ded'][SINGLE]
        out.append(_label(rec, 'ss_taxability',
                          'ss_exempt_int_in_combined_income_' +
                          ('with' if exempt else 'without'), 'single'))
    return out


def _estate_kinks(law: dict) -> list[dict]:
    # Estates on the exemption, on the graduated schedule's edges, and on the
    # SOI bin edges where the measurement parameters change.
    out = []
    valuation_r = law['estate_valuation_r']

    def build_decedent(reported_gross, fs=SINGLE, married_death=False):
        rec = _blank(MARRIED_JOINT if married_death else fs)
        rec['economic_gross'] = reported_gross / valuation_r
        rec['q_death1'] = 1.0
        rec['q_death2'] = 1.0 if married_death else 0.0
        return rec

    # The exemption: one for a single decedent, two when both spouses' deaths
    # put the estate in the joint state.
    out += _triple('estate', 'estate_exemption_single',
                   law['estate_exemption'], build_decedent)
    out += _triple('estate', 'estate_exemption_joint',
                   2 * law['estate_exemption'],
                   lambda v: build_decedent(v, married_death=True))

    # The top edge of the graduated schedule, where the marginal rate reaches
    # its maximum. Below the exemption this is absorbed by the unified credit,
    # but it still moves the tentative tax the credit is netted against.
    top_edge = law['estate_brackets'][-1]
    out += _triple('estate', 'estate_top_rate_edge',
                   law['estate_exemption'] + top_edge, build_decedent)

    # The SOI size bins: a cent either side of an edge picks up a different
    # deduction share, portability probability and DSUE amount.
    for i, edge in enumerate(law['estate_soi_bins']):
        if edge == 0.0:
            continue
        out += _triple('estate', f'estate_soi_bin{i}_edge', edge,
                       build_decedent)

    # An estate whose wealth is mostly closely held business interests: the
    # valuation bridge marks pass-throughs differently from listed assets.
    rec = build_decedent(30_000_000.0)
    rec['value_pass_throughs'] = rec['economic_gross'] * 0.8
    rec['estate_debts'] = 4_000_000.0
    out.append(_label(rec, 'estate', 'estate_pass_through_heavy', 'single'))
    return out


def _donor_clones(law: dict) -> list[dict]:
    # Identical records under different ids and weights. The frame's estate
    # mortality step groups donor clones on rounded economic wealth and caps
    # the cluster, so a cluster of exactly-equal estates is the case that
    # exercises it; the weights differ to catch anything that silently
    # assumes clones share a weight.
    out = []
    clone_weights = (100.0, 100.0, 250.0, 250.0, 25.0)
    for i, weight in enumerate(clone_weights):
        rec = _blank(SINGLE)
        rec['economic_gross'] = 22_000_000.0
        rec['q_death1'] = 1.0
        rec['weight'] = weight
        out.append(_label(rec, 'donor_clone', f'estate_donor_clone_{i}',
                          'single'))

    # A pair of identical income returns, same everything but id and weight:
    # any per-record calculation must give them the same answer, and any
    # weighted total must count them separately.
    for i, weight in enumerate((60.0, 40.0)):
        rec = _blank(MARRIED_JOINT)
        rec['wages_base1'] = 85_000.0
        rec['wages_base2'] = 35_000.0
        rec['agi'] = 120_000.0
        rec['std_ded'] = law['std_ded'][MARRIED_JOINT]
        rec['txbl_inc'] = rec['agi'] - rec['std_ded']
        _add_children(rec, 2)
        rec['weight'] = weight
        out.append(_label(rec, 'donor_clone', f'income_clone_{i}', 'single'))
    return out


def _panel_presence(year: int, law: dict) -> list[dict]:
    # Units that are not in every year. The id universe is the union across
    # run years, so a unit missing from a year must be masked out rather than
    # zero-filled — these are the fixtures the panel-shape tests read.
    out = []
    first_year, last_year = YEARS[0], YEARS[-1]

    def worker(wages):
        rec = _blank(SINGLE)
        rec['wages_base1'] = wages
        rec['agi'] = wages
        rec['std_ded'] = law['std_ded'][SINGLE]
        rec['txbl_inc'] = max(0.0, wages - rec['std_ded'])
        return rec

    # Present in the first year only: a filer who stops filing.
    if year == first_year:
        rec = worker(52_000.0)
        rec['id'] = _PANEL_ID_BASE + 0
        out.append(_label(rec, 'panel', 'panel_exits_after_first_year',
                          'single'))

    # Present in the last year only: a new filer.
    if year == last_year:
        rec = worker(31_000.0)
        rec['id'] = _PANEL_ID_BASE + 1
        out.append(_label(rec, 'panel', 'panel_enters_in_last_year',
                          'single'))

    # Present in every year, with income that changes: the control against
    # which the entering and exiting units are read.
    rec = worker(70_000.0 + 5_000.0 * (year - first_year))
    rec['id'] = _PANEL_ID_BASE + 2
    out.append(_label(rec, 'panel', 'panel_present_all_years', 'single'))

    # Present in every year with a filing status change between them, which
    # moves every threshold the unit is compared against.
    rec = _blank(SINGLE if year == first_year else MARRIED_JOINT)
    fs = int(rec['filing_status'])
    rec['wages_base1'] = 64_000.0
    rec['agi'] = 64_000.0
    rec['std_ded'] = law['std_ded'][fs]
    rec['txbl_inc'] = rec['agi'] - rec['std_ded']
    rec['id'] = _PANEL_ID_BASE + 3
    out.append(_label(rec, 'panel', 'panel_marries_between_years', 'single'))
    return out


#-------------------------------------------------------------------------------
# Assembly
#-------------------------------------------------------------------------------

# Each case class gets its own block of ids so that a unit keeps the same id
# in every year even when another class changes length. Blocks are wide enough
# that no class can run into the next one.
_ID_BLOCK = 10_000
_PANEL_ID_BASE = 900_000

_BUILDERS = (
    ('ordinary', 1, _ordinary_bracket_edges),
    ('preferred', 2, _preferred_bracket_edges),
    ('preferred_character', 3, _capital_gain_character_edges),
    ('amt', 4, _amt_kinks),
    ('payroll', 5, _payroll_kinks),
    ('mixed_earner', 6, _seca_fica_mixed),
    ('eitc', 7, _eitc_kinks),
    ('ctc', 8, _ctc_kinks),
    ('itemization', 9, _itemization_tie),
    ('dependents', 10, _dependent_corners),
    ('extreme_agi', 11, _zero_and_negative_agi),
    ('ss_taxability', 12, _ss_taxability_kinks),
    ('estate', 13, _estate_kinks),
    ('donor_clone', 14, _donor_clones),
)

CASE_CLASSES = tuple(sorted({name for name, _, _ in _BUILDERS} | {'panel'}))


def generate(year: int, law: dict | None = None) -> pd.DataFrame:
    """One year of the synthetic battery.

    `law` replaces placeholder thresholds by name — pass, say,
    `{'ss_wage_base': 176400.0}` and the payroll cases move to that base.
    Anything not named keeps its placeholder value.
    """
    params = placeholder_law(year)
    if law:
        params.update(copy.deepcopy(law))

    records: list[dict] = []
    for _, block, builder in _BUILDERS:
        built = builder(params)
        for offset, rec in enumerate(built):
            rec.setdefault('id', block * _ID_BLOCK + offset)
        records += built
    records += _panel_presence(year, params)

    for rec in records:
        rec['year'] = year
        rec.setdefault('weight', DEFAULT_WEIGHT)

    return pd.DataFrame(records, columns=list(COLUMNS))


def generate_all(law: dict | None = None) -> pd.DataFrame:
    """The full battery over every fixture year, stacked."""
    return pd.concat([generate(year, law) for year in YEARS],
                     ignore_index=True)


if __name__ == '__main__':
    frame = generate_all()
    print(f'{len(frame)} synthetic units over years {YEARS}')
    print(frame.groupby(['case_class', 'year']).size().to_string())
