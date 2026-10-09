"""The calculator chain's column manifests: what each step returns, and what a
run's records must carry.

`taxes/statutory_order.py` runs the calculators in statutory order and reads
these tuples to know which columns each step hands the next; the frame layer
reads `RECORD_COLUMNS` to know what a population must hold; the reporting layer
reads `CALCULATED_COLUMNS` to know what a run can write out.
`tests/unit/taxes/test_statutory_order.py` pins each output tuple to what its
function returns, so a new column cannot be added to a calculator without
appearing here. Everything here is also importable from
`taxes/statutory_order.py`, which is where callers read it.
"""

from __future__ import annotations

from taxmodel.taxes.calc.estate import ESTATE_ASSET_COLS, ESTATE_DEBT_COLS

# The itemized-deduction components the 1040 zeroes out for a taxpayer who
# ends up taking the standard deduction, so that reporting never shows a
# Schedule A amount for a non-itemizer.
ITEMIZED_COMPONENTS = (
    "med_item_ded",
    "salt_item_ded",
    "mort_int_item_ded",
    "inv_int_item_ded",
    "int_item_ded",
    "char_item_ded",
    "casualty_item_ded",
    "misc_item_ded",
    "other_item_ded",
    "item_ded_ex_limits",
    "item_ded",
)

# Everything `run_1040` produces. These are the columns the charitable double
# run chooses between, and the columns `run_taxes` passes on.
ONE_1040_COLUMNS = (
    "kg_pref",
    "txbl_kg",
    "txbl_ss",
    "excess_bus_loss",
    "char_above_ded",
    "auto_int_ded",
    "gross_inc",
    "above_ded",
    "agi",
    "std_ded",
    *ITEMIZED_COMPONENTS,
    "pe_ded",
    "qbi_ded",
    "tip_ded",
    "ot_ded",
    "senior_ded",
    "item_ded_limited",
    "itemizing",
    "ded",
    "txbl_inc",
    "liab_ord",
    "liab_pref",
    "liab_1250",
    "liab_collect",
    "liab",
    "alt_max_cap_binds",
    "alt_max_liab",
    "alt_max_rem",
    "alt_max_ord",
    "alt_max_pref",
    "alt_max_1250",
    "alt_max_coll",
    "amt_gross_inc",
    "amt_txbl_inc",
    "liab_amt",
    "liab_bc",
    "cdctc_nonref",
    "cdctc_ref",
    "ed_nonref",
    "ed_ref",
    "savers_nonref",
    "savers_ref",
    "caregiver_cred_nonref",
    "caregiver_cred_ref",
    "ctc_nonref",
    "ctc_ref",
    "become_filer_ctc",
    "eitc",
    "rebate",
    "become_filer_rebate",
    "wage_subsidy1",
    "wage_subsidy2",
    "liab_niit",
    "liab_surtax",
    "nonref",
    "ref",
    "ref_iit",
    "ref_other",
    "refund",
    "liab_iit",
    "liab_iit_net",
    "number_of_credits",
)

# The rest of what `run_taxes` produces, step by step, so the full list of
# columns the calculators create can be read without running them. A caller
# that wants per-household detail files has to be told which column names it
# may ask for before a run starts, and this is where that answer comes from.
# `tests/unit/taxes/test_statutory_order.py` pins each tuple to what its function
# actually returns, so a new column cannot be added without appearing here.

# `derive_vars`: the Schedule E aggregates.
DERIVED_COLUMNS = (
    "part",
    "scorp",
    "part_scorp",
    "part_scorp_loss",
    "pt",
    "net_rent",
    "net_estate",
    "sch_e",
)

# `taxmodel.taxes.calc.payroll.calc_pr`: earnings by earner and the payroll taxes.
PAYROLL_COLUMNS = (
    "gross_wages1",
    "gross_wages2",
    "gross_wages",
    "se1",
    "se2",
    "se",
    "ei1",
    "ei2",
    "ei",
    "liab_fica_er1",
    "liab_fica_er2",
    "liab_fica",
    "liab_seca",
    "liab_seca_er",
    "liab_oasdi",
    "liab_hi",
    "liab_add_med",
    "liab_pr_ee",
    "liab_pr_er",
    "liab_pr",
)

# `taxmodel.taxes.calc.estate.calc_estate`, which runs only when the run was given
# the estate measurement bridge. A run without it produces none of these.
ESTATE_COLUMNS = (
    "liab_estate_nodsue",
    "liab_estate_dsue",
    "estate_p_dsue",
    "estate_distributable",
)

# `taxmodel.taxes.calc.wealth.calc_wealth`: the wealth tax on net worth, zero under
# baseline law because baseline law has no wealth tax.
WEALTH_TAX_COLUMNS = ("liab_wealth",)

# `filer_and_expanded_income`. `filer` arrives from Tax-Data saying whether the
# household has a filing requirement; the model recomputes it here, because a
# household with no requirement still files when a refundable credit is worth
# money to it. The recomputed flag goes out under its own name,
# `filer_recomputed`, and Tax-Data's `filer` is left as it arrived.
# `corp_tax_change` comes out either way: the entity shifting module's own
# figure where that module ran, and zero where it did not.
FILER_AND_EXPANDED_INCOME_COLUMNS = (
    "filer_recomputed",
    "expanded_inc",
    "simple_filer",
    "corp_tax_change",
)

# `taxmodel.taxes.calc.remit.remit_taxes`: which part of each liability is withheld
# and which part is paid directly.
REMITTANCE_COLUMNS = (
    "pmt_iit_nonwithheld",
    "pmt_iit_withheld",
    "pmt_refund_nonwithheld",
    "pmt_refund_withheld",
    "pmt_pr_nonwithheld",
    "pmt_pr_withheld",
)

# Every column the calculator chain creates, in the order the steps run. A
# detail request may name any of these. Input columns are left out because a
# downstream consumer already has the inputs from Tax-Data and links to them by
# household id.
CALCULATED_COLUMNS = (
    *DERIVED_COLUMNS,
    *PAYROLL_COLUMNS,
    *ONE_1040_COLUMNS,
    *ESTATE_COLUMNS,
    *WEALTH_TAX_COLUMNS,
    *FILER_AND_EXPANDED_INCOME_COLUMNS,
    *REMITTANCE_COLUMNS,
)


# Every record column the chain reads and does not itself produce. The frame
# layer supplies these, and a caller can be told what it is missing before the
# chain runs.
RECORD_COLUMNS = (
    "filing_status",
    "filer",
    "dep_status",
    "age1",
    "age2",
    "blind1",
    "blind2",
    "n_dep",
    "n_dep_eitc",
    "dep_age1",
    "dep_age2",
    "dep_age3",
    "dep_ssn1",
    "dep_ssn2",
    "dep_ssn3",
    "dep_ctc1",
    "dep_ctc2",
    "dep_ctc3",
    # Wages, tips, overtime and self-employment, by earner. Nothing here
    # overlaps anything else: `wages_base1` is the first earner's pay that is
    # neither tips nor overtime, so that earner's W-2 wages are
    # wages_base1 + tips1 + ot1 and the household's wage line is the same sum
    # over both earners. `frame/prepare.py` splits the columns this way as
    # Tax-Data is read; a calculator that wants a total adds the pieces up.
    "wages_base1",
    "wages_base2",
    "tips1",
    "tips2",
    "tips_lh1",
    "tips_lh2",
    "ot1",
    "ot2",
    "trad_contr_er1",
    "trad_contr_er2",
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
    "part_se1",
    "part_se2",
    "ei_prior_yr",
    # Schedule E, before netting
    "part_active",
    "part_active_loss",
    "part_passive",
    "part_passive_loss",
    "part_179",
    "scorp_active",
    "scorp_active_loss",
    "scorp_passive",
    "scorp_passive_loss",
    "scorp_179",
    "rent",
    "rent_loss",
    "estate",
    "estate_loss",
    "wagebill_sole_prop",
    "wagebill_part",
    "wagebill_scorp",
    "wagebill_farm",
    "sstb_sole_prop",
    "sstb_part",
    "sstb_scorp",
    "sstb_farm",
    # Other income
    "txbl_int",
    "exempt_int",
    "div_ord",
    "div_pref",
    "state_ref",
    "txbl_ira_dist",
    "txbl_pens_dist",
    "gross_pens_dist",
    "gross_ss",
    "ui",
    "alimony",
    "divorce_year",
    "other_inc",
    "other_gains",
    # Capital gains
    "kg_st",
    "kg_lt",
    "kg_lt_basis",
    "kg_lt_years_held",
    "kg_lt_cpi_ratio",
    "kg_1250",
    "kg_collect",
    # Above-the-line deductions
    "ed_exp",
    "hsa_contr",
    "trad_contr_ira",
    "keogh_contr",
    "se_health",
    "early_penalty",
    "alimony_exp",
    "tuition_ded",
    "dpad",
    "other_above_ded",
    "sl_int_ded",
    "auto_int_exp",
    # Schedule A
    "med_exp",
    "salt_inc_sales",
    "salt_prop",
    "salt_pers",
    "first_mort_bal",
    "second_mort_bal",
    "first_mort_int",
    "second_mort_int",
    "inv_int_exp",
    "casualty_exp",
    "job_exp",
    "tax_prep_exp",
    "other_misc_exp",
    "other_item_exp",
    "char_cash",
    "char_noncash",
    # Credits and other taxes taken from the microdata rather than computed
    "ftc",
    "amt_ftc",
    "amt_nols",
    "excess_ptc",
    "net_ptc",
    "care_exp",
    "aoc_exp",
    "llc_exp",
    "savers_cred",
    "old_cred",
    "res_energy_cred",
    "gbc",
    "prior_yr_cred",
    "other_nonref",
    "recapture_tax",
    "ira_penalty",
    # From the frame steps
    "salt_workaround_part",
    "salt_workaround_scorp",
    # The draws the calculators consume, id-keyed in the frame
    "r.bus_loss",
    "r.cdctc_takeup",
    "r.new_car",
    "r.eitc_precert",
    "net_worth",
)

# What the estate tax needs on top, when it runs. The balance sheet columns
# are listed in `taxes/calc/estate.py`.
ESTATE_RECORD_COLUMNS = ("q_death2", *ESTATE_ASSET_COLS, *ESTATE_DEBT_COLS)
