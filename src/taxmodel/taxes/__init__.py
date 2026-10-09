"""Calculating taxes for one year of households under one year of law.

The worlds of a year (`years/worlds.py`) call `statutory_order.py` on each
household frame they build; the behavior modules read the marginal rates
priced here.

- `calc/`                    — the tax calculators, one file per part of the
                               return (AGI, deductions, credits, the estate
                               tax, and so on)
- `statutory_order.py`       — the calculators called in the order the law
                               applies them: records and law in, results out
- `chain_columns.py`         — what each step returns, and what a run's
                               records must carry
- `law_for_calculators.py`   — one year of compiled law turned into the
                               objects each calculator takes
- `marginal_rates.py`        — the marginal tax rates a run stores per household
- `individual_income_tax.py` — marginal tax rates by differentiating the
                               calculators, or by adding a dollar and rerunning
"""
