"""The year loop: one scenario's projection years, each a set of worlds.

A run works one year at a time. `loop.run_scenarios` compiles one projection
year once, with a scenario axis so that a group of scenarios runs side by
side, and calls it once per year. `one_year.one_year_step` is that year for
one scenario: `worlds.annual_step` runs the five worlds in order (first_order,
mechanical_ex_saving, mechanical, conventional, then the advance of the
wealth stocks), and the year step passes the worlds' columns to
`reporting/device_sums.py` for the sums a run reports.

- `loop.py`               — the loop over years and over groups of scenarios
- `device.py`             — device placement, compiled programs, and memory
- `one_year.py`           — one projection year of one scenario
- `worlds.py`             — the five worlds of a year, in evaluation order
- `world_names.py`        — the world names; results keyed by frame and law
- `income_adjustments.py` — the mechanical income side effects of a reform
- `corporate/`            — the corporate tax: revenue from a rate change,
                            its incidence, and its place in a run
- `wealth_state.py`       — the wealth stocks carried year to year
- `wealth_arithmetic.py`  — the wealth bathtub arithmetic
- `wealth_history.py`     — the bathtub's per-year quantities for the
                            diagnostics tables
- `burden.py`             — the finished tax change that the bathtub drains
- `gains_solve.py`        — the host-side capital-gains solve beside the loop
- `stacking.py`           — several scenarios' or years' copies side by side
- `record_columns.py`     — reading dollar columns off household records
"""
