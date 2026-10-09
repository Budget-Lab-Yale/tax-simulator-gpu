"""Preparing one scenario before its years are simulated.

The scenario loop (`run/scenarios.py`) calls this for every scenario in a
group and passes the result to the year loop (`years/loop.py`).

- `provenance.py`   — the scenario's economy, law, and behavior legs, their
                      vintages, and the files read off disk
- `law_by_year.py`  — every projection year of the scenario's law, bound once
- `prepare.py`      — the laws, calculators, and inputs the year loop needs
- `gains_inputs.py` — the capital-gains model's household inputs
"""
