"""Behavioral response modules: how households change what they report when
tax prices change.

These channels run only in the conventional world, so the conventional world
minus the mechanical world is the behavioral effect.

A module is a function with this signature:

    changed_columns = module(records, prices_reform, prices_baseline,
                             params, draws)

It reads the year's household columns and the marginal tax rates computed
under reform law and under baseline law on those same households, and returns
a dictionary holding only the columns it changed. The year writes those
columns onto the frame before the next module runs. Alongside each function
sits a ``MANIFEST`` dictionary declaring what the module reads, writes, prices
and draws, and a defaults dictionary naming the source of every parameter
value.

Each subpackage holds one margin: ``kg`` is the capital-gains realization
solver, ``gains`` is the capital-gains behavior module built on it (its
settings, and the pieces that run inside the compiled year; the host-side
solve between the year loop's two executions is `years/gains_solve.py`),
``conversion`` is salary relabeled as capital gain, ``entity_shifting`` is
business income moving between corporate and pass-through form, and the rest
are the single-file modules (tipped pay, overtime, charity, noncompliance,
auto-loan interest, wealth-tax and estate-tax avoidance).

The corporate tax scorer and its incidence paths are a mechanical channel and
sit with the year loop in `years/corporate/`.

``loader`` turns a scenario's list of modules into the single function the
simulation year calls: it checks each module's declaration, orders the
families so that a module reading a column runs after the module that writes
it, and composes them.
"""
