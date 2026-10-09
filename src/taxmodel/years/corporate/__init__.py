"""The corporate tax: incidence of a change, revenue from a rate change, and
the allocation of the level across households.

The corporate pieces are mechanical transmission channels. They run in the
mechanical worlds alongside the employer-payroll adjustment, and nothing here
responds to a tax price.

The four modules:

- ``paths`` holds the incidence path, the year-by-year arithmetic of how a
  corporate tax change reaches household records, cutting capital income flows
  and marking equity down.
- ``scorer`` scores a change in the corporate statutory rate against CBO's
  projected receipts.
- ``burden_jax`` spreads a national corporate-tax dollar figure across
  households for the distribution table, inside the compiled projection
  year: the three legs of the allocation and the constants they use. It does
  not touch simulation records.
- ``channel`` connects the scorer and the paths to a scenario: it reads the
  statutory rate off reform and baseline law, scores the change, and puts the
  resulting year-by-year factors where the run will find them.
"""
