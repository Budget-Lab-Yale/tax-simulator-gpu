# Wealth-dynamics financing profiles

A financing profile is the per-scenario input to the wealth bathtub
(`src/taxmodel/years/wealth_arithmetic.py`): a bracket-varying saving share `s(age, net-worth
percentile)` and a within-age percentile transition matrix `M`. Each profile is a
folder under `config/calibrations/wealth_profiles/` holding two files.

These are operational inputs rather than tax law, and a reform must never
override them.

## Files

| File    | Format | Meaning |
|---------|--------|---------|
| `s.csv` | header `age,nw_pctile,s`; one row per cell | saving share `s = 1 − MPC ∈ [0, 1]`. Must cover every cell of the `18..80` (age) × `1..n_pctiles` grid exactly once; the loader stops on a gap, a duplicate, or an out-of-range value. |
| `M.csv` | headerless `n_pctiles × n_pctiles` grid | within-age percentile transition, applied to every age and raked to doubly-stochastic on load. The identity is full persistence, and a uniform `1/n` is extreme diffusion. Absent means the identity, which is what the shipped profiles rely on. (The R model also accepted a per-age `M.rds`; that was never ported.) |

`age` is `pmax(age1, age2)` for joint records, topcoded to `[18, 80]`.
`nw_pctile` is the within-age net-worth percentile bin, over positive net worth
only, matching the cells the pre-pass and the applier rank into.

## How a scenario selects a profile

An economy alternative sets `wealth.financing_profile`, which
`wealth_arithmetic.resolve_profile()` reads. It takes one of three forms:

1. `none` or `off` forces the channel off.
2. A profile folder name selects that profile.
3. `flat:<s>` is shorthand for a constant share everywhere and an identity `M`.
   `flat:0` is a deliberate way to switch the channel off.

A scenario that overrides nothing gets the `default` profile.

The channel is active when the resolved surface has any positive cell, so a
flat-zero profile does nothing and skips the roughly doubled split-pass compute.

## Shipped profiles

- **`default/`** — the calibrated saving surface with an identity `M`, applied to
  every scenario that does not override the setting, along with the extra compute
  that entails. The share is one less the marginal propensity to consume out of a
  persistent net income change, anchored to Straub (2019), Mian–Straub–Sufi
  (2021), Dynan–Skinner–Zeldes (2004), Fagereng–Holm–Natvik (2021) and
  De Nardi–French–Jones (2010). It rises from about 0.1 at the bottom of
  within-age net worth to 0.80 at the top percentile, hump-shaped in age with the
  tilt attenuated to zero at top ranks. The sourcing memo and the generator
  that wrote these files are in the R model repository, under
  `other/wealth_dynamics/`; neither was ported. The two CSVs here are the
  generator's output, checked in.

## Regenerating

There is no generator in this repository. The shipped profiles were written
by `other/wealth_dynamics/write_profiles.py` in the R model repository and
are checked in here as data.
