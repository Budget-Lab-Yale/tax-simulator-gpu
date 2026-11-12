# Index series from Macro-Projections

The columns `year`, `cpiu_irs`, `ccpiu_irs`, `ccpiu_irs_sept` and `awi` of
Macro-Projections v3, vintage 2026110912, scenario `baseline`: the series the
law compiler indexes dollar amounts with. The batch command reads this copy
when given `--bundled-macro`, so that it runs away from the cluster.

The copy must match the vintage pinned in
`config/scenarios/economy/default/interfaces.yaml`. A test in
`tests/unit/run/test_batch.py` fails on the cluster when the two differ. To
refresh it, copy the same five columns of `historical.csv` and
`projections.csv` from the newly pinned publication and change the vintage
above.
