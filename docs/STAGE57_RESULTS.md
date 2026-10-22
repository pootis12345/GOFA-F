# Stage57: actual ChatTime graph-memory source results

The fixed source screening failed. The run completed normally; all own GPU/controller processes exited. The active real-graph zero-/few-shot performance goal remains unachieved. No source holdout or target evaluation was opened, and no second recipe was selected.

Variable-equal raw-unit MAE on the same16development queries/view,8samples/pointwise median:

|Source view|Native|Real graph|Self memory|Static permuted|Real vs Native|
|---|---:|---:|---:|---:|---:|
|PEMS04flow|31.668092|35.657872|37.064068|34.534603|+12.60%|
|PEMS04speed|5.645279|2.782140|3.830919|2.645191|−50.72%|
|Boston2017temperature|3.409426|4.017890|3.528405|3.764237|+17.85%|
|Penmanshiel2017power|199.147512|206.525769|205.903143|207.600578|+3.70%|

Real roads improve traffic relative to the separately trained self-memory control, but are worse than the separately trained static-permuted graph in both views:flow+3.25%,speed+5.18%. All traffic real/control variable-cluster95%intervals cross1. Speed improves on only8/16individual queries and much of its mean gain comes from one time block; it is not evidence of uniform or stable road-specific improvement.15predeclared checks failed, including time-block guards and weather/wind Native protection. Keep the original gate unchanged.

There is a useful diagnostic lead: flow's first-step MAE improves17.529275→14.857949, while later-step MAE worsens32.953439→37.548774. Speed also has a stronger first-step gain than later-step gain. Weather/wind are already worse at the first step, so prefix drift alone cannot explain all views. This motivates paired source conditional-distribution analysis, not an immediate coefficient/loss change.

All inverse transforms were finite. Real-graph complete constant forecasts are0/16flow,0/16speed,2/16weather,0/16wind. Endpoint fraction is0except speed3.65%. Source representable-range minimum-MAE bounds are6.834/0.524/0.045/11.834respectively, well below observed errors; range limitations alone do not explain the failures. First/last training CE logs use different query identities and cannot prove paired CE improvement or a CE/MAE mismatch.

Three fresh controls completed100updates each with the same initial tensor hash and exact query-order hash. Real/self/permuted training costs112.249/53.511/52.459seconds; frozen-prefix reuse accelerated later controls. Total GPU activity1683.192490seconds(28min03s), of which training218.219seconds; most cost is fixed sample8evaluation. Peak project14,960MiB, wait0/resource pauses0.40related CPU checks passed. Independent CPU audit0.244seconds verified all2048raw trajectories, seeds, query medians, checkpoint hashes and raw-unit MAE; metric replay max difference0. Original results, failed gates and costs are retained.

Runtime source71371b1e3d20d7fa3ee46e43b5d33c518a88f2fd; manifest e7d6e4e59c9a617a5b2cbab35a70189d3c42ec19a1495dbf4c49f743d8288343. Private evidence archive SHA25689ba72771d2242fd75aca74801027d492506d496a4ab3d730f01d73687f968a1. Detailed sanitized curves, coefficients/checkpoint hashes, paired intervals and failed checks:results/stage57-graph-memory-source-final-summary.json.

Next registered action: the fixed16-query source diagnostic in STAGE57_DIAGNOSTIC_PROTOCOL.md. No more source training or target work before interpreting that evidence. The UFGTime compatibility component is implemented, but no new recent-GNN forecast or ranking is claimed.
