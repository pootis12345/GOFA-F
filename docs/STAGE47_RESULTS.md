# Stage47 fixed road-distance prior results

Registered source gate passed: **False**. This is a source diagnostic, not a ChatTime or six-target result.

Both graph exchanges add the fixed inverse-hop attention prior, without new parameters. Correct and target-fixed permuted distance controls use identical physical rosters and inputs; the latter is not the earlier random-roster condition. No coefficient search. Stage40 dense reference remains sealed.

| Condition | Flow variable-equal MAE | Speed variable-equal MAE | Selected update |
|---|---:|---:|---:|
| Correct road distance | 24.403985684 | 1.482628279 | 1000 |
| Self only | 26.520034941 | 1.535240291 | 1000 |
| Target-fixed permuted distance | 24.270554390 | 1.508313059 | 1000 |

Original seed42 gate failures: pems04_flow:rewired:less_than_1pct_gain; pems04_flow:rewired:time_block_regression; pems04_speed:self:time_block_regression; pems04_speed:rewired:time_block_regression.
pems04_flow: correct/permuted MAE ratio 1.005497662; against Stage40 dense baseline +3.266%.
pems04_speed: correct/permuted MAE ratio 0.982971187; against Stage40 dense baseline -0.844%.

Acceptance:33 related CPU tests passed with inherited full proof. Same10831 parameters, initialization and query cursors; self best tensor exactly equals Stage40. All physical-roster cached inputs and original two edge channels unchanged; appended geometry is hashed and target distance multisets match. Independent metric reaggregation within1e-6 and bound verification passed. The synthetic two-channel profile explicitly uses neutral geometry; every scientific cache uses bound three-channel input.

Controller wall 102.252s; worker activity 96.539s is included, not additional. Closing verification 1.092s. GPU0. Full elapsed through summary generation 458.7s excludes final Git synchronization. All workers/controllers exited; hourly follow-up stays off.

Exposed PEMS04 source validation, small diagnostic continuous-output GNN and seed42 initial screen. A failed gate does not prove graph zero-shot impossible. Do not return to another distance strength or select another checkpoint after failure. No expensive ChatTime promotion without the registered source/seed/confirmation gates.

Initial gate failed: no seeds43/44, source confirmation, ChatTime updates or target forecasts. This single recipe is sealed.

Runtime `f8a8f1bcce1dc70f1b90978e0926a744c5a4d107`, report `719c682aa16e051c55462430b9c8a8aae48249fac7d510334476938de53398dc`, archive `65a7266d697086cf2b29aff2c9fee229817c1ae3da4d0c5db85b34e091d0f0d5`.
