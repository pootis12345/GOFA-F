# Stage50 fixed-budget convergence control results

Registered source gate passed: **False**. This is a new fixed-budget source experiment; prior stages remain sealed.

Use the original Stage40 normalized model and inputs, not Stage49 residual/ridge system. Exact1000-update last states were rebound to this experiment, preserving model/AdamW/RNG/torch RNG/query cursor/selection state; each control then received exactly2000 more updates, cap3000. Architecture/loss/lr/query/validation rule unchanged. Historical bests were inherited with provenance, so a selected update<=1000 is not a new model achievement.

| Condition | Flow variable-equal MAE | Speed variable-equal MAE | Selected update | Completed |
|---|---:|---:|---:|---:|
| Physical roster | 23.632124150 | 1.495254458 | 900 | 3000 |
| Self only | 25.237670688 | 1.518035275 | 2900 | 3000 |
| Matched-count random roster | 23.419834937 | 1.550143754 | 1300 | 3000 |

Original initial gate failures: pems04_flow:rewired:less_than_1pct_gain; pems04_flow:rewired:time_block_regression; pems04_speed:self:time_block_regression; pems04_speed:rewired:time_block_regression.
pems04_flow: physical/self ratio 0.936382935, physical/random ratio 1.009064505. Quarter ratios remain in the summary.
pems04_speed: physical/self ratio 0.984993223, physical/random ratio 0.964590835. Quarter ratios remain in the summary.

The physical model retained its historical best; extra training did not improve its registered selection score. The evidence does not support simply extending this recipe again. This does not prove mathematical convergence or graph impossibility.

Initial gate failed: no fresh seeds43/44, source confirmation, ChatTime or target predictions. Stop this one budget experiment without another extension or schedule search.

Acceptance:33 related tests passed, inherited full proof. Every field except identity binding is preserved by the transplant; numerical-state and next-step replay tests passed. Source prefix and best hashes remain unchanged. All caches identical, matched per-seed initial parameters and final query cursors, independent metric replay1e-6 and actual graph norm bound passed. Historical best prefix is identified explicitly, not represented as a newly trained winner.

New controller wall 195.518s; new worker 190.027s is included, not added. Inherited Stage40 worker 99.377s remains a separate prior cost. Closing verification0.420s. GPU0. Overall elapsed through summary generation658.9s excludes final synchronization. All project experiment workers/controllers exited; automation stays off.

Already exposed source validation, seed42 initial screen, small continuous-output diagnostic GNN. No independent dataset transfer or GOFA improvement proven; reaching3000 does not establish convergence. Prior failed stages and target results unchanged.

Runtime `86ab075d69e548c8795bdfb8e5558134806b2eeb`; report `1fa7b05ca40b83fd7dc6687543f2575bc65f2f3db9ff6366dee1c8de52a2dd72`; evidence `099bbdba772e1c08f9ead30dab09ab98ec6a20d6acde86ce7d4a74b0874a6417`.
