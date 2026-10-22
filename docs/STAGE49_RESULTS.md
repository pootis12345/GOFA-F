# Stage49 source innovation-residual results

Registered source gate passed: **False**. This is the predeclared source diagnostic system, not a ChatTime/GOFA or six-target result.

Jointly center each node history at its own last observed normalized value, and train the same normalized GNN to correct the frozen Stage48 source-self ridge prediction. Original raw truth/scales/masks/persistence/query weights remain unchanged. Baseline is computed in float64 from the original cached history before converting centered neural inputs toFP32. No new baseline fitting or bias/strength search. Two frozen source-specific CPU metadata sets are auxiliary diagnostic components, not a new language-model head or a deployable shared target system.

| Condition | Flow variable-equal MAE | Speed variable-equal MAE | Selected update |
|---|---:|---:|---:|
| Physical-roster residual GNN | 24.305444945 | 1.494625222 | 1000 |
| Self residual GNN | 26.030657536 | 1.559887852 | 500 |
| Matched-count random-roster residual GNN | 24.871072966 | 1.473150466 | 1000 |

Frozen Stage48 self references: flow26.635544492, speed1.547404923. Stage40 physical reference: flow23.632124150, speed1.495254458. These are historical references, not extra selected candidates.

Original initial gate failures: pems04_flow:rewired:time_block_regression; pems04_speed:rewired:less_than_1pct_gain; pems04_speed:rewired:time_block_regression.
pems04_flow: physical/self ratio 0.933723818, physical/random ratio 0.977257595; quarter ratios retained in the summary.
pems04_speed: physical/self ratio 0.958161973, physical/random ratio 1.014577436; quarter ratios retained in the summary.

Initial gate failed: no seeds43/44, independent source confirmation, ChatTime updates or target predictions. Seal this single recipe without another checkpoint, loss weight or longer budget.

Acceptance:33 related CPU tests passed, inherited bound full proof. Matched initialization/cursors, immutable source cache and frozen baseline hashes, label-preserving residual math, zero correction returning frozen baseline, context restoration and optimizer replay checked. Independent metric replay and source norm bound verified. Self trajectory need not match Stage40 because both input and residual target changed. No single-factor causal attribution is valid.

Controller wall 100.310s; worker activity 94.626s is included, not added. Closing verification 0.540s. GPU0. Overall elapsed through summary generation619.2s excludes final synchronization. Workers/controllers exited, follow-up remains off.

Exposed source validation and small diagnostic GNN only. This does not prove spatial forecasting impossible, demonstrate independent dataset transfer, or establish a GOFA improvement. Representation and residual training changed together; the fixed per-source CPU baselines further limit direct deployment interpretation.

Runtime `8c4655bed6be9e90c856ee0014b6d2abbe0cc979`; report `f929bd89df174442f2dee36c411960adfe0559e04d5c611d2eaad20ea7cd43e7`; evidence `23d863846c8b8b162b919d73ce1d4804b13705193fe4e2c8867352961745a745`.
