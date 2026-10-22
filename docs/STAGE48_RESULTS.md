# Stage48 fixed linear source-information diagnostic

This is new source-supervised closed-form CPU fitting, not zero training. No neural network was trained and no GOFA/ChatTime candidate was produced. One fixed ridge=1, no coefficient or feature search.

Use the unchanged2048 source training queries/view,72 validation queries/view and held-out18 source variables. Features are24 target normalized past increments/masks, optionally24 neighbor mean own-scale increments/coverage. Physical/random controls are matched in feature dimension and neighbor count. All features use the original12-point past window only; no future neighbor values, IDs or added data. Target is12-step direct residual in the target origin scale. Float64, training-only query-weighted standardization, horizon-wise mean weighted squared error plus unit L2, unpenalized intercept. Source query weights preserve the existing variable/valid-point weighting convention.

| View | Self-history MAE | Physical neighbors | Matched random neighbors | Physical vs self | Physical vs random |
|---|---:|---:|---:|---:|---:|
| pems04_flow | 26.635544492 | 26.412003033 | 26.721128262 | -0.839% | -1.157% |
| pems04_speed | 1.547404923 | 1.496876388 | 1.578024272 | -3.265% | -5.142% |

Speed physical/self ratio interval0.9373–0.9946 supports usable neighbor-history information for this fixed representation on exposed source validation. Physical/random ratio0.8967–1.0132 crosses1, so road-specific stable gain is unproven. Flow physical/self interval0.8928–1.1398 and physical/random0.9587–1.0180 both cross1; the first two flow time blocks degrade6.94%/8.52% relative to self. The speed physical/random point estimate improves in all four blocks but that alone is not a generalization proof.

Intervals use2000 paired variable-cluster draws, seed4248. There are only18 held-out source variables, already examined in multiple development stages. No independent confirmation was opened, and no test results fed fitting. This is conditional on one feature summary, squared-error objective and fixed regularization. Failure cannot prove spatial information absent; success cannot prove graph causality. Self has fewer features, so the matched physical/random control is necessary.

Decision: stop further graph-bias/scale hyperparameter trials. The next research question is source training that reliably extracts neighbor-history innovations across variables/time; this audit does not authorize an empirical claim of improved GOFA or promote a model to expensive ChatTime/target evaluation. No additional recipe was appended.

Five focused core checks passed: future labels ignored by features, missing/constant finite handling, analytic ridge solution, weighted missing-label fitting, and validation not changing fit coefficients. Independent prediction replay max difference1.14e-13, metric difference<=4.45e-16, within original1e-6 tolerance. Old cache/report hashes unchanged; all foreground processes exited. No broad regression rerun or automation.

Audit CPU wall0.252s includes six closed-form fits and self tests; independent replay0.040s. These exclude interpreter startup/development/transfer. Full elapsed through summary generation535.9s excludes final synchronization. GPU0, neural updates0, target forecasts0; CPU peak RAM not monitored.

Manifest `fb65145f8f763e5543abb23b2ecb0a350303e662a7d855dbe7dd4464a4c066ee`, report `2fc9e431dc0eebf0c358c22b5b79421d76e4c4c313df52bf3497855600f7a036`, evidence `05fa2b76eb234cf41c2613ca81175ebe491d6eec27c9870af7986c637a39a6ab`.
