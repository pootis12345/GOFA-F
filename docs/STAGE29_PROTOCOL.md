# Stage29: target-supervised CPU forecasting with frozen ChatTime

This is a target-supervised hybrid forecast system, **not zero-shot GOFA-F**. Native ChatTime and the parent3500 graph adapter are frozen. No neural optimizer, numeric neural head, new data, architecture search, or hourly follow-up is enabled. Earlier viewed target results remain development exposure.

## Fixed inputs and CPU models

The six targets remain PeMS03 and PEMS-BAY (12→12), MeteoNet, EngRAD, SDWPF and Kelmarsh (48→24). Keep their audited data, graph mappings, fixed Stage21 test queries, BF16 numerical core and eight-sample stepwise medians.

Train one shared-variable `HistGradientBoostingRegressor` per dataset using scikit-learn 1.5.2. Each has 4096 first-60%-only training queries, selected by ID/capacity/time with seed42. Start with earliest disjoint label chains, allocate queries by capacity-aware variable round-robin, then stratify each variable's chain in time. Labels cannot cross splits or time discontinuities. Scarce eligible variables are retained; remaining quota goes to variables with spare capacity. No error-based selection occurs.

Fixed settings: absolute_error, 150 rounds, learning_rate=.05, max_leaf_nodes=15, max_depth=4, min_samples_leaf=40, l2_regularization=1, max_bins=63, random_state=42, early_stopping=False. Six datasets run serially with OMP/MKL two threads. Sample weights have mean1 and give each variable equal total effective weight, preserving a normal weight scale for the fixed L2 setting.

Each horizon slot predicts a residual from the last valid observation. Features are raw L-step values and masks; observed mean/population standard deviation/time-index OLS slope/coverage for distinct windows {3,6,12,L}; last value and its age, historical min/max; normalized horizon; and forecast daily/weekly sine/cosine inferred from the last past timestamp and fixed cadence. Missing values remain NaN; empty-window slope/coverage are0 and mean/std are NaN. No variable IDs, neighbors, future covariates, future missingness or recursive forecast inputs are used. Negative power and true zeros retain the audited rules.

## Development and sealing

Reuse the existing Native/GOFA predictions: Stage26 first64 queries for five datasets, Stage28 chronological128 for SDWPF. No LLM development generation is added. The CPU tree is trained only on the first60%; development is in the middle20%.

Fit one three-expert convex blend per dataset, with Native/GOFA/CPU weights, no bias or final clipping. Minimize variable-equal raw-unit MAE using float64 HiGHS LP; ties maximize Native weight then minimize graph weight. Independently fit the Native+same-CPU control; report the CPU alone. Controls never select deployment.

Forward folds are32→16 and48→16, or SDWPF64→32 and96→32. All six must have neither fold degraded and pooled MAE at least0.5% below Native, beyond max(1e-12,1e-9·NativeMAE). Failure seals the experiment with no GPU confirmation or test. After success refit blends on all development records, without retraining trees. An exact Native endpoint stops before GPU. Seal six CPU model SHAs,18 weights, controls, query/data/code/config hashes before accessing confirmation.

## Conditional confirmation and one test replay

Use the pre-existing last64 confirmation queries per dataset in fixed order. Reuse identical Stage27 partial Native trajectories with full provenance. Always obtain Native; omit new graph sampling exactly when its deployment weight is0 and leave its confirmation metrics null. CPU predictions are direct. After each dataset, stop immediately if its blend is not strictly better than Native. Do not append samples, refit, retry another model, or access tests after a scientific gate failure.

Only after six confirmations pass, replay the frozen Stage21 tests on CPU once, reusing both LLM predictions without generating test trajectories. Report Native, parent3500, CPU, Native+CPU and full ensemble, variable-equal/micro MAE/RMSE/MAPE/WAPE, per-variable/per-step metrics, paired-query counts and2000 variable-cluster paired bootstrap draws (seed4226). MAPE requires |y|>1e-5; zero-denominator WAPE is null. Six Kelmarsh clusters limit inference. No extra graph-LLM gain is established by a zero graph weight or failure to beat the Native+CPU control. Derived forecasts have no language-generation coverage.

## Acceptance, recovery and resource policy

Test causal features, fixed-cadence clocks, missing/constant/negative data, quota allocation, valid-label variable weighting, exact fusion endpoints, serialization, SHA rejection, forward-fold isolation, sealing, skipped graph paths and confirmation/test barriers. CPU tolerance remains1e-6. Reuse the unchanged accepted frozen numerical-core proof; before any new GPU prediction replay all four exact source cases and check full/incremental paths with BF16 tolerance2e-3 and identical sampled tokens. Never relax tolerances.

Models and trajectories are saved atomically. Resume verifies their bindings, skips completed models, recomputes only unfinished models and restores only missing trajectories; sealed blends cannot return to training. CPU/GPU costs are retained. Progress includes query preprocessing, boosting rounds, development, required sampling paths and conditional overall ETA; terminal ETA is not applicable.

GPU admission is memory_cap_only with a40GiB project-group cap,38GiB Torch cap and10-second own-group audits. Do not wait for GPU idle, alter GPU modes, or operate on foreign processes. Resource pauses save safely and can resume; there is no wall-clock or activity deadline. Scientific gate failure ends this single fixed recipe. Success only describes the frozen finite test samples and does not guarantee population-wide advantage.

## Entry points

Configuration: `configs/stage29_supervised_target_ensemble.yaml`.
Main module: `gofa_f.supervised_target_ensemble` with prepare, train-cpu, run-cpu, develop, seal, predict-confirm, replay-test, progress and report phases.
Controller: `scripts/bootstrap_supervised_target_ensemble.py` performs bound CPU acceptance, CPU training/development and only then conditional supervised GPU confirmation.
Progress: `python -m gofa_f.supervised_ensemble_progress OUTPUT --watch`.
