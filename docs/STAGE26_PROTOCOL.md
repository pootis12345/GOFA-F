# Stage26: target-validation-calibrated six-target system

The frozen parent3500 adapter is `816d38b577a6664f30dd6e6b6d50986a77b098ce9c4c0b0ffb63d70078b26f15`.
No neural training, new adapter, numeric head, graph-amplitude gate or target-driven checkpoint selection is allowed.
Stage21 results remain immutable. The six datasets and their frozen test queries remain unchanged.

## Calibration and isolation

Each dataset uses `prediction=(1-lambda)*Native+lambda*GOFA+bias` after independent eight-sample medians in original units.
There are twelve deployment scalars, plus six Native-only bias control scalars.
Float64 variable-equal MAE is solved by SciPy/HiGHS linear programming; successive equality-fixed LPs resolve ties by MAE, lambda, absolute bias, then bias. Solver feasibility tolerance is recorded, never used as the strict-improvement gate.
No rescaling, clipping or coefficient grid search is permitted.

The middle20% is split into five contiguous temporal blocks. Queries per block are16/16/16/16/64.
First48 queries fit, next16 pilot-confirm, final64 independently confirm. Lists are sealed before any prediction.
Seed42 chooses IDs and temporal strata using only legal time/missing eligibility. Histories and labels are contiguous, labels remain within blocks, same-variable label windows do not overlap, and history/target each contain an observation.
High-dimensional datasets select128 eligible variables. Kelmarsh covers all six turbines with balanced repetition.

Fitting reads only fit shards. The twelve scalars and control parameters are sealed before opening pilot records.
Every dataset must strictly beat Native by `max(1e-12,1e-9*NativeMAE)` in both confirmation phases. Failure terminates the experiment without extra queries, refitting or test access. Parent preservation is reported but is not a gate.
Only after both gates pass may CPU replay the frozen Stage21 test predictions; no new test trajectories are generated.

## Execution and acceptance

The accepted BF16 graph/sampling core, prompts, scale, source-validation module flags and eight-path pointwise median remain unchanged. All graph nodes use the shared frozen LLM. Inference runs without gradients or an optimizer.
Atomic trajectories bind query, seed, model/alphabet and full manifest. Restart fills missing paths only, rebuilding KV. Changed fit inputs invalidate the coefficient seal.
CPU tests run with CUDA disabled and two OMP/MKL threads. Frozen-core CPU proof is reused only under exact numerical file hashes. GPU admission requires four exact source replays and new-query full/incremental comparisons (BF16 2e-3 plus identical sampled tokens); no tolerance relaxation.

`memory_cap_only` verifies the pinned GPU UUID and Default mode and starts shared work immediately. No idle/free-memory condition is inherited. Audits every10s limit the verified project group to40GiB and Torch to38GiB. Unknown usage or excess triggers cooperative save/exit, with60s maximum. Only this project's verified processes may be signalled. Activity/wait/pause/peaks survive recovery. There is no wall-clock/activity deadline or hourly follow-up.

## Interpretation and reporting

Report Native, parent3500, Native bias control and blend+bias; original-unit macro/micro and per-variable/per-step MAE/RMSE/MAPE/WAPE, paired improvement counts, and2000 variable-cluster bootstrap draws (seed4226). MAPE uses `abs(y)>1e-5`; zero WAPE denominator is null; missing labels are excluded. Kelmarsh has only six clusters.
Derived predictions have no language-generation coverage claim. Base fallbacks and derived out-of-range proportion are retained.
Lambda0 or no improvement over Native bias does not demonstrate extra graph gain. Both target validation labels and previously viewed test results are disclosed: this is a development reevaluation of a calibrated system, not strict zero-shot or an independent blind test. Fixed-sample point-estimate wins do not guarantee population advantage.

## Interfaces

`python -m gofa_f.six_target_calibration --config configs/stage26_six_target_calibration.yaml --output <independent-output> --phase prepare`

Phases: prepare, predict-pilot, fit, pilot-gate, predict-confirm, confirm-gate, replay-test, report, progress. Prediction requires the resource supervisor; the supervisor performs CPU gates only after GPU workers exit. `--resume` preserves atomic results; terminal failures do not automatically restart.

`python -m gofa_f.six_target_progress <output> --watch` displays per-dataset/model queries, sample index, conditional overall ETA and resource costs. There are no training epochs or updates in this stage.
