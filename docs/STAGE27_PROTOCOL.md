# Stage27: causal historical-expert target calibration

Stage26 ended at its registered pilot gate (4/6 improved), without independent confirmation or test replay. Its first64 queries per dataset are now development data. The still-unexecuted last64 are inherited unchanged as independent confirmation. Stage21 frozen test predictions remain immutable. This stage does not train a neural network, alter the parent3500 adapter, or modify graph alpha, prompts, scales or sampling.

## Fixed estimator

`prediction=sum(weight[e]*expert[e])+origin_inverse_scale*(b0+b1*u)`, with nonnegative simplex weights, `u=(h-1)/(H-1)`, `abs(b0)<=.25` and `abs(b0+b1)<=.25`. Final predictions are not clipped. All parameters are shared across variables, queries and steps within a dataset.

Experts in fixed order: Native sample8 median; parent3500 GOFA sample8 median; last observed value; mean of observed values in the last12 time indices; OLS trend from observed values in the last12 time indices. Trend is extrapolated from the actual last-observed timestamp, not an assumed fully-observed final point. No recent values: use persistence; fewer than two recent values: zero trend. Only MeteoNet/EngRAD add previous-day same-hour values from the existing48-point hourly history, with persistence for missing slots. No input history is extended. The official origin inverse MinMax scale is reused, including its constant-history convention.

Fit variable-equal original-unit MAE divided by `max(native_fit_macro_MAE,1e-12)`, plus `tau*((1-native_weight)+abs(b0)+abs(b1))`. Strengths are fixed `.01,.1,1`; HiGHS float64 feasibility tolerance1e-10. Lexicographic objectives: regularized objective, penalty, negative Native weight, remaining weights in fixed expert order, abs(b0), abs(b1), b0, b1. A zero-error Native training baseline has the known global zero-objective identity solution. Tiny solver simplex roundoff is canonicalized and the original objective is recomputed; invalid/numerically inconsistent fits fail closed.

Maximum stored deployment parameters are8 for each hourly weather dataset and7 for the others (44 total, including simplex redundancy). A separately fitted Native-history control removes only GOFA, using identical folds, penalty strengths and selection; it never selects the deployed system.

## Fixed progression and isolation

Two forward folds: fit first32, validate next16; fit first48, validate last16. Select each model/dataset's strength on the32 concatenated out-of-time predictions with variable-equal MAE; exact score ties prefer stronger shrinkage. Old pilot data are explicitly development, not independent confirmation. No confirmation/test record can be opened during fitting or development selection.

Only if all six deployed-model out-of-time MAEs strictly beat Native by `max(1e-12,1e-9*NativeMAE)` are the six estimators and six controls refitted on all64 queries and sealed. Failure stops before GPU admission and records zero new GPU work. The same threshold gates the inherited64/query per dataset independent confirmation. Any failure ends the experiment without adding features, strengths or samples, refitting, or changing to another estimator.

After confirmation passes for all six, CPU replay alone may read Stage21 test predictions. No test trajectories are regenerated, and tests cannot change the calibration. Record four systems, per-variable/per-step and macro/micro MAE/RMSE/MAPE/WAPE, paired query improvements, base fallback and derived range/coverage. MAPE `abs(y)>1e-5`; zero WAPE denominator null; missing targets excluded. Test uncertainty uses2000 paired variable-cluster draws seed4226, not a promotion gate; Kelmarsh has only six clusters.

## Execution, acceptance and recovery

Entry `python -m gofa_f.history_target_calibration`; phases prepare/develop/seal/predict-confirm/confirm-gate/replay-test/progress/report, plus resume and CPU finalization. There is no neural training entry. `python -m gofa_f.history_target_progress <output> --watch` prioritizes terminal status, marks unscheduled work skipped and suppresses stale ETA.

Independent immutable snapshot/output `stage27-history-target-calibration-v1`. Development LP caches, coefficients, input hashes and gates are atomic and bound. Resume supplements missing fits/trajectories only; it never resets cumulative costs or returns from confirmation to development. CPU activity and inherited Stage26 GPU cost are reported separately from new GPU cost.

CPU acceptance: CUDA disabled, OMP/MKL2; causal mutation tests, missing/constant/elapsed-time and day-lag alignment, weighted objective/constraints/ties/folds/control separation, seals/unauthorized reads, no-GPU failure, resume and progress. Accepted frozen numerical files must match prior CPU proof by exact SHA. Runtime GPU replay requires four original source cases and new full/incremental comparisons with identical sampled tokens, CPU FP32 1e-6/CUDA BF16 2e-3, unchanged source-validation flags and no gradients.

GPU only after CPU gate: memory_cap_only shared admission, pinned UUID/Default, project40GiB/Torch38GiB audits every10s, cooperative save/exit on excess or unknown own usage. No idle threshold, free-memory requirement, activity/wall deadline or hourly follow-up. Only verified project processes can be signalled. Base eight-sample BF16 single-block generation is unchanged.

## Interpretation

Historical calibration may explain improvements. Zero GOFA weight or failure to beat the independently fitted Native-history control is not evidence of extra graph benefit. This is a target-validation-calibrated development reevaluation with prior test exposure, not strict zero-shot, an independent blind test, or guaranteed population advantage. All-six fixed-sample point estimates are reported honestly; no automatic second scheme follows failure. Only sanitized Stage27 summaries are appended publicly; server paths/process identities remain private.
