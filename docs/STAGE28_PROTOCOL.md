# Stage28: frozen five calibrators and bounded SDWPF residual calibration

This is one target-validation-calibrated development reevaluation, not strict zero-shot or blind testing.
No neural training, numeric neural head, adapter/checkpoint selection, graph-amplitude change or new dataset is allowed.
Freeze the parent3500 adapter and Stage27's five non-SDWPF full/control calibrators byte-for-value.

## Data and fixed calibration

Add exactly64 SDWPF queries, one per original64 development variable, using the opposite temporal block `(b+2)%4`.
Use seed42 and only legal time/ID/missing eligibility; label windows never overlap for the same variable.
The four old development blocks each contain32 combined queries; the fifth confirmation block and last20% test remain excluded.
Seal the lists and feature protocol before inference. Do not inspect interim errors to choose queries.

SDWPF alone has `prediction=Native+s*beta@x`, `sum(abs(beta))<=0.25`, with eight bounded features:
intercept; normalized horizon; clipped `(GOFA-Native)/s`; clipped `(last-Native)/s`;
clipped `H*OLS_last12/s`; horizon times target trend; horizon times neighbor mean trend;
target-minus-neighbor mean normalized last-history level. All clipping is feature-only; final forecasts are never clipped.
Neighbors are only observed external-graph incoming senders within the existing accepted origin subgraph, excluding self.
Each node has its own history MinMax scale and level; constant range uses1. Missing neighbors are excluded;
no usable neighbors yields zero neighbor features. Fewer than two recent observations gives zero slope.
Only the48 original history steps are read; original negative/zero powers retain their audited meanings.

Fit float64 variable-equal raw MAE normalized by Native fit MAE plus `tau*L1(beta)`.
Use the existing HiGHS LP and exactly tau0.01/0.1/1. Resolve ties by primary objective, L1,
feature-order absolute coefficients, then feature-order signed coefficients. Zero Native fit error gives beta0.
Also fit a Native+same-history/neighbor control with graph-difference feature removed.

## One-way gates

Two forward folds are64 train/32 validate and96 train/32 validate. Pick tau by pooled variable-equal
out-of-time MAE; exact ties choose larger tau. Control chooses independently and cannot select deployment.
Advance only if both chosen folds do not degrade and pooled improvement is at least0.5%.
Refit all128 once. Exact beta0 stops BEFORE confirmation GPU work; nonidentity is not evidence of improvement.
No extra features, strengths or samples after failure.

Seal all six parameter sets and input hashes before reading confirmation. Reuse Stage26's frozen final64 per dataset.
Stage27's partial PeMS03 Native results may be imported only after sealing, with matching query/seed/core/model/scale
provenance and explicit original path/hash receipts. Old evidence is not edited and its GPU cost is not charged again.
All six must strictly improve by `max(1e-12,1e-9*NativeMAE)`. Failure ends the experiment without refitting.
Only after all six confirmations pass may CPU replay Stage21's frozen tests, once; no test GPU inference is allowed.
Previously viewed tests and partial prior confirmation are disclosed. No graph-LLM benefit claim without beating
the Native+same-history/neighbor control; spatial summary benefits alone are not proof of LLM graph increments.

## Acceptance and operation

Independent entry `gofa_f.state_residual_calibration` and progress `gofa_f.state_residual_progress`.
Phases: prepare, predict-development, develop, seal, predict-confirm, confirm-gate, replay-test, report, progress.
The supervisor alone authorizes GPU workers; CPU gates execute only after worker exit/ledger clearance.
CPU regression disables CUDA and uses OMP/MKL2 threads. Reuse frozen-core proof only under exact numerical hashes;
runtime requires four source token replays and new full/incremental checks, CPU1e-6/BF16 CUDA2e-3 and identical tokens.
Keep accepted module flags and no gradients. Resume retains atomic trajectory shards and all activity/pause/peak records.

Use memory_cap_only; do not wait for GPU idle/free-memory thresholds. Project40GiB, Torch38GiB, audit10s,
cooperative own-process-only save/exit within60s on excess or unknown usage. No activity/wall deadline or hourly follow-up.
Display dataset/model/query/sample and gated overall ETA; terminal status takes precedence and skipped work has no ETA.
New development inference costs1024 paths; confirmation is conditional, final tests are CPU-only.
Report macro/micro/per-variable/per-step MAE/RMSE/MAPE/WAPE, fallback/base coverage, derived outside-range fraction,
paired improvement counts and2000 variable-cluster bootstrap draws seed4226. MAPE uses abs(y)>1e-5;
zero WAPE denominator is null. Kelmarsh has only six clusters; point-estimate wins do not prove population superiority.
