# Stage33: one bounded past-SCADA wind experiment

Preregistered before fitting any Stage33 model. Stage32 has stopped with fresh SDWPF confirmation MAE221.662693 native versus232.290438 calibrated (+4.79%). Its original candidate and coefficients remain sealed. There was no frozen-test replay. The failed confirmation is now explicitly exposed development data, never described as an unseen holdout.

The existing first60% target-supervised setup remains. This system is not zero-shot GOFA-F. Five datasets' Stage31 parameters/models are frozen exactly. Native and the3500 graph adapter remain frozen. No neural head, backbone update, new target dataset, or GPU hyperparameter search is introduced.

## Hypothesis and new information

Power-only histories can fail to distinguish calm wind, curtailment and operating transitions. The official [SDWPF data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC11187227/) identifies measured historical wind/temperature/pitch/nacelle variables as dynamic context. This motivates an independently specified experiment, not a claim that the earlier failure's cause is proved.

Use only the already downloaded original full parquet. Add strictly pre-origin measured Wspd(m/s), Etmp/Itmp(C), Pab1(degrees), Prtv(kW), and sine/cosine of Wdir and Ndir. Do not use ERA5, future measurements, future operational states, or future missingness. Keep Stage21 target values/masks, including zero and negative power, unchanged. The SCADA-assisted CPU expert receives more input channels than the LLM; any advantage is a supervised system result, not proof of graph-LLM contribution or a matched-input zero-shot comparison.

The data converter checks exact time/node/power alignment against the immutable Stage21 arrays, pins raw and derived hashes, and writes to a new Stage33 output only. PyArrow17 is isolated in a new data-reader directory; old environments are untouched. Timestamp ambiguity, missing required fields or alignment failure stops preparation.

## Fixed four CPU experts

Use the same16384 ID/time/capacity-balanced first60% training queries as Stage32, all134 turbines. Raw-unit absolute-error HGB:300iterations, learning rate.05,31leaves/depth5, minleaf64, L2=1,63bins, seed42, no internal early stopping, two threads. Each variable has equal total valid-label weight. No label cleaning, clipping or changed evaluation masks.

Exactly four candidates: residual_local, residual_neighbor, level_local, level_neighbor. Residual predicts y-last; level predicts y directly. Both retain raw AND normalized48-point power features. Measured SCADA features are per-channel last/coverage plus mean/std/time-index slope over3,12,48points. Neighbor variants append the mean/std of these past summaries across at most8 published static incoming neighbors, canonical ID order. Missing inputs remain NaN; unavailable neighborhood summaries are zero. No turbine-ID parameter or extra history is added. No fifth candidate or additional tuning.

## Development, selection and independent confirmation

Development consists of the128 original SDWPF development records plus64 Stage31 and64 Stage32 failed confirmation records,256 exposed queries total. No old confirmation is called blind, and no frozen test is read. Fit no model weights on their labels: they are used only for convex Native/CPU calibration and finite candidate selection. Graph weight is fixed to0 for this new wind calibration; the Native+CPU control is identical and no extra graph gain will be claimed.

Use three forward folds: timeblocks1-2 fit→block3(32) validate;1-3 fit→block4(32);1-4 fit→block5(128). Each block is a fixed fifth of the middle20%. Fit only a single Native/CPU convex coefficient by variable-equal original-unit MAE, float64 HiGHS, tied optima prefer Native. Candidate eligible only if EVERY fold is nondegraded and pooled variable-equal MAE improves at least0.5% plus original strict tolerance. Select minimum worst-fold MAE ratio, then pooled MAE, then the above fixed recipe order. Refit the one selected coefficient on all256 exposed development records and seal hashes. Identity Native refits stop.

Before any model fitting, seal64 NEW SDWPF confirmation origins in the original fifth validation block, same64 target identities as before, seed[42,33,index,variable]. Exclude all same-variable development/previous-confirmation label-window overlaps; labels do not cross blocks/breaks. Selection uses IDs, masks and lawful origins only. The validation period itself is development-exposed; report this limitation. No replacement variables or alternative confirmation after failure.

The other five Stage31 parameters are exact. Reuse the four fully completed confirmation sets only under exact query/model/data/core/parameter bindings. Kelmarsh's original64 confirmation is still unexecuted. SDWPF uses ONLY the new64; none of its failed old forecasts count as new confirmation. Native always required, graph only where its frozen coefficient is nonzero. One failed dataset seals the scheme and skips all remaining confirmation/test. Six strict confirmation improvements are required before ONE CPU replay of the unchanged Stage21 frozen6×128 test predictions. No test feedback to this or a second candidate.

## Acceptance, resources and cutoff

CPU causality/missing/constant/angle/time/topology tests, raw-unit weighting, forward-fold isolation, new-confirmation disjointness, save/load/model and input hash checks, full CPU regression. Reuse unchanged frozen numerical core proof. Before GPU confirmation replay four source cases exactly and check full/incremental new-query logits within BF162e-3 AND identical sampled tokens; CPU1e-6. No gradient updates or relaxed tolerances.

memory_cap_only shared admission, project40GiB/Torch38GiB, audits every10sec; resource uncertainty/overcap saves and exits cooperatively, own verified processes only. Inherit all previous activity, record new CPU/GPU separately. The original2026-10-22 11:15:12 Beijing cutoff is unchanged; stop GPU before it with existing save margin. Hourly monitors remain off; only the explicitly authorized temporary30-minute follow-up runs. Report terminal ETA as not applicable.

Deliver full outcomes including failures, coefficients/models/hashes, per-variable/step metrics and uncertainty, exposed-development limitations, cost and skipped fields. Do not imply success before six frozen-test improvements. Conditional LLaMA/GNN experiments require genuine versions, fair protocols and truthful rankings; no guarantee of desired ordering.
