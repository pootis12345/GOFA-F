# Stage38: isolate physical-neighborhood selection before LLM work

Preregistered before fitting, under the user's renewed instruction to plan and execute the next minimal-cost step. Stage36/37 remain failed and immutable. Their rewiring retained graph-selected neighbors, so it cannot isolate whether neighborhood selection supplies useful information. This is one new source-only experiment, not a continuation of the failed graph-convention study.

Reuse Stage37's exact source observations, 257/18/32 disconnected-node split, 2048 train queries/view, 72 validation/view and unopened 128 confirmation/view, 12→12, masks, per-node origin scales and causal edge features. Validation was already exposed in previous development; no blind-validation claim. Target datasets and ChatTime weights remain unopened/unchanged.

Three independently trained controls, same 10831-parameter model, initialization, query order, variable-equal loss, optimizer, 1000 updates and source checkpoint selection:

1. Physical BFS roster, complete cross-node attention within at most9 selected nodes.
2. Same physical roster but no cross messages (self control).
3. Uniform random roster within the SAME train/validation/confirmation node partition, with identical target and roster size, complete cross-node attention. Sample without replacement from all other eligible partition nodes, allowing chance overlap with physical neighbors. Fix by seed42/stage38/target/partition; share across channels/dates/model seeds; do not match based on error, future labels or correlations. This controls node count/computation, not sensor marginal distributions.

The old on-disk condition aliases `physical/self/rewired` are retained only to reuse the unchanged fitting/gating routines. Here `rewired` means random NEIGHBOR IDENTITIES, not rewired edges within the same physical roster. Its legacy non-target adjacency permutation is an exact identity for complete support; test this. Edge features and histories stay with their actual sampled nodes. All-node computation is retained. No feature, loss, model, lag or amplitude search.

Keep the Stage36/37 numeric gates unchanged: seed42 physical must beat both independently trained controls by≥1% in EACH flow/speed view, no quarter worse by>2%. Failure seals the stage. Only passing triggers seeds43/44, the original direction/mean/2000-node-cluster interval checks, then checkpoint locking before opening source confirmation labels. Same gates on confirmation. No favorable seed selection. Passing still proves neither whole-dataset transfer nor a ChatTime graph gain; further LLM work needs a separate bounded plan. On failure do not append another recipe in this turn.

CPU FP32/CUDA disabled/OMP-MKL2. Reuse verified Stage37 full738-pass/21-optional-skip numerical proof only when model, training/selection and source-reader hashes match. Run the existing17 related tests plus new roster/isolation/causality/gate tests ONCE; no preceding duplicate QA or second full project suite. New explicit roster argument has identical old default behavior. Source/model/query binding, recovery, atomic saves, exact zero-message identity and 1e-6 remain mandatory. Do not edit old snapshots or results.

Use `gofa_f.source_neighborhood_probe`, `configs/stage38_source_neighborhood_probe.yaml`, independent snapshot/output `stage38-source-neighborhood-probe-v1`. Reuse fitting/checkpoint/gate code rather than add a training framework. Record separate prepare/test/source-worker/controller timings; controller time excludes development/transfer and cannot be reported as total user wait. Synchronize code/protocol once before running, results once after. No GPU use, target forecasts, ChatTime updates, UFG run or restored automation. Report all outcomes, costs and limitations.
