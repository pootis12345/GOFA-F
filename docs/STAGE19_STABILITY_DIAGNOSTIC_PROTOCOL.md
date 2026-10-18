# Stage19 graph-amplitude and objective consistency diagnostic v1

User-approved preregistration. Training, target evaluation and hourly followup remain paused.
No optimization, backward, parameter search, new checkpoint selection or automatic training resume.

## Locked inputs

- Production output: `stage19-short-generative-chattime-v5`; paused at 1177 updates, pending 30 queries.
- Paused last SHA256: `a37c4f4e793d073d20c8bc2af9ba437977a068f953518e134fd4794ac278cd3b`.
- Adapter: source-best 1000-update SHA256 `18128f46578da279f842e9b2efe3cfa3d5c6a27f7404b67e135ab5c07f765db9`.
- Reuse the prior 32 ID-stratified source queries and two known NN5 cases; no substitutions.
- Conditions: native, alpha=1, alpha=0.25. Reuse native/alpha=1 sample8; generate only 272 new alpha=0.25 trajectories.
- All conditions get 34 teacher-forced forwards (102 total). Known cases are excluded from the primary 32-query summaries.
- Keep BF16, prompt, immutable origin scale, finite numeric vocabulary, single block H<=24, top-k100/temperature1/top-p1.
- Model-independent query/sample-bound seeds; serial execution and fresh per-trajectory graph-conditioned KV.

## Intervention and observations

Diagnostic hooks multiply both all-node exchange outputs by 0.25 before their following Transformer blocks.
Exchange2 is recomputed from the changed exchange1 path. No production model file or parameter is edited.
Invalidate graph contexts/KV on scope and checkpoint changes; frozen prefix reuse requires its complete input binding.

Teacher metrics use logits at answer position minus one and the existing causal fill/missing-label rule:
full 10000-class CE, argmax MAE, numeric-order full median MAE, actual top100 conditional median MAE,
true-token top100 coverage, entropy, endpoint probability and first-slot concentration.
Teacher-conditioned values contain true earlier responses and are not free forecasting scores.

Save every teacher and sample atomically with manifest, query, condition and seed bindings.
Resume only absent shards and reconstruct KV. Observe free probabilities without changing sampling or reading future truth.
Record raw/scaled delta-to-actual-input norms, positional distributions and gates for both exchanges.

Report query-weighted original-unit MAE per source/template, paired improve/tie/worse counts,
constant multistep-query rate (H=1 excluded), endpoint proportions, first/later and H24 first16/last8 errors.
Do not average raw-unit errors across sources. Retain raw tokens, trajectories, medians, masks and provenance hashes.

## Acceptance and resources

B edits/static checks; C disables CUDA and uses two OMP/MKL threads for regression and real 7B FP32 cases.
NN5 variable92 and PEMS04 variable102 alpha=0.25 first trajectories must match full/KV logits within
CPU rtol/atol 1e-6 and CUDA BF16 2e-3, and have identical actual sampled tokens.
Native/alpha=1 first-sample replays on those fixed cases must exactly reproduce stored BF16 tokens.
Failure stops; tolerances are not widened. Nonzero all-node scaling, exchange2 recomputation,
context invalidation, missing targets, no future input and unchanged weights are checked separately.

Use CPU idle supervisor: three consecutive observations ten seconds apart, no compute/graphics applications,
0% utilization, <=256MiB used, same GPU UUID and Default mode. No CUDA context while waiting.
Runtime project group <=40GiB, Torch <=38GiB, ten-second audit, cooperative save/exit on resource pause.
Other users joining does not trigger exit. Never change GPU mode or signal another project.
GPU estimate 2–5 minutes is not a deadline; record actual activity, waiting, pauses and peaks.

## Interpretation and completion

CE and teacher median improve while free MAE worsens: consistent with generation-prefix shift being relevant.
CE improves while teacher median worsens: consistent with a probability versus numeric point-objective difference.
Shrinking helps both teacher median and free MAE: supports further amplitude-control research, not selection of 0.25.
Fewer endpoints/repeats without MAE improvement is not an effective fix. Preserve mixed source findings.
These are descriptive small-sample diagnostics, not causal isolation or a replacement for complete source validation.

Deliver source/template/query evidence and mechanism results. Verify all diagnostic workers exited and paused
last/best hashes unchanged. Keep training, targets and hourly followup paused. Future sampling-policy intent is
recorded only; this diagnostic does not rewrite the existing training or evaluation implementation.
