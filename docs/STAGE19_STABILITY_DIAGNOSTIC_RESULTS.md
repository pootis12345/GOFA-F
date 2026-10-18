# Stage19 graph-amplitude/target consistency diagnostic

Status: implemented; CPU verification passed. New BF16 teacher and alpha=0.25 sampling results are pending GPU admission.
Training remains user-paused; no target task, optimizer update or hourly automation is enabled.

## Zero-GPU reanalysis of existing evidence

The same 32 ID-stratified source queries are used throughout. Two known NN5 cases remain separate.
MAE below is an equal average over the eight queries within each source; raw errors are not averaged across sources.
These numbers reuse old teacher and sampling shards and are not new free forecasts.

| Source | Native teacher CE | 500 CE | 1000 CE | Native sample8 MAE | 1000 sample8 MAE |
|---|---:|---:|---:|---:|---:|
| Solar | 3.371194 | 3.074551 | 3.062278 | 0.907423 | 1.188772 |
| NN5 | 9.074735 | 8.287375 | 8.183745 | 8.343070 | 6.412708 |
| PEMS04 | 5.430347 | 4.875037 | 4.915436 | 43.265623 | 28.116404 |
| AQShunyi | 4.517262 | 3.652372 | 3.729999 | 10.042120 | 11.383619 |

Thus lower CE and worse free sample8 error remain paired at the source level for Solar and AQShunyi;
NN5 and PEMS04 have improved sample8 error on this small subset. Teacher argmax is not a conditional median.
The new full/top100 conditional-median measurements are required before distinguishing numeric decision effects
from generation-prefix effects. The existing findings alone cannot establish that large graph increments are harmful.

| Source | 1000 exchange1 delta/input norm | Exchange2 delta/input norm | Gate1 | Gate2 |
|---|---:|---:|---:|---:|
| Solar | 1.395636 | 0.761902 | 0.923252 | 0.854896 |
| NN5 | 0.903147 | 0.974840 | 0.848476 | 0.913683 |
| PEMS04 | 1.018229 | 0.655036 | 0.885235 | 0.859388 |
| AQShunyi | 1.126460 | 0.643235 | 0.917719 | 0.845675 |

Each ratio uses the actual input to its respective exchange; these are averages of per-query ratios.
The historical 500 model has no eight-sample trajectories in this evidence set, so its sample8 score stays unavailable.
Existing per-query teacher/greedy/sample8 evidence, source/template and first/later/H24-segment summaries,
with original file paths and hashes, are saved in `results/stage19-stability-existing-analysis.json`.

## Fixed new measurements

102 teacher forwards: native, alpha=1 and alpha=0.25 on 34 queries each. Only alpha=0.25 adds 272 sample trajectories.
Reused native/alpha=1 samples retain original query, weight, data, code, scale and seed provenance.
The BF16 amplitude intervention, mechanisms, new metrics, actual GPU activity/wait/peak and exit audit are pending.
No alpha=0.25 result, best checkpoint or training recommendation is inferred before those measurements exist.

CPU Ruff and complete regression: 506 passed, 21 optional-dependency skips. Both real 7B FP32 cases
(PEMS04 variable102 and NN5 variable92, H24, first fixed sample) passed original 1e-6 tolerances
and exact actual sampled tokens. The complete causal CPU path replays the actual sampled prefix in one forward;
the GPU checks retain actual stepwise full generation versus KV generation. No CPU CUDA context was created.
[CPU evidence](results/stage19-stability-diagnostic-v1-cpu.json).

The adapter remains locked at the source-best 1000-update SHA256
`18128f46578da279f842e9b2efe3cfa3d5c6a27f7404b67e135ab5c07f765db9`.
Production last SHA256 remains
`a37c4f4e793d073d20c8bc2af9ba437977a068f953518e134fd4794ac278cd3b`.

## Limits

Single fixed seed family, 32 small source-development queries and two separately disclosed cases.
The Stage19 BF16/prompt/finite-grammar/single-block sampling protocol differs from official FP16/rolling blocks.
This does not replace complete source validation, relabel an adapter as best, test targets or establish graph-enhanced QA.
Future training/validation sampling-policy intent is recorded, but no training or production evaluation path is resumed here.

[Preregistered protocol](STAGE19_STABILITY_DIAGNOSTIC_PROTOCOL.md).
