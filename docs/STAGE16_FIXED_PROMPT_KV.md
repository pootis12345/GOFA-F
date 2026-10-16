# Stage16 fixed-prompt KV implementation

User authorized implementation of acceleration item1 on2026-10-16. Only the
frozen first30 layers' common textual prompt are reused; graph placement,
continuous head, native language path, native greedy decoding, microbatch1,
effective batch64, optimization and full training windows remain unchanged.

Official input IDs are tokenized as before. The prefix before history contains
47 matching tokens in the real77-token METR-LA prompt. A detached single-copy KV
cache is built for layers0..29, together with their boundary hidden states.
Each input must match every cached ID and have no token padding. The remaining
30 tokens are computed with original absolute RoPE positions and the full causal
mask; the full77-position boundary tensor is reconstructed before the unchanged
graph exchanges and differentiable suffix. There is no prompt truncation.

Warm-up runs the original full query and batch shape once and retains cloned
causal states for the first47 tokens of one row. This avoids changing BF16
prefill rounding by warming with a shorter query shape. Retained slices do not
keep the original full-window KV tensors alive. Historical values in the warm-up
tail cannot affect the cached causal prefix; a separate test covers this.

The cache object never appends or modifies its shared keys/values. Native device,
dtype, parameter versions and pointers invalidate it if changed. Native weights
must be frozen and in evaluation mode. Mismatched or padded inputs use the
original full-prefix path. The padding guard preserves existing strict FP32
gradient tolerance after CPU testing found rounding differences in that case.
No numerical tolerance was widened. The cache is not a parameter or saved model
state; it is rebuilt after recovery. Single-copy BF16 KV plus boundary storage
for the first30 layers is approximately22.4MiB; expanded attention temporaries
remain subject to the existing GPU caps.

GPU verification must compare predictions and all trainable gradients for both
fresh active adapters and the actual continued checkpoint under the registered
BF16 rtol/atol2e-3. State migration itself is exact. Full CPU regression and a
real7B CPU check precede any interruption of the old training. The old GPU worker
continues during preparation. It is then signaled through its verified project
supervisor to save optimizer/model/RNG, cursor and unfinished accumulation.
No new GPU task starts before all previous project processes have exited.

An explicit continuation records the source window/manifest/checkpoint SHA and
rebinds only implementation identity. Model, optimizer, gradients, RNG, sample
cursor, label counts, pending loss and validation journal are copied without
changing values. Source data/weights/graphs and scientific configuration must
match. Completed prediction evaluation prohibits training continuation. Old
snapshots/results are retained; no old caches are upgraded or overwritten.
Activity ledgers and phase memory peaks are carried forward; the static prefix
is rebuilt under the new implementation. Repeating a completed migration never
overwrites newly progressed training.

The original start remains2026-10-16 16:03:02 Asia/Shanghai; training ends21:30
and the hard stop remains23:30. Shared admission still requires three checks30s
apart with actual free memory>=56GiB and Default mode, permits foreign GPU tasks,
and retains project40GiB/Torch38GiB/reserve16GiB. No standalone throughput
benchmark is added; any speed estimate comes from normal training progress.

Current implementation source `30943b9acccd5d73c3f0a6a9000b7730e537ecdf`, artifact
`stage16-chattime-metr-v8`, continuing `stage16-chattime-metr-v6`. The earlier
v7 CPU check failed its unchanged BF16 gradient tolerance: max difference
0.03375864 (approximately0.8056% at the affected scalar). No v7 GPU task ran.
The full-shape warm-up fix passed real7B BF16 CPU prediction and gradient checks
with maximum absolute differences0 and0. v8 passed306 CPU cases with11 existing
optional skips (317 XML entries including skips); Stage16 contributed70 cases.
Evidence: `results/stage16-cpu-verification-v8.json`. CUDA remains a separate gate.

At21:12 Asia/Shanghai, verified v6 supervisor4140814 saved and exited; its
bootstrap4135772, worker4143990 and GPU child4150553 also exited before v8 launch.
Source checkpoint SHA
`2315a930296ca3ddb7fe561c3bb7cb2e5e3db53efcbb1ac3b9244c4a310ebddb`
contains167 updates, epoch0 offset10737/22969, pending49 windows and117586
observed targets. Migration verified model/optimizer/RNG/pending gradients and
counters exactly; source activity9908.55239078775seconds was retained.
Evidence: `results/stage16-fixed-kv-migration-v8.json` and
`results/stage16-fixed-kv-window-v8.json`.

The v8 CPU controller134316/start_ticks607430843 deployed supervisor176180/
start_ticks607739521 and worker177359. Initial state is shared-capacity waiting;
another GPU task entered during the transition. GPU cache checks and actual
training acceleration are not established by the CPU checks or migration.
Actual activation, counters and any ordinary-training speed observation belong
in `RUN_STATUS.md`. Neither a full validation score nor test result exists yet.
