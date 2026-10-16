# Stage16: METR-LA ChatTime and supervised GOFA-F(ChatTime)

Preregistered by the user on 2026-10-16. Only METR-LA 12->12, seed42, all
chronological 70/10/20 windows are run. No LLaMA-2, no no-graph adaptation, no new
GNN, and no automatic architecture/loss/LR/decode search.

Official source: ForestsKing/ChatTime commit
`8c2d4c209302d2b2bd6cc3c154586842341e3247`. Model:
`ChengsenWang/ChatTime-1-7B-Chat`, revision
`20baa88d6ab838b34abbd1d2b1f4cea57c8ff870`. Every downloaded file is hashed and
LFS shard hashes are checked. The model has 32 layers, width4096, vocabulary42001.

Native forecasting uses the official observed-history-only min-max scaling,
10002-entry centers, numeric serialization and prompt, followed by greedy
generation of at most64 new tokens, ending once12 serialized values are complete.
This early stop preserves the first12 greedy outputs. It is **not** the author's default eight
sample median. The first12 serialized outputs are decoded. Missing/nonfinite
outputs fall back per point to the last observed history; empty histories use
observed-training node means. Main native scores are therefore full-coverage
generation+fallback scores, with fallback rates and valid-generation-only scores
reported separately. No failed examples are removed from full-coverage scores.

GOFA freezes all native weights, including the original embedding, tokenizer and
LM head. Graph exchanges are after zero-based blocks29/30, leaving block31.
Only the last valid prompt token is modified. The original causal attention is
retained. Graphs use the current physical prior and the locked Stage8 train-only
top8 correlation support SHA
`d25fb015fb887d5fae60411af29cd20315fec31f8d635b4a965f07708312e8d0`.
The prior keeps at least50% mass, graph rank64, attention rank16, 19-dimensional
history/value/known-clock context. Native forecasting does not get these extra
graph/context inputs. This information difference is disclosed.

A zero-initialized shared-node4096->12 linear head predicts a raw-unit residual
around the last observed value, scaled by the observed window range; constant
history scale is1. Empty histories return the training node mean. Initial
continuous forecasts equal Persistence, **not** native ChatTime generation.
Expected trainable DOF: graph+router1071750, head49164, total1120914. Existing
UFG-A108927 is only a historical reference, not a new matched baseline.

AdamW, raw-unit masked MAE, graph/routerLR1e-3, headLR1e-4, weight_decay1e-4,
clip1, effective graph batch64, microbatch1/accumulation64, max100epochs,
patience15. Validation is the complete validation set after every full epoch.
Frozen-prefix and native-generation node chunks are at most256, so one METR-LA
window's207 nodes can share a call; this is not a larger graph-window microbatch.
Native generation chunks halve on OOM. Empty-history native rows use the fixed
fallback without running generation.
Only full validation raw mean MAE selects the checkpoint; step12 is reported.
Gradient accumulation divides the summed loss gradient by all valid labels,
not by microbatch count. Pending gradients/counts, model/optimizer/RNG, cursor,
validation journal and absolute window survive recovery. Validation journals
make early-stopping increments idempotent.

The first30 blocks are frozen-prefix computations. Complete disk cache is
estimated before allocation; this task exceeds128GiB and uses on-demand prefixes
with lossless bounded64GiB CPU LRU. No Bolt caches, truncation or quantization.
Suffix checkpointing is initially disabled; OOM at graph microbatch1 allows only
suffix checkpointing, with gradient/RNG rollback. Another OOM fails the stage.

CPU preparation and real-weight checks finish before starting the clock. A
persisted11-hour window includes idle waiting, with9hours for mechanisms/training
and2hours reserved for locked evaluation. Earlier early stopping starts evaluation
earlier. No cumulative GPU activity limit. Each new GPU child requires three
idle samples30seconds apart: no compute/graphics processes, utilization0%,
memory<=256MiB, Default mode. Running jobs do not exit when others join. Low
free memory pauses work. Project40GiB/Torch38GiB and16GiB free margin apply.
Only the project's verified process group is signaled; save begins60seconds before
the hard deadline. Recovery never extends that deadline.

`scripts/bootstrap_chattime_stage.py` is a durable CPU preparation controller:
it adopts an existing matching downloader instead of duplicating it, verifies
official hashes, runs the real-weight CPU preflight, and only then starts the
registered idle waiter. Failed download/CPU checks prohibit GPU launch. The
controller and waiter have separate PID/start-tick records and locks. Download
transport may use hf-mirror.com with checksums obtained directly from the official
fixed Hugging Face snapshot on B; unverified or different model files are rejected.
The independent environment reuses the old pinned packages read-only; actual
loaded versions, module hashes and private path configuration are captured.

CPU tiny-model and real-weight checks cover official preprocessing, initial
Persistence, causal masks/readout, cache prediction/gradients, empty nodes, batch
isolation, frozen weights, pure-text bypass, pending accumulation and exact
optimizer state recovery. CPU FP32 path tolerance1e-6; BF16 CPU/CUDA numerical
prediction/gradient tolerance2e-3 relative/absolute, registered in advance.
Saved parameter/optimizer/gradient states themselves must restore exactly; no
tolerance is relaxed after a failure. GPU mechanisms must pass fresh idle
admission before training. CPU checks set CUDA_VISIBLE_DEVICES empty and OMP/MKL2.

During joint test evaluation, native greedy prefill may be reused for GOFA only
when every prompt token is valid (left/right padding then coincides), with the
same frozen weights, token IDs, mask, positions and block30 boundary. Only the
first prefill hidden states are reused, never generated future tokens. Empty
history nodes are excluded from messages and retain the registered mean fallback.
Separate/shared predictions and gradients must pass CPU and CUDA checks first.
Prefixes are transient per window; interrupted node chunks may store at most one
partial window. Shared-window counts are reported; joint evaluation activity is
not standalone model throughput.

The best checkpoint and native decode configuration are sealed before test access.
Per-origin predictions and partial native-node cursors are persisted. Unfinished
evaluation is excluded from main tables. Report all-step and per-step raw MAE,
RMSE, MAPE valid fraction, WAPE, METR5/15/30/60min, trainable/registered/forward
module counts, updates/stop reason/activity/cache and phase memory peaks.

Limitations: single seed, prior development test inspection, unverified pretraining
data overlap, supervised regression versus native generation, and no control
separating graph/context/head gains. TITAN3.08 is only a cross-implementation
reference. Historical UFG/HSD/Bolt scores stay separate; Stage15 unperformed tests
remain blank. Language bypass preservation does not establish graph-grounded
explanations or single-checkpoint transfer. LLaMA-2 requires a later independent
stage. Notify only on terminal states, exceptions or required user action.
