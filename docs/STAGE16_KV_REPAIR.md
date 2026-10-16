# Stage16 fixed-prompt KV CUDA repair

The user requested repair after v10 stopped on its unchanged BF16 gradient
equivalence requirement. Preserve v10's immutable snapshot, failure log and
checkpoint. Keep METR-LA12-to-12, seed42, frozen ChatTime, graph/head and all
optimization settings. This is an implementation repair, not another model
candidate or a relaxation of rtol/atol2e-3.

The old cached prefix computes30 query positions against77 key/value positions
with an explicit mask, whereas the original prefix uses77-to-77 square causal
SDPA without a mask. This is a concrete dispatch-input difference; it does not
by itself prove the cause of the observed CUDA gradient discrepancy. The pinned
Transformers4.46.3 implementation and torch2.6 source on C were audited.

The repair keeps native batch/head/sequence dimensions and tensor strides at
the attention call. Fixed-text queries are filled with zeros; their outputs are
discarded. Causal attention does not mix outputs between query rows, so these
discarded query values cannot affect retained history rows. Cached text K/V,
original absolute RoPE positions and native BF16 weights remain unchanged.
Only dynamic-token Q/K/V projections and MLPs are recomputed. Attention itself
again uses the full native square causal shape. There are no native model hooks,
shared-head edits, quantization, input truncation or output prediction changes.

The revised ephemeral format is fixed_prompt_kv_v2. Padded or nonmatching inputs
still use the original complete prefix; cached tensors remain single-copy and
read-only. All caches are rebuilt under the new source binding. Performance is
unproven until ordinary resumed training produces measurements; preserving more
attention work means a token-count ratio is not a speedup measurement.

Mechanisms compare complete boundary hidden states and record named gradient
differences/mismatch counts before asserting the original tolerances. Fresh and
actual continued-checkpoint checks write separate diagnostic JSON on success or
failure. CPU three-node results do not establish207-node CUDA equivalence. A
failed GPU mechanism still stops the phase; no automatic tolerance or model
fallback is introduced.

An explicit repair continuation accepts only an exited preflight-only failure
with no predictions, closed ledgers, valid immutable inputs and a checkpoint SHA
identical to the earlier migration. It preserves parameters, optimizer, pending
gradients/counts/loss, RNG, cursor, memory peaks and cumulative activity. Old
failure/selection artifacts are not copied over the new stage.

The already-authorized deadline amendment is inherited, not recreated: original
start2026-10-16 16:03:02, training cutoff2026-10-17 09:00, hard stop11:00 Beijing.
Every new GPU task requires three completely-idle samples10seconds apart. GPU
unknown/busy states wait; running tasks coexist when others join, with project
40GiB/Torch38GiB caps and16GiB margin. No other user's processes are touched.

B static checks and94 targeted CPU cases on C passed before the full real-weight
preflight. Full regression, deployed source and actual CPU/GPU observations are
recorded in RUN_STATUS.md and results JSON. Do not claim a repaired CUDA path or
resumed training solely from the code and CPU tests.
