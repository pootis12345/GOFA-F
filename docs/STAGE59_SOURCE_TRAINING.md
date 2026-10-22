# Stage59 formal source training: one connected source-only experiment

This adds formal training to the already accepted dedicated-Memory architecture.
It does not change the protected Native, the accepted tensor kernel, sampler,
metadata skeleton, four source pools, continuous head or the six-target goal.
The old mechanism output stays immutable. Runtime/source output is independently
named `stage59-dual-memory-source-v1`; hourly automation remains disabled.

## Inherited proof and common data

Require parent manifest6fee1a93b56fe5757a0bb6eb9635e923132e9a6e876f01f197eabfcd1554d9b3
and real7Bproof6622dc6b1b2a5e44647380e7fbe9a4c166fed2deb762ac9cfc6efd1704d6ef18.
The accepted execution/input files must hash-identically. Do not repeat the
completed Native/31-node tensor proof when its code has not changed.

Serialize the exact source input rosters once in the Native runtime. Both
engines read these same hashed shards; UFG does not independently resample under
its different Torch version. Cache only train/development labels, never source
confirmation labels or target observations. All valid future slots of sampled
real nodes receive the same loss eligibility in both engines. A node with an
entirely missing origin is ineligible; missing/padded slots never enter loss.

The query loss is raw-unit absolute error averaged over valid slots of each
eligible real node, then over nodes. Divide by the per-view, training-only mean
absolute consecutive observed difference already sealed in the parent. Across
queries cycle flow/weather/wind/speed/weather/wind; this makes three domains
equal over cycles and flow/speed equal within traffic. Both shapes are trained
within each view, alternating deterministically. PEMS04 channels remain one
source group in every leave-group-out fit.

## Finite model fitting and depth selection

Use the already fixed AdamW1e-4,wd1e-4,clip1,batch8,micro1,warmup50. Freeze ALL
graph-exchange parameters during the first50interface-only updates, then train
the existing additions jointly. Native and backbone remain frozen. Numerical
Memory queries retain autograd through the frozen tail. The continuous expert
is not constrained to the Native vocabulary interval.

Initial1024queries/view are512per shape. Build nested, variable-balanced/time-
spread pools using variable round-robin and bit-reversal of sorted lawful time
indices. On completing an active shape pool, double it, up to the sealed4096.
Shuffle only using source/shape/epoch-bound local RNG; save its exact cursor.
This avoids starting solely in the earliest part of the source training period.

For each of depths6and3, train three leave-group-out models. Checkpoint selection
uses only development labels of the groups included in that fit. Its score is
the geometric average of per-view variable-equal raw MAE divided by its sealed
training unit normalizer, then domains equally (two traffic views equally).
Strictly smaller saves best; separate0.1%early-stop reference, minimum400,
patience5,validation every100,maximum2000. Retain an update0initial reference and
report if it remains best. A cap is not a convergence claim.

Only after a fit's unique source best is locked may it predict the held group's
development list. No held-source score returns to that fit's checkpoint rule.
Depth selection uses all three out-of-fit groups: minimize worst domain MAE
ratio to Native8sample median, then geometric domain ratio, then shallower
depth on exact ties. Floor a Native view denominator by max(1e-12,1e-9 times its
training unit normalizer). Train one fresh final all-source model at this depth
under the same finite rule. No target labels or scores enter any choice.

## Shared fusion

Fit ONE9parameter logistic gate on the selected depth's cross-fitted source
predictions. Eight input features: H/24,L/48,target historical missing rate,
neighbor historical coverage, normalized historical standard deviation, signed
asinh(expert disagreement/origin extent), its magnitude, and relative forecast
step. No dataset or variable ID, and no future missingness/state features.
Fuse in raw units without clipping. A root without any observed direct external
neighbor returns Native exactly by a prediction-time hard rule.

Fit CPU float64 Adam,lr0.02,2000steps,intercept initial−2,other coefficients0.
Objective is variable-equal raw MAE normalized by each view's Native macro MAE,
domains equally weighted, plus0.001mean enhanced weight. No hyperparameter
search or target-dependent coefficients. This is a source-trained shared gate,
not the earlier target-calibrated or target-supervised hybrid. Seal expert,
depth,gate and all inputs before source confirmation. Confirmation/targets are
not exposed by this training entry; completion of training is not goal success.

## Strong UFG source baseline

Retain pinned official effective FreqTimeUFGV2,width128,31roles,frequency graph
and two shape-specific weights. Same serialized histories and eligible all-node
labels. Adam1e-3,batch64,micro64,clip5; source validation every250,min2000,
patience8/0.1%,max10000. Use the same source-metric aggregation and strict best
selection, with separate early-stop reference. Source curves and stop reason
must show whether the finite budget plateaued; do not claim convergence merely
on reaching a cap. Default CPU2threads allows independent progress while the
Memory fits use the resource-supervised GPU. No target update or extra SCADA.

The node/frequency parameters operate on sampled-roster roles, not permanent
node IDs. Masked-zero trailing pads and shape-specific weights are explicit
transfer adaptations, not a claim to reproduce original per-dataset scores.
Do not shrink UFG inputs/capacity, stop it for ranking, or suppress its results.

## Recovery, cost and progress

Each completed update atomically preserves model,optimizer,RNG,cursors and
selection. On cooperative pause, also preserve partial gradients and pending
query count. Autograd produces separate microgradients before committing them,
so a failed backward cannot corrupt prior accumulation; cursor commits occur
only after successful gradients. OOM may reduce microbatch; at micro1stop with
exact saved state for a memory-strategy repair, never reduce the sealed graph.

Frozen pre-graph history cache:2GiBCPU RAM,8GiBbounded disk, weight/depth/token/
mask/position-bound and content hashed. Never cache graph Memory or graph KV.
Cache pruning is confined to this version's derived cache. No old result or
snapshot is edited. Memory workers retain40GiBproject/38GiBTorch and10saudits,
memory_cap_only; CPU UFG has CUDA disabled and OMP/MKL2. No time deadline.

Show fit/depth/update/pending/validation and per-engine progress. After10updates
estimate remaining training and validation; the read-only progress tool adds
remaining Native development sampling, source confirmation and target inference
costs conditionally. Estimates exclude unknown resource pauses, may shorten by
early stopping, and are not hard limits. Reused costs remain inherited, not
counted again as newly generated work. Final target accuracy remains unproven
until all six MAEs strictly beat both fair baselines on sealed evaluation lists.
