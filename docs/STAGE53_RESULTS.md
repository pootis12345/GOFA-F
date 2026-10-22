# Stage53: full-observed-history exchange — completed, original gate failed

One new user-triggered experiment after the previous automatic campaign had stopped. The[predeclared source protocol](STAGE53_ORIGIN_HISTORY_PROTOCOL.md)was followed; no automation was restored. All three fresh models completed1000updates before the retained18:07:50 safety deadline. No seed43/44, unopened source confirmation, target prediction, ChatTime update or UFG evaluation followed the failed initial gate.

## What changed and what it established

The previous graph operation reads only same-time/previous3 neighbor hidden positions. The fixed variant lets each receiver history token attend all12 observed neighbor history positions, all before the immutable forecast origin. This is origin-causal encoder-style history exchange, not prefix-causal autoregressive LM caching. The original temporal convolutions and pooling could already convey some older context; the restriction was a hypothesis, not a confirmed implementation error.

Two full-history exchanges retain the bounded residual formula, all-node source supervision, point weights, source queries, physical rosters, static target-fixed permuted topology/self controls, optimization and original source selection/gates. The obsolete four-lag bias is removed in each block:10823parameters versus10831; every common initial tensor is unchanged and all controls match. No hyperparameter/feature/scale/loss search.

| Fixed condition | Selected update | Flow raw-unit variable-equal MAE | Speed raw-unit variable-equal MAE |
|---|---:|---:|---:|
| physical | 1000 | 24.554225519 | 1.499222273 |
| self | 900 | 25.674400125 | 1.502888467 |
| rewired | 1000 | 24.109828544 | 1.541705531 |

Correct topology improves flow4.363% versus self, but is1.843% worse than static permutation. Speed improves only0.244% versus self (below required1%) and2.756% versus permutation. The worst flow quarter is10.64% worse than permutation; the first speed quarter is8.07% worse than self, above the fixed2% cap. The self MAEs exactly match the independently fresh-trained Stage52 self control, consistent with the no-cross-edge operation remaining unchanged.

The new correct-road model also regresses relative to Stage52: flow24.113416770→24.554225519 (+1.828%); speed1.472902945→1.499222273 (+1.787%). These are exposed source point estimates, not new selection criteria. Reading every observed historical position did not solve the problem in this fixed experiment. Do not promote this operation to an expensive ChatTime/UFG/target experiment or interpret the failure as proof that all graph modules are ineffective. Do not append another variation in this turn.

## Verification and cost

45 related CPU tests passed; inherited full proof retained, CUDA disabled, OMP/MKL2. Added tests checked the explicit attention reference within1e−6, intended later-within-history access, final-origin future independence, no-edge equivalence, missing/empty stability, node permutation/batch isolation, nonzero gradients, parameter accounting and runtime restoration. A separate post-run synthetic replay tested the actual dedicated per-update trainer with the modified model; model/optimizer/RNG/cursor states match uninterrupted execution exactly. All real training transactions save full recovery state at every update.

Independent raw-source labels, input arrays, control label/weight equality, training-only all-node normalization and complete metric replay passed: maximum metric difference0, actual graph increment/input ratio≤0.972083807,257train variables per view,209244flow/209088speed valid training points. Auxiliary evaluation labels were not built. All controllers/workers exited, activity.worker=null, CUDA uninitialized. The external closing verifier had one local module-alias/loop-variable naming collision; it was corrected outside the immutable snapshot before evidence writing. No source model retraining/recovery or result change was involved.

New source worker active wall94.574946s; controller104.973463s (prepare1.619151s, related checks7.371770s, source subprocess95.974478s, included rather than added). Successful closing verification5.755947s separately. Peak source worker RSS721.633MiB. GPU0. Development, transfer and the failed verifier startup are outside those compute timings; exact request-to-final wall was not instrumented and is not substituted with activity time. Previous stage costs/results are not reset or double-counted.

Runtime `0bfc7e6b5fdd82ba7557f1fda7d7e280d6e20e05`; manifest `4b7911e2b33110d12442739104aa38d47ef7971fec31f1cd3944357242c21412`; report `a8aad3dae897cf204b5ca990b8016280c9c2cc4a413607fd3554d8b1a85f53c7`; evidence archive SHA256 `7a10942d1654c28690aa58e4b8bf63363ff5896d9912cd14fed651207dc86d7e`. Detailed process/path evidence remains private B/C. Published summary retains failures and no target results.

Status: sealed scientific failure, all processes exited, automatic follow-up remains off. This single small traffic-source diagnostic does not prove strict zero-/few-shot graph GOFA-F improvement or defeat recent GNNs.
