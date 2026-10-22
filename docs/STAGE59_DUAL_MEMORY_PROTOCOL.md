# Stage59: protected Native plus dedicated bidirectional graph Memory

This is a new source-only experiment, not a restart of Stage58 and not a claim
of target success. The persistent goal requires six target MAEs below BOTH
Native ChatTime and adequately trained UFGTime. A mechanism pass is insufficient.

## Fixed architecture and information boundary

Every real node executes the same frozen ChatTime7B. Native's official prompt,
origin MinMax, finite grammar, BF16 kernel,8independent samples and pointwise
median remain unchanged. No graph hook exists in Native. Its local sampling RNG
and KV are separate. Retain historical4source exact sampled-token references.

Append4trainable soft Memory vectors per node at the frozen-prefix boundary.
They read the prefix summary, a GRU over ALL raw-history values/masks/times, and
separately frozen-encoded task metadata. They are not new vocabulary tokens.
Main depth6interleaves before zero-based blocks26–31; sole depth3candidate
interleaves before29–31. One GNN exchange is one-hop same-slot graph attention
with scale/offset/hop edge features, residuals and zero tanh gates. Only Memory
is modified. Each subsequent frozen LLM block processes it, and the next GNN
reads that updated state. Do not detach Memory between exchanges or run its tail
under no_grad. Ordinary historical positions cannot attend appended Memory.

The shared continuous head reads final Memory and the complete raw-window GRU,
directly predicts allH, and inverse-transforms without range clipping. Its
initial output is persistence, not Native. Zero graph gates guarantee zero graph
residual, NOT equality of the new numerical head to Native. Parent3500 and all
earlier adapters are not used to initialize additions.

Reference tail executes all history+Memory positions. Optimized tail recomputes
pure frozen historical KV and executes Memory queries with gradients. Accept
only after independent output/gradient comparisons, FP321e-6/BF162e-3. No shared
fallback is permitted to hide divergence. Prefix caches bind weight snapshot,
execution kernel, depth, IDs, masks and prompt boundaries, never graph parameters.

## Sampling and metadata

Pin TAGLAS e30dca89cd89a238ba3b935121d516bee7c27548,utils/graph.py,
k_hop_subgraph dense single-root logic. Incoming ancestor fringe is truncated as
a WHOLE to10at each of3hops, BEFORE final union/deduplication. Repeated fringe
entries retain official dense semantics. Pin receiver-major edge enumeration.
Target role0then stable global IDs; retain induced original edges, direction,
sample seed, hop and sampled parent path. At most31REAL nodes; no fabricated
nodes, Memory/prompt/padding do not count. UFG receives the exact same roster.
PEMS04 retains the audited publisher bidirectional convention; coordinate4NN
weather/wind graphs are disclosed constructed proxies, not measured causal edges.

Task text is one fixed skeleton with physical quantity, verified/explicitly
unknown unit, observed frequency,L,H. Dataset name is not a trainable identity.
Metadata are encoded separately into Memory; Native prompt is untouched. No
target-error-based wording or sampling changes.

## Source pools and finite fitting protocol

Use audited Stage23-v2 source manifest9d2b5cb1d9c2856adda7d53699379d9eaf4b86a380e8051da9616f0c8ec4f224.
PEMS04flow+speed are ONE source group; Boston2017and Penmanshiel2017are the
other two. All four views contain BOTH12→12and48→24. No SCADA covariates.
Retain train/validation/confirmation traffic node partitions; time60/20/20.
No labels cross partitions/segments. Seal8192training queries/view,4096/shape,
and32development+32confirmation queries per view/shape. Training windows may
overlap; evaluation windows for the same variable cannot overlap prior sealed
source development/confirmation windows. Old source development exposure is
disclosed. Preparation reserves but does not read confirmation labels.

Source-only global capacity selection compares depths6/3with three leave-group-
out fits; traffic channels are held out together. Within a fit, use ONLY its
training groups' middle20%for checkpoint selection; held-out group predictions
are produced only from that locked best. Depth selection minimizes worst held-
group ratio to Native, then geometric mean, then shallower depth on exact ties.
Do not require each source/time block real>shuffle as a new advancement gate.

AdamW1e-4,wd1e-4,clip1,effective batch8,micro1,warmup50. First50interface-only
updates hold graph gates at0; thereafter train additions jointly. Query loss is
continuous absolute error averaged over valid observed-origin node/target slots,
with fixed per-view training-only physical-unit normalization; domains equally
weighted, flow/speed equally within traffic. Missing auxiliary origin histories
receive no supervised labels in either model. Start1024queries/view and expand
the sealed pool deterministically on completion of its training epoch, at most
8192/view. Every100updates source validation; minimum400updates, patience5with
0.1%significant improvement; maximum2000. A cap/early stop is not a convergence
claim. Record training/pool curves, not only the chosen point.

After global depth selection, train one final all-source expert under the same
finite rule. The shared fusion is trained only on cross-fitted source predictions
and Native8sample medians; no target-specific parameters. Source-only reliability
features: historical missingness/variation, neighbor availability, normalized
expert disagreement and forecast distance. No variable/dataset IDs. Shared convex
fusion and exact Native fallback on invalid graph/head; no outcome-dependent
oracle, clipping sweep or forced graph weight. Freeze gate before unopened source
confirmation. Report Native, enhanced, fusion and UFG separately regardless of
rank. Source confirmation is an independent generalization check, not a target
selection device; report failures honestly and do not silently reuse it to tune.

UFGTime pins61f55135bc2bd6e9b9e5d592009d358015eea681,unchanged effective
FreqTimeUFGV2,official width128,31roster roles,explicit masked-zero trailing pads,
own frequency graph. Two shape-specific checkpoints are disclosed. Same source
query pool, real node roster, primary histories and valid all-node future labels;
no target labels or extra features. Adam1e-3,batch64,max10000,source validation
every250,min2000,patience8/0.1%; retain training curves/unique source best. Do not
reduce inputs/capacity or stop it merely to obtain the requested ranking.

## Target gate, execution and deliverables

Only after source selection+confirmation and sealed deployment hashes, create
and seal the new evaluation query lists BEFORE reading their errors. Target data
are PeMS03/BAY12→12and MeteoNet/EngRAD/SDWPF/Kelmarsh48→24. No target labels
for fitting, calibration, text, depth, sampling, hyperparameters or checkpoints.
Old target development exposure is disclosed; new samples do not erase it.
Per-target variable-equal raw MAE must be strictly below BOTH baselines by
max(1e-12,1e-9*the smaller baseline). Otherwise the goal remains unachieved.

Related CPU tests precede the one real7Bacceptance. Check31real nodes,maxL48/H24,
all-node future causality, Memory gradient paths, exact Native sampled tokens,
cache/checkpoint equality and numerical tolerances. First64source queries give
actual throughput/cache footprint before ETA. Auxiliary nodes never autoregress.
Shared admission memory_cap_only,40GiBproject/38GiBTorch,10saudits,cooperative
save≤60s,only verified own processes. No wall/activity deadline or idle wait.
Never shrink a sealed roster to fit. Preserve optimizer/partial gradients/RNG/
cursors/selection/evaluation shards/costs across resource recovery. No heartbeat.

Current implementation status must distinguish preparation,mechanisms,formal
cross-fitting,gate,confirmation,UFG,and target stages. The first implementation
commit includes source preparation+execution acceptance; formal runners remain
pending until implemented, never counted as complete by green mechanism tests.
