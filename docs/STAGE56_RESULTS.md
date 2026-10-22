# Stage56 graph-memory mechanism acceptance

The actual shared frozen ChatTime graph-memory prototype passed its mechanism checks. This is implementation progress toward the real-graph zero-/few-shot goal; that performance goal remains unachieved. No new source accuracy or target predictions were computed.

-32related CPU checks passed, no skips, CUDA disabled and OMP/MKL2. v1 had30passing tests before a duplicate bootstrap metadata keyword stopped it; v2 repairs that bookkeeping and adds actual sampled-path tests. Old evidence is preserved and source reservations match exactly.
-All sampled nodes traverse the same frozen ChatTime. Graph memories are built before blocks30/31; the second is recomputed after the first exchange. Decoder reads depend on causal numeric-slot count. No numerical output head, target calibration or CPU ensemble is used.
-Initial graph-on/native logits were exactly equal on four real source shapes. After three discarded optimizer updates,9-node12→12traffic and9-node48→24weather passed full/compact gradient comparisons at unchanged BF16 tolerance2e-3. Full/incremental teacher logits and separately seeded top100sample trajectories were bitwise equal on these cases. CPU tolerance remains1e-6. Native parameters stayed frozen. Trainable parameters:1,061,066.
-GPU activity24.462985seconds, model-code timer21.505099seconds, project peak14,964MiB, waiting0, resource pauses0. All own controllers/workers exited. No ongoing automatic follow-up. QA weights are discarded, not a selected trained candidate.

These checks show the intended path executes consistently. They do not show forecast improvement, beneficial real topology, stable transfer, or superiority to Native ChatTime/UFGTime. Only one trajectory per nonzero shape was replayed for consistency; this is not a sampled accuracy estimate.

Continue the original route with a narrower next gate: one predeclared source-only pilot comparing fresh, same-budget real-graph/self-memory/static-permuted controls with identical nodes and labels. Validate graph-specific gains before scaling training; replicate and use the sealed source holdout before any target evaluation. If physical topology is not reliably useful, audit graph semantics/direction and temporal alignment rather than adding endless recipes. Coordinate4NN weather/wind graphs are proxies and require that limitation.

Lock a source-selected model before separately evaluating zero-shot and16/64support-query few-shot branches. The no-graph adapter and recent GNN must receive matched label budgets; UFGTime needs a documented adaptation for its node/frequency-sized official parameters. Use fresh sealed times/stations, multiple training seeds, and spatial/time-block paired uncertainty. Old six-target scores remain development re-evaluation.

Evidence archive SHA256:d00f683c35d93e7403c8ce95a6e4a1ab7d36440106bdfa4fd1dee47aabaabaa6. Runtime source:f58fe5683d7b6ed88529a802ebf3383d31dd0351. Full sanitized evidence:results/stage56-graph-memory-mechanisms.json.
