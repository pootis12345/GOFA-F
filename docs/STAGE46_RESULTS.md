# Stage46: paired failure concentration and road-support audit

Read-only replay of the six already selected Stage40/45 models, all 72 source validation queries per view, with independent metric agreement within1e-6. No model fitting, candidate selection, confirmation-window materialization, ChatTime or target inference. Existing predictions were not saved at query level, so a single CPU replay was necessary. All query predictions/errors and past-only history statistics are retained privately.

| Model | View | Physical better/worse queries | Better/worse variables | Physical/random MAE ratio 95% interval |
|---|---|---:|---:|---:|
| Stage40 | Flow |37/35|11/7|0.9104–1.0205|
| Stage40 | Speed |39/33|9/9|0.8884–1.0836|
| Stage45 | Flow |32/40|7/11|0.9592–1.0595|
| Stage45 | Speed |38/34|7/11|0.8920–1.0398|

Intervals use 2,000 paired variable-cluster resamples, seed4246. They are diagnostics on exposed source nodes, not a substitute for the registered multi-seed gate. All cross1.

Stage40 flow: variable114 contributes90.17% of the net physical-over-random gain; excluding it diagnostically leaves only0.327% improvement. Stage40 speed: variable48 contributes365.88% of net gain, offsetting regressions elsewhere; excluding it gives3.300% degradation. Stage45 speed has the same concentration: excluding48 gives0.305% degradation. These sensitivity calculations do not remove variables from any official metric or select another checkpoint.

Past lag correlations do not show a strong universal neighborhood distinction. Mean best signed correlation over lags0–3 is0.4694 physical versus0.4458 random for flow, and0.3286 versus0.3307 for speed. These use only pre-origin histories with minimum6 overlapping observations and do not establish causal predictive value.

The validation rosters contain9 nodes but original published degree is1–2. Of5,184 dense directed message pairs,1,152 are published road edges:77.78% are multi-hop/non-road direct messages. Sampled target-neighbor distances reach8 hops;41.67% exceed3 hops. Training has degree1–7 and only17.00% of sampled target-neighbor pairs beyond3 hops. This is structural distribution shift and a possible message-mixing concern, not an implementation error or proven failure cause. Dense roster support was deliberate in Stage38. Stage37 already tested unnormalized direct published-road support unsuccessfully; do not misrepresent this audit as the first discovery of road edges or assume simply restoring one-hop support will solve it.

The next justified fixed test is a Stage40-based soft shortest-road-distance attention prior, rather than another scale encoding or state-feature search: add -log(hop distance) to existing cross-node attention logits; retain all nodes, two blocks, loss, initialization, queries and budget. Compare correctly aligned distances with target-fixed permuted distances on identical histories/rosters (same per-target distance multiset), plus exact self and the dense baseline. Use the old gates; no coefficients to scan. This is a proposed source experiment, **not executed by this read-only audit**, and no advantage is claimed. All source-validation development exposure must remain disclosed.

Synthetic correlation/missing checks and directed-road shortest-distance checks passed. Original parent reports and all six selected checkpoints remained SHA-identical; foreground processes exited. Audit CPU wall2.085s plus postprocessing1.035s, excluding interpreter startup, implementation and transfer. GPU0, training0, no automation. Overall elapsed through summary generation625.8s is separate and excludes final synchronization.

Manifest `7f97c00cee7f6a18f9a88d057426881852acc5c28718da129c52bf198e68a9f6`; report `c3f5ccea52c94674ccd02bf4f51023533cbdfab9e600afca3c8ea263647d1935`; evidence archive `154d9953172939d354c11d072c94d6bde2788a6b602cee2758f30503a8a48ced`. Detailed source queries and server identities remain in private B/C evidence.
