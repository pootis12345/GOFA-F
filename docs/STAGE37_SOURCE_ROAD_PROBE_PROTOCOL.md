# Stage37: one paired study of the upstream road graph convention

Preregistered after Stage36's source-only initial gate failed, before any Stage37 fit. Parent reportSHA `7a9fcb2c745f536759e7d1c21bb476f106b3643a28aeaae65a0e8ac0cc461731` is immutable. Parent three models each completed1000 updates with matched initialization/order; no seed43/44, source confirmation or target forecast occurred. All parent workers exited. This is a NEW source-development study, not an exact recovery or a relabeling of the failed experiment.

## Why this single factor merits checking

The pinned upstream ASTGNN[PEMS04 config](https://github.com/guoshnBJTU/ASTGNN/blob/9c2e19b98c4cedf1f35214d8789685b6381b3aad/configurations/PEMS04.conf) sets`direction=2`. Its[training script](https://github.com/guoshnBJTU/ASTGNN/blob/9c2e19b98c4cedf1f35214d8789685b6381b3aad/train_ASTGNN.py) explicitly calls`get_adjacency_matrix_2direction`; the pinned[loader](https://github.com/guoshnBJTU/ASTGNN/blob/9c2e19b98c4cedf1f35214d8789685b6381b3aad/lib/utils.py) symmetrizes endpoint pairs. The locally audited CSV has340 nonreciprocal rows; retaining CSV direction is a possible convention, but cannot be called the upstream default or a verified causal traffic direction. No claim that this mismatch is the established root cause of failure.

Keep Stage36 sealed and check the official benchmark's bidirectional convention ONCE: binary`A or A.T`,680 directed message entries, no self edge in the cross-node module. This changes only source message support. Node components and BFS rosters are invariant under this symmetrization. All source observations, masks, physical units, queries, input histories, edge scale/offset features, seeds, parameter initialization, architecture, optimization, step budgets, source selection and unchanged gates stay exactly paired. Original raw/prepared data are read-only. No source checkpoint or optimizer is reused; all three controls train fresh with the same budgets, providing an independently verifiable identical self-control trajectory across stages.

Upstream revision`9c2e19b98c4cedf1f35214d8789685b6381b3aad`; pinned SHA256:

- PEMS04.conf:`70fbf3ed92805287c714042ff13de6766fabcb8206ff68525cf2cb4ba4b9ce31`
- train_ASTGNN.py:`5c4c25a9fe9425f5cefc9aa658cc23f7c35368fb6e52fd823fc121aba0cc7e89`
- lib/utils.py:`3ac4575c8a2bfb044cb8751e97e213e088ab94dcfba17b59dfd623dcba90114f`

Only small upstream code/config evidence is downloaded, no new observations. Verify these hashes in CPU preparation. Stage36 source validation is already exposed and legitimately reused for this documented source-development factor; it is not a fresh blind evaluation. The32-node confirmation remains unopened and presealed; require identical full queries/partition to Stage36, parent worker=null and final verification passed before preparation.

## Execution and stopping

Same[Stage36 protocol](STAGE36_SOURCE_GRAPH_PROBE_PROTOCOL.md), including seed42 screen, conditional43/44 repeats, checkpoint lock then32-node source confirmation,1% per-view improvement over both matched controls,2% time-block limit, paired intervals, exact zero cross-edge identity. No gate relaxation, different model, lag scan, added samples, reverse-only candidate or target-driven return. If this single additional graph-convention study fails, end this CPU mechanism investigation without automatically appending another recipe or expensive LLM work. Passing would justify a further source transfer check and bounded ChatTime interface pilot; no presupposed six-target success.

Independent snapshot/output`stage37-source-road-probe-v1`, same generic source-only entry`gofa_f.source_graph_probe` with`configs/stage37_source_road_probe.yaml`. Source-policy and stage metadata independently bound. CPU FP32 OMP/MKL2, CUDA disabled, full CPU regressions before fits. No GPU/target/ChatTime updates or hourly followup. Exact original kernel/model file is unchanged. Save all old failures/costs; record this stage separately and cumulative CPU costs explicitly. Do not edit old snapshots/results or claim graph conventions are identical experiments.
