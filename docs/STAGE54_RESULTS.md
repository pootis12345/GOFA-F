# Stage54: fixed-weight source topology intervention audit

Completed the single predeclared2stage×3training-condition×3inference-topology×2view audit, without fitting/selection. All weights/checkpoints/query IDs/inputs/scales stay fixed. Reuse Stage52/53 unique source-selected seed42models; compare correct road, static target-fixed permuted topology and zero cross edges. All cases/parent hashes were sealed before the first forward. Only already exposed source validation was read; source confirmation and all six target datasets remain closed.

## The models use neighbors, but correct roads have quantity-dependent effects

For each fixed correct-road-trained model, changing inference topology gives:

| Frozen model stage | Source view | Correct road MAE | Permuted road MAE | Cross messages off MAE |
|---|---|---:|---:|---:|
| 52 | pems04_flow | 24.113416770 | 23.415863416 | 29.599590676 |
| 52 | pems04_speed | 1.472902945 | 1.515593605 | 1.607358808 |
| 53 | pems04_flow | 24.554225519 | 23.361294383 | 28.040481110 |
| 53 | pems04_speed | 1.499222273 | 1.583756289 | 1.570786342 |

Stage52 correct road versus messages-off lowers flow18.53% and speed8.37%; Stage53 lowers12.43%/4.56%. Thus the trained predictions materially depend on cross-node information. Removing messages is an inference intervention, not a separately fair retrained baseline, and can create distribution shift.

However, for those same weights the permuted topology gives LOWER flow MAE in both stages: the correct-road error is2.979%/5.106% higher. For speed the correct road gives2.817%/5.338% lower MAE. The separately permuted-trained models show the same direction: switching them to correct roads worsens flow but improves speed. This is not explained solely by choosing different trained checkpoints. It is also not evidence that arbitrary topology is intrinsically better: only one fixed target-preserving permutation was registered.

Both independently self-trained models are bitwise insensitive to all three topologies, a useful negative control. All12 matching-condition baseline/view replays and all36 independent metric recalculations match exactly; parent manifests/reports/checkpoint SHAs remain unchanged. Same physical rosters, values, edge features and all non-adjacency arrays were verified. Target in/out degree and global edge count are preserved by the permuted support. No double permutation.

## Uncertainty and limits

Ratios below1 favor correct roads.2000paired18-variable-cluster bootstrap draws, seed4254; this is an exposed validation diagnostic, with no multiplicity correction or independent confirmatory claim.

| Stage | View | Correct/permuted MAE ratio | Paired95% ratio interval | Improved/degraded variables |
|---|---|---:|---|---:|
| 52 | pems04_flow | 1.029790 | [0.984811, 1.069788] | 8/10 |
| 52 | pems04_speed | 0.971832 | [0.920960, 1.027010] | 10/8 |
| 53 | pems04_flow | 1.051064 | [1.009369, 1.093370] | 4/14 |
| 53 | pems04_speed | 0.946624 | [0.893005, 1.001314] | 12/6 |

All correct/permuted intervals cross1. Speed therefore supplies a directional road-specific signal, not a proven stable advantage. Mixed query/variable effects and the small repeatedly inspected set remain important. The audit distinguishes message usefulness from correct-topology usefulness; it does not identify the data-generating causal graph, rank ChatTime/UFG, select a new topology, or validate zero-shot transfer.

## Decision and cost

Do not continue tuning history length/residual magnitude or treat the graph as unused. The unresolved problem is extracting relationships that consistently help each physical quantity across held-out variables/times. Current evidence does not identify a validated repair. Any next training study needs a separately fixed scientific protocol and a development/confirmation design that accounts for the existing source exposure; do not turn these intervention errors into new selection targets or open the old gated confirmation after a failure. No additional training recipe was appended to this audit.

CPU internal wall1.374137s; controller sync/snapshot/execution/independent aggregation3.954816s, including the internal time (not added). GPU0, training updates0. Reported peak process RSS777.723MiB. Full user waiting from the first timestamped read to seal preparation458.1s; final synchronization is additional. Development/transfer is separated from execution, not hidden as activity time. No broad regression rerun or automation. The foreground diagnostic exited; no model or target job remains running.

Fixed runtime `dc45034bc266e22ee951a0b24c9eff62dabf5e91`; manifest `bc4db8e51a12956b0c599de703f61887aced3549665cb51bc06ce7bec4afda32`; report `a7d5c8e38cd170e2d5e29dcdad2e10b3d0be9b00468bba29d1794452ab2e9b86`; evidence SHA256 `77030f58b657c6c1e1e5497db51eb10edf258a7b630d0b82b37a1033fcebee34`. Detailed per-query records and process identity remain private B/C; public summary contains the complete comparison matrix/counts/intervals.
