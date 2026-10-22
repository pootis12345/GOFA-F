# Stage52: all-node source supervision — sealed scientific failure

The last predeclared recipe in the bounded2026-10-22 16:07:50–18:07:50 Beijing campaign completed; the original source gate failed. No Stage53, ChatTime training, target prediction or UFG comparison was started. Both bounded-window recipes are now stopped; the previous target-supervised Stage33 and source DCRNN results remain unchanged.

## Fixed experiment and results

All three10831-parameter normalized sparse-road diagnostic GNNs trained freshly for1000updates, seed42, identical initialization/physical rosters/queries/auxiliary label slots and weighting. Stage51 self reuse was disabled. The dedicated trainer consumes the complete[B,9,12] output. Only source-training supervision changed; validation/selection still used target node0. This increases the supervised point budget relative to Stage51.

| Condition | Selected update | Flow variable-equal raw MAE | Speed variable-equal raw MAE |
|---|---:|---:|---:|
| physical | 1000 | 24.113416770 | 1.472902945 |
| self | 900 | 25.674400125 | 1.502888467 |
| rewired | 800 | 23.725153956 | 1.527135571 |

Correct road vs self: flow−6.080%, speed−1.995%; correct road vs static topology permutation: flow+1.637%, speed−3.551%. Thus there is useful neighborhood interaction on some source quantities, but the actual road topology does not provide a stable advantage. The final flow quarter regressed6.32% vs self and8.17% vs permuted; speed quarters also exceeded the fixed2% cap. The required≥1% mean gain over both controls and≤2% per-quarter regression were retained. Seeds43/44 and source confirmation were not opened after the initial failure.

Compared with Stage51 correct-road values25.805331535/1.519529251, all-node supervision improved the point estimates to24.113416770/1.472902945. This cross-stage observation uses an increased label budget and already exposed source validation; it is not independent confirmation or evidence that graph GOFA-F beats Native ChatTime or UFG.

## Acceptance and evidence

39 related CPU tests passed, with the previous full proof inherited; CUDA was disabled and OMP/MKL were2. Tests covered training-only labels, future/graph/input separation, variable weighting, missing/padding/constants, auxiliary gradients, full-output consumption, context restoration, and optimizer/RNG/cursor replay. Independent post-run verification passed: all non-adjacency original inputs unchanged, all controls' auxiliary arrays equal,257variables/view retained, persistence normalization independently matched exactly, all metrics replayed with absolute difference0, graph increment/input ratio at most0.976398886. All workers/controllers exited and activity.worker is null.

Valid all-node training points: flow209244, speed209088. Training-only all-node persistence macro MAE:30.4987759702/1.9415292660. No auxiliary validation or confirmation futures were built. Live recovery retained the original save cadence (update1, every50updates, and cooperative stop), plus validation/best records; all final1000-update states were saved and independently checked. The exact-resume regression interrupted after update1 and matched the uninterrupted completed state at update50. No unplanned interruption occurred in the real run.

Runtime source: `15be54c89605f526bed234465c308f7dace0299e`. Manifest: `c9a16f787e9802f57bd958c3269dc4d11a47d955b6164370f56e125da03fe332`. Report: `13f371dec18001681cad1ff99dc7647fc1a577922ee4ef1f21e9b62cccb26011`. Archive SHA256: `b61cc7e6f0a3cee5ecd96ce2671c76fbc4bb23549195d0cfed854ce97ca6710d`.

## Costs and decision

Stage52 active CPU worker90.007s; controller97.233s (prepare1.610s, related tests4.196s, worker subprocess91.419s). Peak worker RSS696.809MiB; GPU0. Stage51+52 newly active workers total155.146s; controllers total168.080s. Stage52 implementation/static inspection/commit/transfer before launch626.272s. Campaign elapsed to seal preparation2284.199s; the remaining1489.848s includes earlier preparation, heartbeat gaps, review and synchronization. Final documentation/transfer continues after this timestamp and is not mislabelled as training. Inherited Stage40 self evidence is not counted as newly trained.

Stop the automatic campaign early as predeclared. These results do not justify an expensive ChatTime or UFG run, do not rule out useful graph modules generally, and do not prove stable zero-/few-shot gains. A future study requires a new justified protocol rather than another unregistered variation on this exposed validation set.
