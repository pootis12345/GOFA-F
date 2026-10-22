# Stage55: fixed edge/lag correlation prior — completed, original gate failed

The new user-triggered[fixed recipe](STAGE55_EDGE_LAG_PROTOCOL.md)completed all three fresh1000-update seed42source controls. Original gates were retained and failed; no seeds43/44/source confirmation/target evaluation/ChatTime/GPU/UFG followed. The user subsequently cancelled time cutoffs: this run had already finished before the old deadline and stopped for scientific reasons. Closing verification used no time cutoff; future source work must not inherit the old18:07:50 cutoff. Automatic follow-up remains off.

## Fixed model and measured effects

Retain Stage52 normalized two-exchange GNN and all-node source supervision, identical node rosters/query order/inputs/labels/weights/optimizer budget across correct sparse road, fixed target-preserving static permutation and self. Add only8zero-initialized coefficients multiplying signed historical increment correlations for4lags0–3,4per exchange;10839parameters total. Existing content attention already supports lag-dependent choices; this is an explicit historical lag prior, not a claim that previous attention could not choose delays. It does not estimate a validated physical propagation time. Features read only original12past points, are fixed at the forecast origin, and cannot directly replace prefix-causal LM KV.

| Condition | Selected update | Flow raw-unit variable-equal MAE | Speed raw-unit variable-equal MAE |
|---|---:|---:|---:|
| physical | 1000 | 24.095503905 | 1.472147025 |
| self | 900 | 25.674400125 | 1.502888467 |
| rewired | 800 | 23.728985954 | 1.527020867 |

Correct roads vs self improve6.150%flow/2.045%speed, but flow is1.545%worse than static permutation. Flow's last quarter regresses6.02%vs self and7.97%vs permutation; speed still has quarter regressions above2%. Therefore the original≥1%mean gain over both controls/≤2%each-quarter gate fails. Do not promote the small point improvements into a passed result.

Relative to Stage52 correct-road values24.113416770/1.472902945, the new values24.095503905/1.472147025 improve only0.0743%/0.0513%. This does not establish a meaningful/general improvement or justify another delay-strength/lag-range search. New coefficients did learn:

- Exchange0: `[0.12599186599254608, -0.11057742685079575, -0.19632646441459656, -0.03155931457877159]`
- Exchange1: `[0.2317633032798767, -0.14594899117946625, -0.11249867081642151, 0.07672718167304993]`

Actual source gradient norms for the two new branches are flow[0.002767917001619935, 0.0026427542325109243], speed[0.002010158495977521, 0.0019250850891694427]; all finite/nonzero. Thus missing parameter participation is not the explanation for this outcome. Self results exactly match Stage52's independently fresh-trained self MAEs; new coefficients in that control have no cross-message prediction effect.

## Acceptance, correction and cost

51related CPU checks passed with inherited full proof; CUDA disabled, OMP/MKL2. New checks: analytic receiver/sender lag direction, missing/constants, zero-coefficient exact replay with nonzero original messages, new coefficient gradients, permutation/batch isolation, future independence, actual modified-model and per-update optimizer/RNG/cursor replay, and8parameter accounting. Independent raw-source and all-control auxiliary-label/weight/input verification passed,257train variables/view; all six source metric replay differences0, bounded increment/input ratio≤0.976029098. Every real update saves model/optimizer/RNG/cursor/selection; dedicated exact-resume verification passed. All processes exited and activity.worker=null.

Stage54 prose was corrected: Stage53 flow correct/permuted95%ratio interval[1.009369,1.093370] lies entirely above1; only the other three road-specific intervals cross1. Numerical evidence and source models were never changed. The earlier all-intervals-cross wording was incorrect.

Active source worker111.087205s; controller124.421593s, comprising prepare1.552963s/related tests10.375752s/source subprocess112.484253s (included, not added). Successful closing verification6.271717s separately. Peak source-worker RSS691.508MiB; GPU0. User elapsed from the first timestamped read to seal preparation739.4s, final sync additional; development/transfer is not labelled training. Historical costs/failed results remain sealed.

Source runtime `53a54e6853576250f85f601ff124fb8dc4f13254`; manifest `b652f68b5900abedd7d1d52c4d0c52461a52821300cd049182ba649b7391383e`; report `f84cc87ff45468f2250e49a2a36c88ca0c7b0b231da08da6b1c80cb3ff34d7e8`; evidence archive SHA256 `509325de2b2aedd35b7c132f218ebff498d1a03c674e3e2312b5410816f132f2`. Already exposed18-variable traffic-source validation; no independent transfer/general graph-gain claim. This one recipe is sealed and no second variant is appended in the same turn.
