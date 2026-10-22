# Stage51 bounded sparse-road source results

Failed the unchanged initial source gate. Correct published sparse messages and static target-fixed permuted sparse messages use identical physical rosters, observations and labels; only adjacency changes. No further seed/source confirmation/target/ChatTime work.

| Condition | Flow variable-equal MAE | Speed variable-equal MAE | Selected update |
|---|---:|---:|---:|
| Published sparse road | 25.805331535 | 1.519529251 | 500 |
| Exact reused Stage40 self | 26.520034941 | 1.535240291 | 1000 |
| Static permuted sparse road | 25.799650613 | 1.502533493 | 200 |

Correct road flow is0.022% worse than permuted; speed1.131% worse. Both have quarter regressions beyond the original2% limit. No stable topology-specific advantage is demonstrated. This fixed bounded+sparse recipe ends; do not extend it or select another checkpoint.

33 related CPU checks passed with inherited full proof. All non-adjacency arrays exactly match the physical Stage38 caches; published adjacency, permuted target in/out degrees and total edge count independently checked. Source metrics replay differences0, actual maximum graph increment/input norm0.971760. Old self best tensor is exactly reused, with1000 inherited updates and0 new self updates; no duplicate training. Physical/random each received1000 new updates. Parent Stage40 whole-worker cost is provenance only, not an isolated estimate of self cost.

New controller70.846s, included worker65.139s, closing verification1.420s; GPU0. All workers/controllers exited. Full setup/transfer/document time is not included in that compute figure. New temporary autonomous window16:07:50–18:07:50 Beijing remains active for only the predeclared Stage52 follow-up; not a reset of the expired11-hour window.

Exposed source validation, single initial seed and a small diagnostic GNN; no independent-dataset or GOFA zero-shot improvement claim. All earlier results preserved.

Runtime `cdd68302d044020848861d4044950b454d809fc3`; report `de3a483d8ac1163503641085487f0c6d559dc01798d84d48586c9b63a7de3c58`; evidence `3c6d3de18287fbe00611ba138b1ccf48e856879c0b597fd65ec447dacfaedd55`.
