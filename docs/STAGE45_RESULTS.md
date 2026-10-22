# Stage45 source scale encoding results

The fixed asinh-only intervention failed the unchanged source gate. No further seed, source confirmation, ChatTime update or target prediction was run.

| Condition | Flow variable-equal MAE | Speed variable-equal MAE | Selected update |
|---|---:|---:|---:|
| Physical roster | 24.226025976 | 1.461243676 | 1000 |
| Self only | 26.520034941 | 1.535240291 | 1000 |
| Matched-count random roster | 23.887727755 | 1.508473939 | 1000 |

Against Stage40 physical baseline, flow MAE changed +2.513%, speed -2.275%. Against the newly trained random roster, flow worsened 1.416%; speed improved 3.131%, but the last speed quarter worsened 9.044%. Speed first quarter also worsened 3.447% against self. No stable physical-roster advantage was established.

The intervention keeps unclipped pair information under a fixed monotone compression, but changes optimization geometry. Its mixed outcome does not prove clipping harmless or the earlier graph benefit causal. Random-roster performance and local time regressions remain unresolved. Do not append another encoding scan or promote this variant to expensive LLM training.

Acceptance: 33/33 related CPU tests passed, inheriting the bound full proof. Same initialization and query cursors; self best tensor exactly matches Stage40. All non-edge cache arrays unchanged; derived edge arrays hashed. Independent metric aggregation differences were zero. Maximum observed normalized increment/input norm ratio was 0.959652. Confirmation labels remained closed.

CPU controller wall 109.320s, including prepare 1.624s, tests 2.554s and source worker 105.133s; worker activity 103.710s is included, not additional. Independent closing verification 1.700s. GPU0, peak worker RSS697.75MiB. Full user wait through summary generation 738.0s includes implementation/read/transfer overhead and excludes final Git synchronization. It must not be presented as 104s total turnaround. All workers/controllers exited; no follow-up automation.

Exposed PEMS04 source development queries only, seed42, one fixed recipe, small continuous-output diagnostic GNN rather than a ChatTime numerical head. No six-target transfer or overall stable superiority conclusion. Previous sealed model/results were preserved.

Runtime revision `9e9af50dffc55a4d77761fe190c13a91f2d57477`. Manifest `101d05ad46bc15380350876efaa776043c35d17de1376fd3ae34fce5c62fea48`. Report `589e0e03135d3feb9d5f92ed3848de399967894af8e0a274648b23af77696a2a`. Evidence archive `a6c9d7301a9f60b626af8690ae550d49ddd3781b0d171d00f8aadd765f9cd8c6`.
