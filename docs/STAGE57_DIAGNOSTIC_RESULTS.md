# Stage57 fixed source prefix/support diagnostic

Completed the preregistered16-source-query/160-forward diagnosis. Six new CPU tests passed; all accepted model/numerical files remained unchanged. No optimizer update, new free generation, target data or reserved source labels. All own processes exited. Stage57source screening remains failed and the overall goal is unachieved.

Conditional median MAE on the SAME16-query diagnostic subset(4/view), not free forecast accuracy:

|Source|Native true prefix|Real graph true prefix|Native saved-prefix condition|Real graph same saved-prefix condition|
|---|---:|---:|---:|---:|
|PEMS04flow|5.282342|4.846675|12.781963|12.347796|
|PEMS04speed|0.790305|0.623335|0.895515|0.837932|
|Boston temperature|0.813158|0.832758|4.186820|4.150889|
|Penmanshiel power|142.549952|138.142120|400.354780|400.980462|

Traffic teacher CE also improves(flow2.593976→2.381538,speed3.696153→3.309691). Under the identical Native sample0prefix, CE improves in all4views but numeric median error changes little and slightly worsens in wind. This confirms a conditional-probability/numeric-point distinction on some cases. The much larger errors under generated versus true prefixes support prefix conditions as an important contributor; true preceding answers inherently provide extra diagnostic information, so this does not prove exposure bias is the sole cause. Weather is already slightly worse under true prefixes.

At fixed physical-trained weights, replacing support with self/permutation only modestly changes teacher results: flow conditional median4.846675(real)/4.869133(self)/4.823925(permuted),speed0.623335/0.647310/0.632675. Thus the substantial adaptation is not established as road-specific benefit. Real-model mean residual/input norms are approximately20–22%at source-history block30and8–11%at block31; decoder means8–16%, max about27%, gates0.61–0.72. The branch is active and within its bound. Large residuals plus weak support sensitivity suggest shared representation adaptation dominates; they do not prove a particular causal mechanism or justify setting a new amplitude.

GPU activity18.284006seconds, project peak14,354MiB, waits/pauses0. CPU replay of160record bindings and condition metrics had max difference0. Parent checkpoint files unchanged. Evidence SHA25625c1137f31273309acbc7a8f1979105a3edb0c540cd2a247e742ba67c395e749. Runtime358dcbba7d1845d7c65e85f6c44715bdd473a8c6. Detailed source/step/top100/CDF/norm tables and existing free-forecast subset are in results/stage57-fixed-source-diagnostic-summary.json.

Next: the single fixed-weight Native-proposal replay in STAGE57_PROPOSAL_REPLAY_PROTOCOL.md. It tests whether keeping candidate predictions out of their own conditioning prefix can help, without a new training recipe, graph amplitude or architecture search. Neither favorable conditional scores nor a new replay can retroactively pass the failed Stage57screen.
