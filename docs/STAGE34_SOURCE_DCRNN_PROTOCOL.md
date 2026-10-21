# Stage34: source-trained inductive DCRNN comparison

Preregistered after Stage33's sealed six-target supervised result, before any DCRNN fitting or target forecasts. This comparison does not change that system or any existing scores. Desired rankings are not constraints; stronger GNN results will be reported unchanged.

Reference: [Li et al. DCRNN, ICLR2018](https://github.com/liyaguang/DCRNN), official revision602afd9d767d3aa1c9b3eac51710d6aeee12c227 and MIT license. Implement the diffusion-GRU equations independently in PyTorch: two64-unit encoder layers and two64-unit decoder layers, dual random walks, diffusion order2 with Chebyshev recurrence, shared scalar output projection, gate biases1. No node/dataset IDs. This is a cross-dataset inductive adaptation, **not an exact reproduction of the TensorFlow traffic-only experiment**.

## Source-only fitting and selection

Reuse the audited Stage23-v2 four source views and exact8192-query training pools per view: PEMS04 flow/speed12→12, Boston2017 temperature and Penmanshiel2017 power48→24. First60% source labels only; original source validation384 queries in middle20%. No target label, calibration coefficient, SCADA-only feature or target metric enters fitting or source checkpoint selection. All six targets have been previously viewed during development; zero-shot here means no target parameter updates, not an independent blind experiment.

Match GOFA-F's origin-frozen at-most9-node sampler and power/temperature/traffic histories, physical+correlation support and canonical IDs. Each node independently uses its own past-window MinMax[-.5,.5], constant scale1. Missing historical values0 plus observation mask. Two supports are the receiver-row normalized combined prior/base matrix and its separately row-normalized transpose. Pad absent nodes with zero messages/states. Input2 channels; no future covariates. Decoder predictions are unbounded continuous values, inverse-scaled without clipping.

Supervise valid source future slots for all sampled nodes (explicitly more simultaneous source supervision than the target-only LLM loss), average valid normalized absolute error within query then queries. Source teacher forcing is causal: prediction h never sees label h; missing teacher values use forward causal fill, no future-missingness token. Inverse-sigmoid scheduled sampling k2000, all-node teacher forcing as a standard seq2seq training choice. Inference is fully autoregressive, deterministic, and receives no labels.

Single fixed model/seed42. Adam lr1e-3/epsilon1e-3, clip5, effective batch64, no dropout or weight decay, at most10000 updates. Cycle flow,temperature,wind,speed,temperature,wind so three domains are equally represented and traffic views equal. Each pool reshuffles deterministically after an epoch. Validate every250 updates with no sampling. Source variable MAE/native MAE ratios, training-only epsilon floors, view geometric means and traffic-view geometric mean, then three-domain equal geometric mean S. Strict lower complete S saves best, ties earlier. Early-stop reference requires0.1% relative S gain; eight nonsignificant complete validations and at least2000 updates stop. No target-driven retries, LR tuning, architecture changes or baseline weakening.

The source protocol and optimizer are adaptations to heterogeneous transfer and must not be called the paper's published benchmark score. Reaching a limit does not establish convergence. Curves and stop reason remain visible.

## Final evaluation and fair reporting

Lock the unique best source checkpoint SHA before opening target records. Evaluate the unchanged Stage21 six×128 query lists once, same histories, masks, units and causal subgraphs; GNN produces deterministic point forecasts while Native/graph LLM use their already frozen8-sample medians. Use the same variable-equal MAE plus RMSE/MAPE/WAPE, per-step/per-variable, paired counts and2000 variable-cluster bootstrap intervals. Report actual results regardless of ordering.

Separate three categories: Native ChatTime, source-only GOFA-F3500 and source-only DCRNN; Stage33 target-supervised hybrid is a separate system with extra target fitting and SDWPF SCADA inputs. Its stronger score would not prove a fair zero-shot architecture advantage. No target score changes training or checkpoint selection.

## Acceptance and resource limits

CPU-only FP32, OMP/MKL2, CUDA disabled and verified uninitialized; GPU activity0. Actual CPU time/RAM, dataset-cache preparation, epochs/updates, source validation and conditional ETA recorded. No other jobs manipulated. Resume all model/optimizer/RNG/pool cursors/best/early-stop/activity at completed-update boundaries; no partial-gradient accumulation exists. Test directed diffusion/permutation/batch isolation, missing/constants/all-missing nodes, future-causality, teacher shift, finite gradients, state/optimizer/RNG replay and source-target isolation, tolerance1e-6.

The original11:15:12 Beijing cutoff remains. Cooperative stop three minutes before it leaves time for the already source-selected checkpoint's single CPU forecast pass; if it cannot finish by the fixed cutoff, leave incomplete metrics empty. This is a deadline safeguard for the added experiment, not a reset of the11-hour window. Do not claim convergence if stopped by cutoff. All prior failed stages remain archived.

LLaMA comparison waits for exact-version/legal-weight availability; never substitute another backbone or predetermine its rank. The user has been asked to clarify original LLaMA-7B versus Llama-2-7B while this independent GNN work proceeds.
