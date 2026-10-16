# Stage16 shared GPU admission amendment, 2026-10-16

The user supplied the active pi05_libero policy-server script and authorized
starting GOFA-F without waiting for full GPU idleness if the server does not
functionally require exclusive use. The script loads a policy and serves
WebSocket inference; it does not request exclusive CUDA mode. This supports
coexistence, not a guarantee of unchanged inference latency. If the server is
used for latency measurements or deadline-sensitive control, concurrent training
can affect those measurements and behavior.

The example's `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` concerns JAX memory
preallocation, not compute or latency isolation. See the
[official JAX allocation documentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
No other user's process, settings, environment or files were changed.

Read-only observation at18:13 Asia/Shanghai: H100 PCIe81559MiB total,
20837MiB used, **60253MiB actually free**, Default mode, GPU utilization30%.
The active openpi-server process used20824MiB. Use reported free memory, not
total minus used, because driver reservations also occupy capacity.

The registered `--shared-gpu` override applies only to this Stage16 deployment.
Original strict-idle defaults remain for other callers. Every new GPU task
requires three successful checks30seconds apart: Default mode and at least
**57344MiB/56GiB actually free**, sufficient for our40GiB group cap plus16GiB
margin. Existing compute/graphics processes and nonzero utilization are allowed.
Missing/failed queries, insufficient space, UUID changes or non-Default mode
fail closed. During execution, foreign tasks joining do not trigger exit; low
free memory pauses updates. Torch38GiB, group40GiB and16GiB margin remain.
These limits do not allocate a fixed fraction of compute to either application.

The stopped v5 CPU waiter had zero GPU activity and no checkpoint/prediction
artifacts. v6 adopts its original start16:03:02 and shortened absolute deadlines:
**training ends2026-10-16 21:30, hard stop23:30 Asia/Shanghai**. There is no
new time allowance. The admission override is bound into the window, launch
fingerprint and manifests; resuming under another policy is rejected.

Source `145789cc218cc5b9f9071ef92ad1ad1a3993909d`, fixed artifact
`stage16-chattime-metr-v6`. The backbone, graph, continuous head, dataset,
optimizer, sample order and numerical tolerances are unchanged. CPU regression
and real-weight checks still precede GPU mechanisms. A GPU mechanism failure
stops the stage; no tolerance relaxation or new architecture candidate is allowed.

At18:27, the real207-node CUDA mechanisms passed without changing tolerances;
preflight peak allocated30.1GiB/reserved35.0GiB. The scheduler entered training
while the external policy server remained present. CPU290 passed/11 old optional
skips and54 new cases covered. Evidence: `results/stage16-shared-v6.json` and
`results/stage16-cpu-verification-v6.json`. This establishes the mechanism path,
not prediction accuracy or unchanged external-service latency.
