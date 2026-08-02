# Lab 1 — Theory

## 1. The overlap model

Let `T_bwd` be backward compute time and `T_comm` the time needed to communicate all
gradients.

Serial execution is

```text
T_serial = T_bwd + T_comm
```

With perfect overlap,

```text
T_ideal = max(T_bwd, T_comm)
```

The last bucket is not ready until the end of backward, so a useful approximation is

```text
T_real = max(T_bwd, T_comm) + T_tail
```

and the communication exposed beyond the backward computation is

```text
T_exposed = T_real - T_bwd
          = max(0, T_comm - T_bwd) + T_tail
```

For four equally spaced buckets whose communication takes half of the corresponding
backward segment, the first three transfers fit beneath later backward work and only the
last transfer remains exposed. If communication takes twice as long as each segment, a
queue of unfinished reductions grows and communication sets the critical path. Tier 1 on
a shared laptop is normally in the latter, communication-bound regime. A large model on
a fast GPU fabric is normally compute-bound enough to hide most buckets.

## 2. The bucket-size trade-off

Let `P` be total gradient bytes, `S` bucket bytes, and `K = ceil(P/S)`. Ignoring the
smaller final bucket, the ring cost is

```text
T_comm(K) = K [2(N - 1)α + 2(N - 1)/N · β(P/K)]
          = 2K(N - 1)α + 2(N - 1)/N · βP
```

The latency and bandwidth terms are therefore

```text
latency   = 2K(N - 1)α
bandwidth = 2(N - 1)/N · βP
```

Changing the bucket size does not change the total bandwidth term. It changes how many
times latency is paid and when each reduction can begin. Very large buckets minimise
latency but delay the first launch and leave a large tail; very small buckets expose
little tail but pay latency many times.

A compact objective is

```text
T_total(S) = T_bwd
           + max(0, T_comm(ceil(P/S)) - T_bwd)
           + T_ring(S, N)
```

where the last term approximates the final, unhideable bucket. At `S → P`, there is one
latency payment but essentially no overlap. At `S → 0`, overlap starts early but the
latency term diverges with the bucket count.

For the tiny preset:

```text
P  = 818,944 · 4 = 3,275,776 bytes
S* = 1,048,576 bytes (pre-benchmark prediction)
K* = ceil(3,275,776 / 1,048,576) = 4 buckets
```

A 7-billion-parameter bf16 gradient is about 14,000,000,000 bytes, or 535 buckets
at 25 MiB. For an eight-GPU ring with `α = 5 μs` and 300 GB/s effective bandwidth,
one full 25 MiB bucket costs about 70 μs of latency and 153 μs of bandwidth, so the
bandwidth term is larger. No fitted laptop-gloo `α` or `β` was supplied with this answer
set, so I do not invent them; symbolically the same bucket costs
`14α + 1.75β·25 MiB`. On gloo, both latency and slow bandwidth are large and a 25 MiB
bucket can delay overlap past the useful point, so the CUDA-oriented default is not
automatically appropriate.

## 3. Why reverse registration order

With forward-order buckets, bucket 0 contains the earliest layers. Their gradients arrive
last, so bucket 0 fills near the end of backward and almost none of its communication can
overlap compute.

With reverse order, bucket 0 contains the later layers. Their gradients are ready early
in backward, so its collective launches early while earlier layers are still computing.

Reverse registration order is only a heuristic. It can be wrong for branches that rejoin,
parameters registered early but used late, and tied weights such as an embedding reused
by the output head. Recording actual readiness on the first iteration is strictly better
because it observes the executed autograd graph rather than guessing from registration
order. Its cost is first-iteration bookkeeping, a bucket rebuild, and another rebuild if
the graph later changes.

## 4. Gradient accumulation equivalence

For mean per-example loss and equally sized micro-batches `B₁ … Bₖ`, each of size `b`,
one large batch gives

```text
∇L_large = 1/(kb) · Σᵢ Σₓ∈Bᵢ ∇L(θ; x)
```

Accumulating losses scaled by `1/k` gives

```text
∇L_accum = Σᵢ [1/k · 1/b · Σₓ∈Bᵢ ∇L(θ; x)]
          = 1/(kb) · Σᵢ Σₓ∈Bᵢ ∇L(θ; x)
          = ∇L_large
```

The equivalence fails when:

1. BatchNorm or another layer uses per-micro-batch statistics.
2. Gradient clipping or another nonlinear gradient transform is applied per micro-batch.
3. The loss couples examples or is not the same plain mean, such as an in-batch
   contrastive loss.

With `no_sync`, `k` micro-steps cost one set of bucket collectives instead of `k` sets.
Each micro-batch loss must be divided by `k`, and the optimizer must step only after the
single synchronized backward.

## 5. Effective batch size and the learning rate

N-way data parallelism multiplies the effective batch size by `N`. The linear scaling
rule multiplies the learning rate by the same factor `k` when the minibatch grows by `k`.
It assumes the larger-batch gradient has a similar direction and that the reduced
gradient noise can support a proportionally larger step.

The rule stops working beyond the problem's critical batch size, where extra examples no
longer reduce useful noise and large updates hurt optimization or generalization.

At initialization, parameter scales and gradient statistics are changing fastest, so the
full scaled learning rate can make an unstable first update. Gradual warmup raises the
step size only after those statistics settle.

The lab keeps learning rate fixed while comparing DDP with a single-process run at the
same total batch. That isolates gradient averaging. Testing learning-rate scaling at the
same time would confound a communication error with an optimizer-policy error because
both can produce the same loss-curve drift.

## 6. Predictions

The committed guesses are in `predictions.yaml`. Exact size and collective entries are
derived above; timing entries are explicitly pre-benchmark estimates, not measurements.
