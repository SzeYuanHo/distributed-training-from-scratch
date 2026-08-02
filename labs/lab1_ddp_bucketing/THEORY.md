# Lab 1 — Theory

Five derivations. Do them before step 3. Blanks marked `____` are yours.

Carry α and β forward from Lab 0. If you have not fitted them yet, go back; every
numeric prediction here depends on them.

---

## 1. The overlap model

Let `T_bwd` be backward compute time and `T_comm` the time to communicate all gradients.

**Serial** (all-reduce after the backward pass finishes):

```
T_serial = ____________________
```

**Perfectly overlapped** (communication hidden entirely behind compute):

```
T_ideal = ____________________
```

Neither is achievable, and the reason is the tail. The last bucket cannot be sent until
the last gradient exists, which is at the very end of the backward pass. So:

```
T_real = ____________________ + T_tail
```

where `T_tail` is the time to communicate the final bucket with nothing left to overlap
it with.

**Exposed communication** is the part that could not be hidden:

```
T_exposed = ____________________
```

Sketch the timeline for 4 buckets where each bucket's communication is half the time of
the backward segment that produces it. Then sketch it where communication is twice as
long. Which regime is your laptop in at Tier 1? Which is a real GPU cluster in?

`_______________________________________________________`

---

## 2. The bucket-size trade-off

`P` = total gradient bytes. `S` = bucket size in bytes. `K = ceil(P/S)` = number of
buckets.

**Total communication cost**, using the ring all-reduce cost from Lab 0 for each bucket:

```
T_comm(K) = ____________________
```

Expand it into the α term and the β term separately:

- α term: `____________________`
- β term: `____________________`

**Notice what the β term does as K changes.** Write down what you see:

`_______________________________________________________`

So total *bytes* are independent of bucket size, and total *latency* is not. Then why is
there an optimum at all, rather than "make buckets as large as possible"?

`_______________________________________________________`
`_______________________________________________________`

The answer is the tail from section 1. Write the combined objective:

```
T_total(S) = ____________________
```

Differentiate with respect to `S`, or just reason about the two limits:

- `S → P` (one bucket): `____________________`
- `S → 0` (one bucket per parameter): `____________________`

**Your predicted optimum** for the `tiny` preset with your Lab 0 α and β:

```
P    = ____ bytes         (0.82 M params, fp32; work it out)
S*   = ____ bytes
K*   = ____ buckets
```

Put `S*` in `predictions.yaml` as `optimal_bucket_bytes`.

**PyTorch defaults to 25 MiB.** For a 7-billion-parameter model in bf16, that is
`____` buckets. On NVLink, α is around 5 microseconds and effective bandwidth is
hundreds of GB/s. Compute the α term and the β term for one 25 MiB bucket on that
hardware and say which dominates:

`_______________________________________________________`

Now do the same with your gloo numbers. Does 25 MiB still make sense?

`_______________________________________________________`

---

## 3. Why reverse registration order

The backward pass computes gradients in reverse topological order of the forward graph.
For a plain feed-forward stack, that means the last layer's gradient is ready first.

`model.parameters()` yields parameters in registration order, which for a normally-written
module is roughly forward order.

**If you bucket in forward order**, bucket 0 holds the *first* layer's parameters. When
does bucket 0 fill?

`_______________________________________________________`

And therefore how much of its communication can be overlapped?

`_______________________________________________________`

**In reverse order**, bucket 0 holds the last layer's parameters, which are ready
`____` in the backward pass, so its collective launches `____`.

Draw both timelines for a 4-layer model with 4 buckets. The picture is the argument.

### 3.1 Where reverse order is a bad guess

Reverse registration order is an *approximation* of readiness order, and it can be a poor
one. Name an architecture where it breaks:

`_______________________________________________________`

(Prompts, if you need them: a model with a branch that rejoins; a model where a parameter
defined early is used late; weight tying between the embedding and the output head, which
this repo's reference model has.)

Real DDP handles this by recording the actual readiness order during the first iteration
and rebuilding the buckets to match. Why is that strictly better than any static
heuristic, and what does it cost?

`_______________________________________________________`

---

## 4. Gradient accumulation equivalence

Claim: for a mean-reduced loss, `k` micro-batches of size `b` with the loss scaled by
`1/k`, accumulating gradients and stepping once, is equivalent to one batch of size `kb`.

**Prove it.** Let `L(θ; x)` be the per-example loss and `B_1 ... B_k` the micro-batches.

One large batch:

```
∇L_large = ____________________
```

Accumulated micro-batches, each loss scaled by `1/k`:

```
∇L_accum = ____________________
```

Show they are equal:

`_______________________________________________________`

**Now break it.** Name three things that make the equivalence false:

1. `____________________`
2. `____________________`
3. `____________________`

(One of them is in almost every convolutional network ever published. Another shows up
the moment you use gradient clipping. The third involves any loss that is not a plain
mean over examples.)

**Why this matters for `no_sync`.** Inside `no_sync`, gradients accumulate locally with
no communication. On the sync step, one all-reduce covers all `k` micro-batches. So `k`
micro-steps cost `____` collectives instead of `____`.

The catch: which loss scaling gives you the same answer as `k` separate DDP steps? Write
it down, because getting it wrong is a silent bug:

`_______________________________________________________`

---

## 5. Effective batch size and the learning rate

N-way DDP multiplies the effective batch by `____`.

**Linear scaling rule** (Goyal et al. §2.1): when the minibatch is multiplied by `k`,
multiply the learning rate by `____`.

State the assumption it rests on:

`_______________________________________________________`

And the regime where it stops working:

`_______________________________________________________`

**Gradual warmup** (Goyal §2.2): why is a large learning rate especially dangerous in the
first few hundred iterations, and what does the warmup schedule actually protect against?

`_______________________________________________________`

**A note on this lab's tests.** The equivalence tests hold the learning rate fixed and
compare an N-rank run against a single-process run *with the same total batch*. That
isolates "is my gradient averaging correct" from "did I scale the LR correctly", which
are two different bugs with the same symptom. Say why testing them together would be a
worse test:

`_______________________________________________________`

---

## 6. Predictions

Fill in `predictions.yaml` now. The bucket-size sweep will take a few minutes and you
want your guess committed before it finishes.

The one people get most wrong is `tier1_speedup_vs_single_process`. Think about it
properly rather than writing 1.0: four ranks on one laptop share the memory bus and the
cores, so you have added communication without adding compute. Predict a number below 1
and say how far below.
