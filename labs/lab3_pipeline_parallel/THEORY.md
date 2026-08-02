# Lab 3 — Theory

Sections 1 to 3 before you write code. Section 4 before step 5.

`p` = number of pipeline stages. `m` = number of micro-batches. `t_f` and `t_b` are the
forward and backward time for one micro-batch on one stage.

---

## 1. Counting the bubble

Draw the AFAB schedule for `p=4`, `m=4` on graph paper. Time along the horizontal axis in
units of `t_f`, one row per stage. Fill in every forward slot, then every backward slot.

Count the empty cells before stage 3 starts its first forward:

```
idle slots on stage 3 at the start = ____
```

And the empty cells on stage 0 while it waits for the backward pass to reach it:

```
idle slots on stage 0 at the end = ____
```

**Total pipeline time**, in units where one forward and one backward each take one slot:

```
T_pipeline = ____________________
```

**Ideal time** with no bubble, meaning every stage busy the whole time:

```
T_ideal = ____________________
```

**Bubble fraction** is `(T_pipeline - T_ideal) / T_pipeline`:

```
bubble = ____________________
```

Simplify. You should get something that depends only on `p` and `m`:

```
bubble = ____
```

The same lost time is also reported as an **overhead ratio relative to ideal
work**, `(T_pipeline - T_ideal) / T_ideal`. Keep the denominator in the name:
the overhead ratio is `(p-1)/m`, while the elapsed-time fraction is
`(p-1)/(m+p-1)`.

**Sanity checks.** Fill these in from your formula:

| p | m | bubble |
|---|---|---|
| 4 | 1 | `____` |
| 4 | 4 | `____` |
| 4 | 16 | `____` |
| 8 | 16 | `____` |

What does `m = 1` mean physically? `_______________________________________`

Why can you not simply make `m` enormous? Two reasons, and the second one is the whole
reason 1F1B exists:

1. `____________________`
2. `____________________`

---

## 2. Activation memory

**AFAB.** Every micro-batch's forward runs before any backward starts. Stage `i` must
hold the activations of how many micro-batches at peak?

```
____
```

So peak activation memory scales as `O(____)`.

**1F1B.** After a warmup phase, each stage alternates one forward and one backward. Write
the warmup length for stage `i`:

```
warmup(i) = ____________________
```

Check it: at `p=4`, stage 0 does `____` forwards before its first backward, and stage 3
does `____`.

In the steady state, how many micro-batches does stage `i` have in flight?

```
____
```

So peak activation memory scales as `O(____)`.

**The result to remember.** Same bubble, memory `O(p)` instead of `O(m)`. Write down what
that buys you in practice:

`_______________________________________________________`

(Prompt: if activation memory is what limits `m`, and lowering activation memory lets you
raise `m`, what happens to the bubble?)

---

## 3. Why the steady state alternates

Argue this from the memory bound rather than from the picture.

Suppose stage `i` in the steady state ran two forwards in a row. What happens to its
in-flight activation count?

`_______________________________________________________`

And if it ran two backwards in a row?

`_______________________________________________________`

So one-forward-one-backward is the unique schedule that `____________________`.

Now the warmup. Stage `i` cannot do a backward until the gradient comes back from stage
`i+1`, which cannot happen until `____________________`. So stage `i` must run
`____` forwards first, and that number is exactly the warmup length you wrote in
section 2.

---

## 4. Interleaved 1F1B

Each device holds `v` non-contiguous chunks of the model instead of one contiguous block.
At `p=4, v=2`, device 0 holds layers `____` and `____`.

Effective pipeline depth becomes `____`, so write both the overhead ratio and
the elapsed-time fraction:

```
overhead ratio = ____________________
idle fraction  = ____________________
```

**The cost.** Every chunk boundary is a point-to-point exchange. With `v` chunks per
device, how many exchanges per micro-batch, compared to `v=1`?

```
____
```

**When is it worth it?** Set the bubble saving against the extra communication. Using α
and β from Lab 0, and writing `A` for the activation tensor size:

```
time saved  = ____________________
time added  = ____________________
```

Solve for the `v` at which they balance, for a 25 GB/s InfiniBand link:

```
v* = ____
```

This is writeup question 5.

---

## 5. Load balance

Total pipeline time is set by the **slowest** stage, so an imbalance of 30% in one stage
costs you 30% everywhere.

An even split by layer count is usually uneven in FLOPs. For this repo's model:

**Stage 0** carries the token embedding, the position embedding, and its share of blocks.
Embedding FLOPs per token: `____________________`

**Stage p-1** carries its blocks, the final LayerNorm, the LM head, and the loss. The LM
head is a `(h, V)` matmul, so FLOPs per token: `____________________`

**A middle stage** carries only blocks. FLOPs per token per block:
`____________________`

For the `small` preset (h=384, V=72, 6 layers) at `p=3`, compute the FLOPs for each stage:

```
stage 0: ____
stage 1: ____
stage 2: ____
```

Which is slowest? `____`  By what factor? `____`

Now do the same with a realistic vocabulary, `V=128000`:

```
stage 0: ____
stage 1: ____
stage 2: ____
```

Which is slowest now? `____`  By what factor? `____`

**Write down the general lesson:**

`_______________________________________________________`

(This is why real pipeline configs give the first and last stages fewer transformer
layers than the middle ones, and why `torch.distributed.pipelining` lets you specify the
split point manually rather than dividing by `p`.)

---

## 6. The deadlock, before you meet it

Every rank runs the same code. Stage `i` sends its activation to stage `i+1` and receives
from stage `i-1`.

Write the naive version as pseudocode:

```
send(activation, to=rank+1)
recv(activation, from=rank-1)
```

Trace it for `p=2`. Rank 0 calls `send`. Rank 1 calls `send`. Now what?

`_______________________________________________________`

**Why the message size matters.** A small message fits in the transport's internal buffer,
so `send` copies it and returns immediately, and the deadlock does not happen. A large one
blocks until the peer receives. So this bug appears when you scale up the batch, which is
the worst possible time for it to appear.

`labs/lab0_collectives/tests/test_01_semantics.py::test_symmetric_send_deadlocks`
demonstrates it. If you have not run it yet, do that now.

**Three fixes.** For each, say what it costs:

| Fix | Cost |
|---|---|
| odd/even ordering | `____________________` |
| isend/irecv with explicit wait | `____________________` |
| `batch_isend_irecv` | `____________________` |

Which will you implement? `____`

---

## 7. Predictions

Fill in `predictions.yaml`. The one worth thinking hardest about is
`measured_bubble_p4_m8`: the elapsed-idle theory says
`(4-1)/(8+4-1) = 3/11 ≈ 0.273`; the related overhead ratio is `(4-1)/8 = 0.375`.
Your measurement will usually exceed the ideal elapsed fraction. Predict *how much*
worse and why, before you run it.
