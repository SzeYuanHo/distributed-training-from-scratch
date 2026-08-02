# Lab 1B — Theory

Do this before reading `common/memory.py`. It is an honour-system rule and it is the
whole lab.

`Ψ` = number of parameters. `N` = world size.

---

## 1. The ledger, from first principles

Mixed-precision training with AdamW. Walk through what has to exist in memory at the
moment the optimizer steps, and why.

| What | Dtype | Bytes per parameter | Why it has to exist |
|---|---|---|---|
| Parameters used in forward/backward | bf16 | `____` | `____________________` |
| Gradients | bf16 | `____` | `____________________` |
| Master parameters | fp32 | `____` | `____________________` |
| Adam first moment `m` | fp32 | `____` | `____________________` |
| Adam second moment `v` | fp32 | `____` | `____________________` |
| **Total** | | `____` | |

The row that needs the most thought is the master copy. Why can you not just keep the
bf16 parameters and update them in place?

`_______________________________________________________`

(Prompt if you need it: bf16 has 8 mantissa bits. A typical update is `lr * m/sqrt(v)`,
which for `lr=1e-4` and a parameter of magnitude 1 is around `1e-4`. What happens when you
add `1e-4` to `1.0` in bf16?)

Now the same table for **fp32 training with no mixed precision**:

| What | Bytes per parameter |
|---|---|
| Parameters | `____` |
| Gradients | `____` |
| Master copy | `____` |
| `m`, `v` | `____` |
| **Total** | `____` |

Surprising, is it not? Write down what that means for the claim "mixed precision halves
your memory":

`_______________________________________________________`

---

## 2. The ZeRO stages

Write `16Ψ` as `2Ψ + 2Ψ + KΨ`. What is `K`, and which rows does it cover?

`K = ____`, covering `____________________`

Now shard progressively. For each stage write the per-rank bytes and say which row moved:

```
ZeRO-0  (plain DDP)   ____________________
ZeRO-1                ____________________
ZeRO-2                ____________________
ZeRO-3                ____________________
```

**Numbers.** Ψ = 1e9, N = 8. Fill in the per-rank memory in GB:

| Stage | Per-rank GB |
|---|---|
| 0 | `____` |
| 1 | `____` |
| 2 | `____` |
| 3 | `____` |

At which stage does a 1B-parameter model first fit on an 8 GB card, ignoring activations?

`____`

Now include a realistic activation term and answer again:

`____`

---

## 3. Communication volume, which is the part people get wrong

**Plain DDP.** One all-reduce over all gradients per step. Using the ring cost from
Lab 0, bytes moved per rank:

```
____________________
```

**ZeRO-1.** Reduce-scatter the gradients, step on your shard, all-gather the updated
parameters. Bytes per rank:

```
reduce-scatter:  ____________________
all-gather:      ____________________
total:           ____________________
```

Compare the two totals:

`_______________________________________________________`

State the identity plainly, because it is the thing to remember:

`_______________________________________________________`

**ZeRO-3.** Parameters are sharded, so every forward pass has to all-gather the
parameters of each layer before using it, and every backward pass has to do it again.
Extra bytes per rank per step:

```
____________________
```

Under what conditions is that worth paying?

`_______________________________________________________`

---

## 4. Activation memory

The ledger above ignores activations entirely, and at large batch times sequence length
they are the largest term. From Korthikanti et al. §4, per transformer layer with no
recomputation, in 16-bit:

```
s·b·h · (34 + 5·a·s/h)  bytes
```

with `s` = sequence length, `b` = batch, `h` = hidden, `a` = heads.

**Which term is quadratic in `s`, and what is it?**

`_______________________________________________________`

**Compute it** for the `small` preset (h=384, a=8, 6 layers) at b=8, s=256:

```
per layer  = ____ bytes
all layers = ____ bytes = ____ MB
```

Compare to the states total for `small` (10.8 M parameters, so `16 × 10.8e6` = `____` MB).

Which dominates? `____`

Now double the sequence length to 512 and recompute:

```
activations = ____ MB
```

By what factor did it grow? `____`  Explain:

`_______________________________________________________`

**Selective recomputation** drops the quadratic term. New estimate at s=512:

```
____ MB
```

What does it cost? `_______________________________________________________`

---

## 5. Will it fit?

Assume an 8 GiB GPU and reserve 15% as an explicit planning allowance for allocator
overhead, workspaces and fragmentation. Actual usable memory varies by device, driver,
other processes and workload.

**Question A.** Plain AdamW, mixed precision, batch 8, sequence 512. What is the largest
Ψ that fits?

```
states budget  = ____ bytes
Ψ_max          = ____ parameters
```

**Question B.** Same, but ZeRO-1 at N=4. Note that N=4 on one GPU is not a thing, so this
is a question about a 4-GPU machine.

```
Ψ_max = ____ parameters
```

**Question C.** You want to train a 1.5 B parameter model. What is the minimum N with
ZeRO-1? With ZeRO-3?

```
ZeRO-1: N ≥ ____
ZeRO-3: N ≥ ____
```

Put these in `predictions.yaml`. `bench/memory_curve.py` measures what it can and the
reconciliation tells you how close you were.

---

## 6. Before you implement

One design question that the starter deliberately leaves open. ZeRO-1 shards
"parameters" across ranks, but a model's parameters are a list of tensors of wildly
different sizes: the embedding is huge and every LayerNorm bias is tiny.

**Option A: shard by parameter.** Assign whole tensors to ranks, greedily balancing total
size. Simple. Imperfect balance, and with one dominant tensor the imbalance can be large.

**Option B: shard by flattened byte range.** Concatenate everything into one virtual flat
buffer and give each rank a contiguous `1/N` slice. Perfect balance. Now a single
parameter can straddle two ranks, and the optimizer step for it happens in two places.

Which will you implement, and what did you give up?

`_______________________________________________________`

(Real ZeRO does B. `ZeroRedundancyOptimizer` does A. Both ship.)
