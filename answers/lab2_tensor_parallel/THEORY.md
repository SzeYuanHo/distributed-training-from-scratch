# Lab 2 notes

I used `X: (b·s, h)`, `W: (h, h')`, and TP degree `t`.

## 1. Column split

With `W = [W₁ | ... | Wₜ]`,

```
XW = [XW₁ | ... | XWₜ]
```

Rank `i` computes `XWᵢ` and keeps `(b·s, h'/t)`. There is no forward
communication while the next operation can use that shard. A consumer needing the full
output forces an all-gather.

The bias follows the output split, so each rank owns `(h'/t,)`.

```
∂L/∂Wᵢ = Xᵀ(∂L/∂Yᵢ)
∂L/∂X  = Σᵢ (∂L/∂Yᵢ)Wᵢᵀ
```

Each rank only has one term of `∂L/∂X`. This is `f`: identity forward, all-reduce
backward.

## 2. Row split

With `W = [W₁; ...; Wₜ]` and `X = [X₁ | ... | Xₜ]`,

```
XW = Σᵢ XᵢWᵢ
```

The partial outputs need a forward all-reduce.

```
∂L/∂Wᵢ = Xᵢᵀ(∂L/∂Y)
∂L/∂Xᵢ = (∂L/∂Y)Wᵢᵀ
```

`∂L/∂Y` is already replicated, and each rank only needs its own input-gradient shard.
No backward collective is needed. This is `g`: all-reduce forward, identity backward.

The bias is full-sized. Adding it before the reduction gives `XW + t·b`, so it belongs
after the reduction.

| mapping | forward | backward |
|---|---|---|
| `f` | identity | all-reduce |
| `g` | all-reduce | identity |

## 3. MLP ordering

For column → GELU → row, the `(b·s, 4h/t)` output of the first layer is already what the
second layer needs. GELU is elementwise. The block has one forward collective.

Row → GELU → column needs the row reduction, then an all-gather to restore the replicated
residual-stream output. That makes two.

The useful rule was to join layers whose output and input layouts already agree.
LayerNorm is different: it reduces over the feature dimension, so normalising each
feature shard gives another result.

## 4. Attention heads

Rank `i` owns heads

```
[i·a/t, (i+1)·a/t)
```

or `a/t` whole heads. The constraint is `a % t == 0`; four heads cap the tiny preset at
`t=4`.

Splitting a head's channels would put an all-reduce inside `q·k`, per head and position.
That removes the communication-free attention interior.

The fused QKV rows are ordered as all Q, all K, then all V. A plain chunk of `(3h, h)` is
therefore wrong. For `h=8, a=2, t=2`, rank 0 takes rows `0:4`, `8:12`, and `16:20`. I
used the `(3, n_head, head_dim, h)` view in the loader.

`c_proj` is row-parallel and ends with one forward all-reduce.

## 5. Collective count

Per transformer layer:

| site | forward | backward |
|---|---:|---:|
| attention `g` | 1 | 0 |
| MLP `g` | 1 | 0 |
| attention `f` | 0 | 1 |
| MLP `f` | 0 | 1 |
| total | 2 | 2 |

That is four all-reduces per layer per step. Each buffer is `(b, s, h)`, or
`b·s·h·element_bytes` bytes. The transformer stack alone moves

```
4·n_layer·b·s·h·element_bytes
```

in the counter's buffer-byte convention.

For the six-layer small preset at `b=8`, `s=256`, `h=384`, fp32:

```
per layer = 4·8·256·384·4 = 12,582,912 bytes
six layers = 75,497,472 bytes = 75.5 MB
```

A whole training step also has one embedding reduction, one `lm_head` `f` backward, and
three `(b, s)` loss reductions. The exact call count is therefore
`4·n_layer + 5`, or 29 for the small preset.

## 6. Scaling ceiling

Ignoring the quadratic attention term, per-layer work on one rank is

```
F(t) = 24·b·s·h²/t
```

so compute falls as `1/t`. The layer's four activation-buffer reductions account for

```
C = 4·b·s·h·element_bytes
```

which stays constant in the counter as `t` grows. The simple FLOP/byte ratio is

```
F/C = 6h/(t·element_bytes)
```

and also falls as `1/t`.

For the timing calculation I used two forward all-reduces:

```
Tcompute = 24·b·s·h² / (t·peak_flops)
Tring    = 2(t-1)α + 2(t-1)M/(t·bandwidth)
Tcomm    = 2·Tring
M        = b·s·h·2 bytes
```

The workload is fp16, so I used the matching tensor-core peaks rather than the fp32
figures:

| link | peak fp16 | bandwidth | α |
|---|---:|---:|---:|
| T4 / PCIe 3.0 | 65 TFLOP/s | 12 GB/s | 10 µs |
| A100 / NVLink 3.0 | 312 TFLOP/s | 300 GB/s | 5 µs |

At `h=4096` and `b·s=8192`, communication first exceeds compute at `t=4` on PCIe and
`t=11` on NVLink. A power-of-two sweep shows the crossings between 2–4 and 8–16.

So TP belongs inside the fastest local fabric. I would use data or pipeline parallelism
across the slower node boundary.

## 7. Vocab-parallel loss

The full fp32 logit buffers are:

```
tier2 preset:  8·512·72·4      = 1,179,648 bytes
real vocab:    8·512·128000·4  = 2,097,152,000 bytes
```

For a local `(b, s, V/t)` shard:

```
local_max  = local_logits.max(-1).values
global_max = all_reduce(local_max, MAX)

local_sum  = exp(local_logits - global_max[..., None]).sum(-1)
global_sum = all_reduce(local_sum, SUM)

zᵢ = local_logits[..., y-startᵢ]  if startᵢ ≤ y < endᵢ else 0
z_target = all_reduce(zᵢ, SUM)

loss = mean(log(global_sum) + global_max - z_target)
```

There are three forward collectives, all on `(b, s)`. The local gradient is its slice of
`softmax(z) - onehot(y)`, so the backward adds no communication.
