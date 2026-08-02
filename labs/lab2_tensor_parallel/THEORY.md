# Lab 2 — Theory

Seven derivations. Sections 1 to 3 before you write any code, sections 4 to 7 before the
corresponding build step.

Notation: `X` is `(b·s, h)` activations, `W` is `(h, h')` weights, `t` is the
tensor-parallel degree, `N` is a general world size.

---

## 1. Column parallel

Split `W` along its **output** dimension: `W = [W₁ | W₂]` with each `Wᵢ` of shape
`(h, h'/t)`.

```
XW = ____________________
```

So rank `i` computes `____________________` and holds a shard of the output of shape
`____________________`.

**Forward communication required:** `____`

That answer has a condition attached. Under what circumstances is it zero, and when does
it stop being zero?

`_______________________________________________________`

**The bias.** `y = XW + b` with `b` of shape `(h',)`. Under column sharding, how is `b`
split?

`_______________________________________________________`

**Backward.** Given `∂L/∂Y` sharded the same way as `Y`:

```
∂L/∂Wᵢ = ____________________
∂L/∂X  = ____________________
```

Look carefully at `∂L/∂X`. Each rank computes a *partial* contribution from its own
shard, and the true gradient is the sum over ranks. So the backward pass needs an
`____________________` even though the forward did not.

**That asymmetry is the whole idea of the `f` operator.** Write it down in your own words:

`_______________________________________________________`
`_______________________________________________________`

---

## 2. Row parallel

Split `W` along its **input** dimension: `W = [W₁; W₂]` stacked, each `(h/t, h')`. The
input arrives already sharded as `X = [X₁ | X₂]`.

```
XW = ____________________
```

This is a **sum across ranks**, not a concatenation. So the forward pass needs an
`____________________`.

**Backward.** Given `∂L/∂Y` (full, replicated after the forward reduce):

```
∂L/∂Wᵢ = ____________________
∂L/∂Xᵢ = ____________________
```

Does the backward need a collective? `____`  Why not?

`_______________________________________________________`

**And there is the dual.** Forward all-reduce, backward identity. That is `g`.

Fill in the table you will implement:

| Operator | Forward | Backward |
|---|---|---|
| `f` = `CopyToTensorParallelRegion` | identity | `____________` |
| `g` = `ReduceFromTensorParallelRegion` | `____________` | identity |

**The bias, again, and this one bites.** `b` is `(h',)`, not sharded. If every rank adds
`b` before the all-reduce, what does the output become?

`_______________________________________________________`

So the bias must be added `____________________`.

---

## 3. Why column then row

The MLP is `X → Linear(h, 4h) → GELU → Linear(4h, h)`.

**Column then row.** First layer column-parallel, output sharded `(b·s, 4h/t)`. GELU is
elementwise, so it applies to a shard with no communication. Second layer row-parallel,
consuming the shard directly.

Collectives in the forward pass: `____`

**Row then column.** First layer row-parallel. Its output needs an all-reduce to be
correct. Then GELU. Then the column-parallel layer needs its input to be sharded again,
which means `____________________`.

Collectives in the forward pass: `____`

**Write the general rule** this illustrates, in one sentence:

`_______________________________________________________`

Now check yourself: would the same argument hold if GELU were replaced by a LayerNorm?

`_______________________________________________________`

(This is not a trick question and the answer is the reason sequence parallelism exists.)

---

## 4. Attention

Attention has an extra structure the MLP does not: heads.

**Shard by head.** With `a` heads and degree `t`, rank `i` owns heads
`____________________`, which is `____` heads each.

**Why whole heads.** Suppose you split the hidden dimension without respecting head
boundaries, so a head's `head_dim` channels land on two ranks. To compute that head's
attention scores you need `q · k` over the full `head_dim`, so you would need a
`____________________` *inside* the attention computation, per head, per position.

Write down what that does to the cost model:

`_______________________________________________________`

**The constraint:** `____________________`

Which for the `tiny` preset (4 heads) means the maximum usable `t` is `____`.

**The layout.** `c_attn` projects `(h) → (3h)`, producing Q, K and V concatenated. Under
column sharding by head, rank `i` needs Q, K and V *for its own heads*. Is a naive
`chunk(t, dim=0)` of the `c_attn` weight correct?

`____`  Explain:

`_______________________________________________________`

(Work this out on paper for `h=8`, `a=2`, `t=2`. The weight is `(24, 8)` in torch's
`(out, in)` layout. Which 12 rows does rank 0 need? They are not the first 12.)

**`c_proj`** projects `(h) → (h)` and takes the concatenated head outputs. Sharded
`____________________`-wise, needing a forward `____________________`.

---

## 5. Communication count

Per transformer layer, per training step:

| Where | Forward | Backward |
|---|---|---|
| Attention (`g` after `c_proj`) | `____` | `____` |
| MLP (`g` after `c_proj`) | `____` | `____` |
| Attention (`f` before `c_attn`) | `____` | `____` |
| MLP (`f` before `c_fc`) | `____` | `____` |
| **Total** | `____` | `____` |

So `____` all-reduces per layer per step, each on a tensor of shape `____________________`
and size `____________________` bytes.

For an `n_layer`-layer model:

```
collective buffer bytes counted per step per rank = ____________________
```

Compute it for `small` (6 layers, h=384) at b=8, s=256, fp32:

```
____ bytes = ____ MB
```

Put that in `predictions.yaml` as `comm_bytes_per_step_small_tp2`. This is the
instrumentation's sum of collective tensor payloads, not literal link traffic; a ring's
wire-volume model adds `2(t-1)/t` per all-reduce.

---

## 6. The scaling ceiling

This is the section that makes the lab worth doing.

**Compute per layer per rank.** The MLP is two matmuls of `(b·s, h)` by `(h, 4h)`, and
attention is roughly `4·b·s·h²` plus the quadratic term. Total FLOPs per layer, sharded
`t` ways:

```
FLOPs/rank = ____________________
```

Which scales as `____` in `t`.

**Communication per layer per rank.** From section 5:

```
bytes/rank = ____________________
```

Which scales as `____` in `t`.

**Now put them together.** The compute-to-communication ratio is:

```
ratio(t) = ____________________
```

and it scales as `____` in `t`.

State the conclusion plainly:

`_______________________________________________________`
`_______________________________________________________`

**Numbers.** Using the α–β model from Lab 0, communication time is
`α·2(t-1) + β·2(t-1)/t·M`. Compute time is `FLOPs / peak_flops`.

Two hardware profiles, and `reference/cluster_specs.py` has the sources:

| | 2 × T4, PCIe 3.0 x16 | 8 × A100, NVLink 3.0 |
|---|---|---|
| Peak fp32 FLOP/s per device | 8.1e12 | 19.5e12 |
| Interconnect bandwidth | ~12 GB/s | ~300 GB/s |
| α (approx) | 10 µs | 5 µs |

For a model with `h=4096`, `b·s=8192`, fp16, find the `t` at which communication time
first exceeds compute time:

```
T4 pair:      t* = ____
A100 NVLink:  t* = ____
```

**And now the industry practice makes sense.** Write down the rule this implies:

`_______________________________________________________`

---

## 7. Vocab-parallel cross-entropy

The logit tensor is `(b, s, V)`. For the `tier2` preset at b=8, s=512, V=72 that is small,
but for a real model with V=128000 it is:

```
8 · 512 · 128000 · 4 bytes = ____ GB
```

which is often the single largest activation in the model. Gathering it across ranks so
you can call `F.cross_entropy` would defeat the purpose of sharding the vocabulary in the
first place.

So compute the loss without ever materialising the full logits. Rank `i` holds
`(b, s, V/t)`.

**Step 1: the max.** Ordinary softmax subtracts `max(z)` for stability. Here the true max
is across all shards, so:

```
local_max  = ____________________
global_max = ____________________          (which collective, with which op?)
```

**Step 2: the denominator.**

```
local_sum  = ____________________
global_sum = ____________________          (which collective?)
```

**Step 3: the target logit.** The target index `y` lies in exactly one rank's vocabulary
range. Rank `i` contributes:

```
z_target_i = ____________________ if ____________________ else ____
global      = ____________________          (which collective?)
```

**Step 4: the loss.**

```
loss = ____________________
```

**Step 5: the backward.** The gradient of cross-entropy with respect to the logits is
`softmax(z) - onehot(y)`. Rank `i` holds `V/t` of those entries. Does computing it need
any additional communication?

`____`  Explain:

`_______________________________________________________`

**Count the collectives:** `____` in the forward pass, `____` in the backward. Each on a
tensor of shape `____________________`, which is much smaller than the logits.

That is the whole trick: three tiny collectives instead of one enormous all-gather.

---

## 8. Before you build

Fill in `predictions.yaml`. The interesting one is
`tp_scaling_ceiling_t_for_t4_pair`, which you derived in section 6 and which no
measurement in this repo can confirm, because you do not have enough T4s. That is fine.
The derivation is the deliverable; it is the kind of number you are expected to produce
before requesting a cluster, not after.
