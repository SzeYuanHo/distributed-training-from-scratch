# Lab 4 — Theory

Little new derivation here. This lab assembles the four models you already built into one
decision procedure, and the assembly is the exercise.

---

## 1. Inventory

Write down, from your own earlier work, the expression for each. Go back and get the real
numbers; do not approximate from memory.

**Ring all-reduce time** for `M` bytes across `N` ranks (Lab 0):

```
T_allreduce(M, N) = ____________________
```

Your fitted values for gloo: `α = ____`   `β = ____`

**Per-rank state memory** for `Ψ` parameters at ZeRO stage `z` across `N` ranks (Lab 1B):

```
mem(Ψ, z, N) = ____________________
```

**Activation memory** per rank (Lab 1B, Korthikanti):

```
act(b, s, h, L, a) = ____________________
```

**Tensor-parallel communication** per step for `L` layers at degree `t` (Lab 2):

```
comm_tp = ____________________
```

**Pipeline bubble overhead relative to ideal time** for `p` stages and `m`
micro-batches (Lab 3):

```
bubble = ____________________
```

---

## 2. Composing them

A configuration is `(dp, tp, pp, m, z)` with `dp · tp · pp = N`.

**Memory per GPU.** Parameters divide by `tp · pp`: tensor parallelism shards within a
layer, pipeline parallelism shards across layers, and the two are independent. ZeRO then
shards further, across the `dp` dimension only.

```
params_per_gpu = ____________________
states_per_gpu = ____________________
```

The activation term needs care. Pipeline parallelism divides activations by `pp` because
each stage holds only its own layers, *and* multiplies by the in-flight micro-batch count,
which is `m` for AFAB and about `pp` for 1F1B:

```
global_sequences = global_batch_tokens / sequence_length
b_micro          = global_sequences / (dp · m)
act_per_gpu = ____________________
total       = ____________________
```

Both divisions must be exact. This planner schedules complete fixed-length sequences:
rounding `b_micro` would change the stated global batch, while hardcoding it to one
sequence would make activation memory independent of `dp` and `m`.

**Communication per step per GPU.** Three contributions on three different groups:

```
DP (gradient all-reduce)  = ____________________
TP (4 per layer, fwd+bwd) = ____________________
PP (activations + grads)  = ____________________
```

Which of the three grows with the parameter count? `____`

Which grows with batch times sequence? `____`

Which is independent of the degree it is parallelised over? `____`

That last answer is the reason for the whole hierarchy in section 3.

**Time per step**, roughly:

```
T_step ≈ (compute / (dp·tp·pp)) · (1 + bubble) + exposed_comm
```

with `exposed_comm` from Lab 1's overlap model. State the assumption this makes about
which communication overlaps and which does not:

`_______________________________________________________`

---

## 3. The hierarchy, derived

PTD-P's guideline is: tensor parallelism within a node, pipeline parallelism across nodes,
data parallelism outermost. Derive it rather than quoting it.

| Axis | Bytes per step | Frequency | Overlappable? | Latency-sensitive? |
|---|---|---|---|---|
| DP | `____` | once per step | `____` | `____` |
| TP | `____` | 4 per layer | `____` | `____` |
| PP | `____` | 2 per micro-batch per boundary | `____` | `____` |

Rank them by how badly each suffers on a slow link:

1. `____`
2. `____`
3. `____`

That ranking is the guideline. Write it in your own words:

`_______________________________________________________`
`_______________________________________________________`

---

## 4. MFU and HFU

**Model FLOPs per token**, forward and backward, for `Ψ` parameters:

```
≈ ____·Ψ
```

Where does that constant come from? A forward matmul is 2 FLOPs per parameter per token,
a multiply and an add. The backward computes both the input gradient and the weight
gradient, so it costs `____` times the forward.

```
MFU = ____________________
HFU = ____________________
```

**With activation recomputation on**, the hardware runs an extra forward pass per layer.
So HFU goes `____` and MFU goes `____`.

Which do you quote when someone asks how efficient the run is?

`_______________________________________________________`

Which tells you whether enabling recomputation was worth it?

`_______________________________________________________`

**Numbers.** A 7B model at 3000 tokens per second per GPU on an A100 at 312 TFLOP/s peak
bf16:

```
model FLOPs/s = ____
MFU           = ____
```

Published large-scale runs report 35 to 55 percent. Where does that sit?

`_______________________________________________________`

---

## 5. The decision exercise

The deliverable. Use the default problem in `starter/planner.py` or state your own:

> Train `Ψ = 7e9` parameters on the "8xA100 (NVLink 3.0)" profile from
> `common/cluster_specs.py`. Sequence length 4096, global batch 1,048,576 tokens
> (256 complete sequences), bf16 AdamW, with full activation recomputation.

Enumerate every `(dp, tp, pp, m, z)` with `dp·tp·pp = 8`, score each, rank them.

Do three by hand first, so you know what the code should say:

**A: `dp=8, tp=1, pp=1, m=8, z=1`**
```
memory/GPU = ____      comm/step = ____      feasible? ____
```

**B: `dp=1, tp=8, pp=1, m=8, z=0`**
```
memory/GPU = ____      comm/step = ____      feasible? ____
```

**C: `dp=2, tp=4, pp=1, m=8, z=1`**
```
memory/GPU = ____      comm/step = ____      feasible? ____
```

Which wins, and by what margin?

`_______________________________________________________`

Now change the cluster to "2 nodes x 8xA100 (InfiniBand)", 16 GPUs, and redo it. Does the
ranking change?

`_______________________________________________________`

If it did not, your cost model is not sensitive to the interconnect and something is wrong
with it. The whole point is that the answer depends on the hardware.
