# Lab 1B — Theory answers

`Ψ` is the parameter count and `N` is the world size.

## 1. The ledger

For mixed-precision AdamW:

| What | Dtype | Bytes per parameter | Why |
|---|---:|---:|---|
| Parameters | bf16/fp16 | 2 | Used by forward and backward |
| Gradients | bf16/fp16 | 2 | Held until the optimiser step |
| Master parameters | fp32 | 4 | Accumulate updates too small for bf16 |
| Adam first moment | fp32 | 4 | Running first moment |
| Adam second moment | fp32 | 4 | Running second moment |
| **Total** | | **16** | |

At magnitude one, a `1e-4` update is below bf16's spacing, so updating the bf16
copy directly can round the update away. The fp32 master copy preserves it.

For fp32 training:

| What | Bytes per parameter |
|---|---:|
| Parameters | 4 |
| Gradients | 4 |
| Separate master copy | 0 |
| Adam `m`, `v` | 8 |
| **Total** | **16** |

Mixed precision therefore does not halve persistent AdamW state memory. It mainly
halves activation and communication bytes and enables tensor-core throughput.

## 2. ZeRO stages

In `2Ψ + 2Ψ + KΨ`, `K = 12`: four bytes for the fp32 master parameters and
eight for Adam's two moments.

```text
ZeRO-0: 2Ψ + 2Ψ + 12Ψ
ZeRO-1: 2Ψ + 2Ψ + 12Ψ/N
ZeRO-2: 2Ψ + (2Ψ + 12Ψ)/N
ZeRO-3: (2Ψ + 2Ψ + 12Ψ)/N
```

For `Ψ = 1e9`, `N = 8`:

| Stage | Per-rank GB |
|---|---:|
| 0 | 16.00 |
| 1 | 5.50 |
| 2 | 3.75 |
| 3 | 2.00 |

Ignoring activations, stage 1 is the first that fits an 8 GB card. Once realistic
activations are included the answer depends on batch, sequence length and model
shape; a long context can require stage 2 or 3 despite the state table fitting.

## 3. Communication volume

For gradient buffer size `P`, plain ring DDP moves

```text
2(N - 1)/N · P
```

bytes per rank.

ZeRO-1 moves:

```text
reduce-scatter: (N - 1)/N · P
all-gather:     (N - 1)/N · P
total:          2(N - 1)/N · P
```

An all-reduce is exactly a reduce-scatter followed by an all-gather, so ZeRO-1
changes ownership and scheduling but not total bandwidth volume.

ZeRO-3 must additionally all-gather parameters for forward and again for backward,
roughly

```text
2(N - 1)/N · Ψ_parameter_bytes
```

extra bytes per rank per step, before finer-grained prefetch and reuse effects.
It is worthwhile when replicated parameters do not fit and the interconnect can
hide these gathers behind layer compute.

## 4. Activation memory

The quadratic term is

```text
s·b·h · 5a·s/h = 5abs²
```

and represents attention score/probability storage.

For `h = 384`, `a = 8`, six layers, `b = 8`, `s = 256`:

```text
per layer  = 47,710,208 bytes
all layers = 286,261,248 bytes = 286.3 MB
```

The state total for `10.8 M` parameters is `172,800,000` bytes, so activations
dominate.

At `s = 512`:

```text
activations = 824,180,736 bytes = 824.2 MB
```

This is `2.88x` the `s = 256` result: doubling sequence doubles the linear term
and quadruples the attention term.

Selective recomputation removes the quadratic term:

```text
512·8·384·34·6 = 320,864,256 bytes = 320.9 MB
```

It costs extra forward computation during backward, but less than fully
recomputing every saved activation.

## 5. Capacity planning

Using an 8 GiB card with 15% headroom gives
`7,301,444,403` usable bytes. If I reserve the worked `824,180,736` activation
estimate, the state budget is:

```text
states budget = 6,477,263,667 bytes
```

For plain mixed-precision AdamW:

```text
Ψ_max = floor(6,477,263,667 / 16) = 404,828,979 parameters
```

For ZeRO-1 at `N = 4`, the state cost is seven bytes per parameter:

```text
Ψ_max = floor(6,477,263,667 / 7) = 925,323,381 parameters
```

For a 1.5 B parameter model, ignoring activations to isolate the stage formula:

```text
ZeRO-1: 6e9 + 18e9/N <= 7,301,444,403, so N >= 14
ZeRO-3: 24e9/N <= 7,301,444,403, so N >= 4
```

## 6. Sharding choice

I implement whole-parameter, greedy largest-first sharding. It is simple and keeps
each AdamW update in one place, but a tensor larger than the ideal `1/N` share
sets an unavoidable imbalance. Flattened byte-range sharding would balance
perfectly at the cost of splitting a parameter and its optimiser update across
ranks.

