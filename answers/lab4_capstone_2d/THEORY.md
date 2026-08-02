# Lab 4 notes

## Models carried over

For a ring:

```
T_allreduce(M, N) = 2(N - 1)alpha + 2(N - 1)beta M / N
```

Before rerunning the fit I used the localhost profile values:
`alpha = 3e-5 s` and `beta = 5e-10 s/byte`.

For bf16 AdamW, the persistent state for `Psi` parameters is:

| stage | bytes per rank |
|---:|---|
| 0 | `16 Psi` |
| 1 | `4 Psi + 12 Psi/N` |
| 2 | `2 Psi + 14 Psi/N` |
| 3 | `16 Psi/N` |

The no-recompute activation estimate is

```
L s b h (34 + 5 a s / h) bytes
```

for bf16. Tensor parallelism divides that estimate by `t`. The default 7B
problem uses full recomputation, whose estimate is `2 L s b h` bytes.

For `L` layers, the tensor-parallel traffic estimate I used is

```
4 L * 2(t - 1)/t * (b s h * dtype_bytes)
```

and the pipeline bubble overhead relative to ideal time is `(p - 1)/m`.
Equivalently, the fraction of elapsed pipeline time spent idle is
`(p - 1)/(m + p - 1)`.

## Composing a configuration

For `(dp, tp, pp, m, z)`:

```
params_per_gpu = Psi / (tp pp)
states_per_gpu = ledger(params_per_gpu, z, world_size=dp)
```

The activation term uses `L/pp` layers and an in-flight multiplier of `m` for
AFAB or `min(pp, m)` for 1F1B:

```
global_sequences = global_batch_tokens / s
b_micro          = global_sequences / (dp m)
act_per_gpu = activation_estimate(b_micro, s, h, L/pp, a, tp) * inflight
total       = states_per_gpu + act_per_gpu
```

Both divisions are exact contracts. The model's global token batch must contain
whole fixed-length sequences, and that sequence count must divide across `dp m`.
TP and PP ranks cooperate on each local micro-batch, so neither divides it
again. A non-integral candidate is infeasible; rounding would silently change
the optimizer-step batch and its activation estimate.

The three traffic terms are:

```
DP = 2(dp - 1)/dp * local_gradient_bytes
TP = 4(L/pp) * 2(tp - 1)/tp * local_activation_bytes
PP = 2 min(2, pp - 1) * local_activation_bytes
```

DP grows with parameter count. TP and PP grow with batch times sequence. TP
volume stays roughly flat as `tp` grows; only the ring factor changes.

These are per-GPU volumes, so an interior pipeline stage counts its two
neighbouring boundaries rather than every boundary in the pipeline. The dtype
is also a feasibility constraint: the default bf16 problem is not executable
on the T4 profile, whose tensor cores support fp16 but not bf16.

For step time I charged 30% of DP communication, all TP communication and half
of PP communication. It is deliberately rough: DP mostly overlaps backward,
TP reductions sit on the layer critical path, and PP transfers overlap only
partly at stage boundaries. The latency count still includes every ring round
and every micro-batch.

Full recomputation multiplies compute by `4/3`; I used `13/12` for selective
recomputation as a rough middle ground.

## Link hierarchy

| axis | traffic | frequency | overlap | latency |
|---|---|---|---|---|
| DP | gradient-sized | once per step | mostly | low sensitivity |
| TP | activation-sized | four per layer | little | high sensitivity |
| PP | activation-sized | twice per micro-batch per boundary | partial | medium |

Slow links hurt TP first, then PP, then DP. I therefore keep TP inside the
fastest domain, put pipeline boundaries across the next tier when needed, and
leave DP as the outer axis.

## MFU

Useful training work is about `6 Psi` FLOPs per token: the forward is `2 Psi`
and backward is twice the forward.

```
MFU = useful model FLOPs / (elapsed time * peak FLOP/s)
HFU = executed hardware FLOPs / (elapsed time * peak FLOP/s)
```

Full recomputation adds another forward, so HFU rises by about one third while
MFU is unchanged. I would quote MFU for training efficiency and compare HFU
against MFU when deciding whether the memory saving justified recomputation.

For 7B parameters at 3000 tokens/s:

```
model FLOPs/s = 6 * 7e9 * 3000 = 1.26e14
MFU           = 1.26e14 / 3.12e14 = 0.404
```

That sits inside the usual 35--55% range.

## Hand check of three candidates

The default problem has 256 sequences per optimizer step and uses full
recomputation. For a common `m=8`, the derived local micro-batches contain 4,
32, and 16 sequences respectively:

| config | memory/GPU | traffic/step | estimated step | verdict |
|---|---:|---:|---:|---|
| `dp=8,tp=1,pp=1,m=8,z=1` | 39.86 GiB | 22.82 GiB | 52.304 s | fits |
| `dp=1,tp=8,pp=1,m=8,z=0` | 45.04 GiB | 1.75 TiB | 58.765 s | fits, TP-heavy |
| `dp=2,tp=4,pp=1,m=8,z=1` | 32.30 GiB | 771.26 GiB | 55.062 s | fits |

The pure-DP candidate wins this cost model because its gradient traffic is
charged as mostly overlapped while TP traffic is on the critical path. The
seconds are model outputs from published peaks and an assumed 45% achievable
fraction, not measurements. On sixteen A100s the top row moves to `dp=16,tp=1`;
the profile still cannot represent NVLink and InfiniBand separately, so that
result should not be read as a topology validation.
