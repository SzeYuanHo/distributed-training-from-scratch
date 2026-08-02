# Scaling report

## Scope

I used the corrected default 7B problem: 32 layers, hidden size 4096,
32 heads, sequence length 4096, and a global optimizer-step batch of
1,048,576 tokens, or 256 complete sequences. The planner explicitly uses
full activation recomputation. The main target was the eight-A100 NVLink
profile. I also evaluated the two-T4 and sixteen-A100 profiles to expose
which constraints move.

The [configuration table](config_table.md) is entirely modelled from the
profiles in `common/cluster_specs.py`. I did not run a 7B model on any of
these clusters.

## Batch contract

For every candidate, the local scheduled batch is

```
sequences_per_micro_batch = 256 / (dp * m)
tokens_per_micro_batch    = sequences_per_micro_batch * 4096
```

The division must be exact. TP and PP ranks cooperate on that same data,
so they do not divide the batch again. The table prints both derived
columns; a candidate that cannot divide the 256 sequences is marked
infeasible instead of being rounded.

## Eight A100s

The first feasible row is `dp=8, tp=1, pp=1, m=1, zero=1`. Its local
micro-batch is 32 sequences, or 131,072 tokens. With full recomputation,
the planner estimates 67.86 GiB per GPU, 22.82 GiB of DP traffic, no TP or
PP traffic, and 52.304 s per optimizer step.

That row is only just below the 68 GiB budget after the 15% reserve. The
ZeRO-2 and ZeRO-3 variants reduce modelled memory to 56.45 and 45.04 GiB
without changing this cost model's time. Increasing `m` also now changes
activation memory as it should: for the ZeRO-1 row, `m=4` reduces the
estimate to 43.86 GiB and `m=32` to 36.86 GiB. Their estimated times tie
because PP is one, the total DP volume is unchanged, and the model has no
gradient-accumulation efficiency term. The leading star therefore should
not be read as evidence that `m=1` is operationally safer.

My hand comparison used `dp=2, tp=4, pp=1, m=8, zero=1`. The corrected
micro-batch contains 16 sequences. It fits at 32.30 GiB but carries
771.26 GiB per step, nearly all on the TP axis, and is estimated at
55.062 s. Under the stated overlap assumptions, pure DP wins because its
once-per-step gradient traffic is mostly overlapped while TP collectives
remain on the layer critical path.

## Other profiles

The default problem is not executable on the two-T4 profile for two
independent reasons. Its dtype is bf16, which T4 tensor cores do not
support, and every displayed 7B row also exceeds the 13.60 GiB memory
budget. The table states both reasons. A T4 experiment must use an fp16
model sized for those devices; the regression suite uses such a separate
125M-parameter sanity problem.

With sixteen A100s, the first row is
`dp=16, tp=1, pp=1, m=1, zero=1`: 16 sequences per micro-batch,
46.97 GiB per GPU, 24.45 GiB of DP traffic, and a modelled 26.455 s step.
This is not a clean experiment on interconnect sensitivity because the
profile changes both device count and link speed. It also supplies one
bandwidth number for every axis, so the planner cannot distinguish
intra-node NVLink traffic from cross-node InfiniBand.

## What actually ran

The retained measurement is the supplied small preset on one CPU:
2,586.23 tokens/s. The `6 * parameters` FLOPs-per-token coefficient was
recorded, but there is no CPU peak in the profile, so MFU was not
computed. I did not extrapolate that throughput to A100s or T4s, and none
of the table's seconds are measurements.

The small single-device measurement did not enable recomputation, so MFU
and HFU would count the same work if a defensible peak were available.
The planner's default 7B problem is different: it explicitly enables full
recomputation for its memory estimates and adds the corresponding compute
factor to its modelled step time.

## Limits and next run

The corrected planner now ties activation memory to the global batch,
data-parallel degree, and accumulation count. Its remaining approximations
include the analytic activation formula, fixed communication-overlap
fractions, omitted kernel and accumulation efficiency, allocator peaks,
ZeRO parameter communication, and a topology profile that has only one
link bandwidth.

Before reserving a cluster, I would choose a row with more than marginal
memory headroom, add topology-specific bandwidths, and measure a few
optimizer steps. I would also compare the same global batch across
several legal `m` values: the arithmetic now captures their activation
memory, but only a run can expose their kernel-efficiency and scheduling
costs.
