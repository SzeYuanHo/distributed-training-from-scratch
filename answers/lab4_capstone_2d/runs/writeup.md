# Run notes

> **Maintenance status.** The subgroup helper defect documented below was fixed by
> broadcasting from group-local rank zero's global rank. Cross-rank tolerance is now
> checked elementwise and the control tensor uses a backend-compatible device. Both
> composition tests pass in the current completed-answer suite. The original diagnosis
> is retained as provenance.

## What was actually measured

One number: 2,586.23 tokens/s, single process, single device, CPU.

That is it. Everything else in this lab is the planner's output, and the planner is
arithmetic over published datasheet figures. The FLOP count uses `6N` per token as
predicted, but the `gloo on localhost` profile has no peak FLOP/s to divide by, so
there is no measured MFU here at all. My 12% figure for a T4 is still a guess and
this run did nothing to confirm it; that estimate concerns a separate small
fp16-compatible workload, not the default bf16 7B planner problem.

Worth being blunt about that in a lab whose deliverable is a scaling report: the
table below is a model, and the only thing separating it from a plausible-sounding
guess is that the model is written down and can be argued with.

## The table

The corrected default is 1,048,576 tokens: exactly 256 sequences of length 4096,
with full activation recomputation. On the 8xA100 profile the first feasible row is
`dp=8 tp=1 pp=1 m=1 zero=1`. Its local micro-batch contains 32 sequences. The model
reports 67.86 GiB per GPU, 22.82 GiB of DP communication, and 52.304 s.

That row has almost no margin below the 68 GiB planning budget. Rows with larger `m`
derive smaller local micro-batches and therefore lower activation memory: at the same
DP/TP/PP layout and ZeRO stage, `m=4` reports 43.86 GiB and `m=32` reports 36.86 GiB.
Their times tie in this deliberately coarse model, which has no accumulation-efficiency
term. A production recommendation should prefer real headroom rather than treating the
first tied row as uniquely best.

My corrected hand comparison uses `dp=2 tp=4 pp=1 m=8 zero=1`. Its 16-sequence local
micro-batch fits at 32.30 GiB, but 771.26 GiB of modelled traffic leaves it at 55.062 s.
The planner prefers pure DP because it charges most DP traffic as overlapped and all TP
traffic on the critical path.

The `6N` compute term uses the full 1,048,576-token optimizer-step batch and full
recomputation adds its stated compute factor. Internal consistency does not validate
the resulting seconds. The achievable-fraction assumption, overlap factors, topology,
and omitted runtime effects still need measurement, so the seconds column remains a
model output rather than a timing claim.

At 16 devices the top row becomes `dp=16 tp=1 pp=1`. This is not a clean interconnect
comparison because both device count and link speed changed, and the profile has only
one bandwidth for all axes. On two T4s, the default 7B problem is both too large and
uses unsupported bf16; a T4 experiment needs a separate fp16-compatible problem.

## Corrected micro-batch contract

The planner now derives
`micro_batch_sequences = global_batch_tokens / (seq_len * dp * m)` and rejects any
remainder. TP and PP do not divide it again. The regression tests vary `m` and assert
the corresponding activation-memory ratio, so the former one-sequence shortcut cannot
return unnoticed.

## Historical subgroup-helper defect

In the original run, `test_2d_composition_matches_reference` and
`test_collective_counts_split_across_the_two_groups` both failed in the same place in
`common/`:

```
common/testing.py:167, in assert_close_across_ranks
    dist.broadcast(reference, src=0, group=group)
ValueError: Global rank 0 is not part of group <ProcessGroup ...>
```

`src` is a global rank. The helper hardcodes 0, which is right for the default group
and wrong for any subgroup that does not contain rank 0. At `dp=2, tp=2` the TP
groups are `[0,1]` and `[2,3]`, so the call works on the first and raises on the
second. The test passes `group=layout.tp_group`, so it hits it every time.

The fix belonged in `common/testing.py`: take the group's first member rather than
assuming 0, which `dist.get_global_rank(group, 0)` gives you. The maintained helper now
does that and also puts its control tensor on a backend-compatible device.

The completed-answer suite now passes both composition tests. The independently checked
mesh coordinates and group membership still confirm that TP is innermost with groups
`[0,1]` and `[2,3]`.
