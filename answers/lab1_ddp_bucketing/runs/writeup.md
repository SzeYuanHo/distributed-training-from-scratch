# Run notes

> **Maintenance status.** The original run correctly diagnosed two harness defects.
> The loss tests now compare the global-batch mean, and the Gantt benchmark now records
> each asynchronous request from issue through wait while removing blocking
> communication from serial compute spans. The completed implementation passes the
> current tests. The invalid historical overlap percentages and charts were removed;
> rerun the maintained benchmark to produce a current overlap measurement.
> The retained bucket-sweep PNG predates the unit-label cleanup: its “MB” axis values
> are binary MiB. The maintained plotter labels regenerated output correctly.

## The bucket sweep

1 MiB won, which is what I predicted, and I do not think the prediction deserves
much credit. The curve is nearly flat from 0.5 MiB up — the spread across the whole
sweep is smaller than the run-to-run noise, and a second run would probably pick a
different size. At this model size the model has about 3.1 MiB of gradients, so even the small
buckets are only issuing four collectives a step, and gloo's per-message cost is high
enough that the latency term dominates everywhere. The tail term never gets a chance
to matter. That is a real result about this machine rather than a failed experiment,
but it means the sweep cannot tell me where the optimum would be on hardware where
bandwidth is the binding constraint.

## Throughput

Predicted 45 ms for the single-process step and got 151.94 ms. Two ranks, best
bucket: 255.97 ms against 95 ms predicted. So 0.594x the single-process throughput.

I expected below 1.0 — two ranks on one laptop share a memory bus and add gloo
traffic without adding compute, and the README says as much. What I did not expect
was to be out by 3x on *both* numbers. The absolute step times being wrong together
suggests my estimate of the model's own compute was the problem, not anything to do
with the distribution. I was estimating from FLOPs and this is a tiny model where
Python and allocator overhead dominate.

## The overlap chart

The original overlap chart timed only `finalize()`, so it compared different spans in
the overlapped and serial paths. That made its exposed-communication percentages
invalid. Those values and images are no longer published as measurements.

The maintained benchmark records every asynchronous request from issue through wait
and removes blocking communication from serial compute spans. It therefore measures
the interval the prediction names. I did not fabricate a replacement result without
rerunning that benchmark; `exposed_comm_fraction` remains visibly unmeasured in the
published reconciliation. The ordering property itself is still pinned by tests:
with overlap enabled, collectives are issued as buckets fill rather than after the
last gradient.

## Historical loss-curve harness defect

In the original run, `test_simple_ddp_matches_reference`,
`test_hooked_ddp_matches_reference` and `test_bucketed_ddp_matches_reference` failed
at 2 and 4 ranks because the harness compared different loss quantities.

The test compared each rank's loss against a single-process run on the global batch.
But `_train_with_wrapper` records `float(loss.detach())` — this rank's loss on its own
4 sequences. The reference is the mean over all 8. Those are different quantities.
Their *mean* is the reference, and that checks out to five decimals at the last step
of the 2-rank run:

```
rank 0   4.12394
rank 1   3.79503
mean     3.959485
ref      3.95947
```

So the gradients are being averaged correctly and the models are in lockstep; only
the logged number is per-rank. `test_simple_ddp_step0_gradients_match` passes at both
world sizes, and that one compares gradients directly against the reference before any
optimizer step, which is the check that actually constrains the implementation.

The maintained harness now averages the recorded rank-local curves before comparing
them with the global-batch reference; the wrappers still do not pretend that real DDP
all-reduces a loss automatically. In Lab H hazard 3 the same mismatch appears inside a
file the learner does own, and there the logged value is explicitly all-reduced.

## Not measured

`bucket_sweep.py` does not emit the tensor byte count, the SimpleDDP collective count
or the `no_sync` count, so those stay as derivations in `predictions.yaml` with no
measured column. The collective counts are pinned by tests instead, which is arguably
the better check.
