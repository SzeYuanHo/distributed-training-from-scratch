# Run notes

> **Maintenance status.** The invalid `batch_size=8, n_micro=16` test fixture described
> below now supplies a divisible batch. The benchmark also distinguishes the overhead
> ratio `(p-1)/m` from the elapsed idle fraction `(p-1)/(m+p-1)`; the latter is what
> `1 - compute_occupancy` measures. Recorded timing/RSS values remain provenance from the
> original run. The two Gantt PNGs with the superseded `0.375` theory label were removed;
> the maintained script regenerates them with the `3/11 = 0.273` elapsed fraction.

## The bubble falls, as advertised

| p | m=1 | m=16 |
|---|-----|------|
| 2 | 0.515 | 0.170 |
| 4 | 0.766 | 0.265 |

Monotone in both directions. The Gantt run at p=4, m=8 gave 0.369 for 1F1B. The
original analysis compared that number with the 0.375 overhead ratio. The elapsed
idle fraction on the same denominator as the measurement is `3/11 = 0.273`, so the
observed gap is about 0.096. I had guessed 0.52, assuming gloo latency would swamp it.

AFAB at the same shape gave 0.246, i.e. *lower* than 1F1B. The two schedules have the
same bubble by construction, so this is measurement noise plus the fact that AFAB's
idle time is bunched at the two ends where my spans do not cover it, rather than
distributed through the steady state. It is not evidence that AFAB is better and I
would not put that number in front of anyone without the caveat.

## The two bubble quantities have different denominators

`(p-1)/m` counts extra slots against ideal useful work, so it is an overhead ratio
and can exceed one. What the timer reports is `1 - compute_occupancy`, a fraction
of elapsed wall time that cannot exceed one. The comparable theoretical fraction
is `(p-1)/(m+p-1)`.

It shows up plainly at p=4, m=1: the overhead ratio is 3.0, while the elapsed
fraction is 0.75 against 0.766 measured. At m=8, the elapsed fraction is 0.273
against 0.369. The maintained sweep now labels and plots the elapsed fraction;
the overhead ratio remains useful when its denominator is stated.

## The memory claim did not show up

AFAB 267,350,016 bytes, 1F1B 266,145,792. Essentially identical, and the schedule
analysis says 1F1B should hold about `p` micro-batches against AFAB's `m`.

Two reasons, both about the measurement rather than the schedules. On CPU there is no
`max_memory_allocated`, so this is whole-process RSS: interpreter, torch, dataset,
allocator, all constant between the two runs. And each worker builds the full model
before keeping its own stage, so a whole spare model sits in that number too. The
activation queue I am trying to observe is a few megabytes inside 267.

`test_00_math.py` checks the bound on the schedule itself with no tensors involved,
and that passes: max in-flight is 4 at p=4 for every m in {8, 16, 32}, against AFAB's
m. That is the claim, verified where it can be verified. The RSS numbers are not
evidence either way and I should have predicted that before running it.

## Production API comparison

The hand-built AFAB schedule maps to `torch.distributed.pipelining.ScheduleGPipe`; the
warmup, alternating steady state and cooldown map to `Schedule1F1B`. The production API
also owns stage metadata, static tensor shapes and runtime communication that this lab
implements explicitly to expose the mechanics.

The maintained suite checks that those two production schedule classes are available on
the supported Torch line. I did not record a numerical timing comparison against this
version-dependent API, so the published results make no such claim.

## Historical fixture defect

The original `test_peak_activation_memory_is_lower_under_1f1b[4-16]` failed before
either schedule executed. It constructed
`PipeConfig(n_layer=4, n_micro=16, batch_size=8)`, so `micro_batch_size` correctly
rejected a batch that could not be divided into sixteen non-empty micro-batches.

The maintained fixture supplies a divisible batch. My original p=4, m=16 sweep also
used batch 16 and ran without that defect.

## Provenance note

In the original run, `measured_bubble_p4_m8` in `measurements.yaml` was written by the
sweep and then overwritten by the later Gantt run's 1F1B value. Same shape, different
run. I kept the recorded value rather than pretending they were one measurement. I did
not retain the old schedule images because their theory annotation used the superseded
overhead ratio; regenerate them from the maintained source before publishing a visual.
