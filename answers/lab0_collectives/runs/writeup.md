# Run notes

> **Maintenance status.** The raw timings below are preserved from the original run.
> The original analysis fitted a complete all-reduce intercept and then applied the
> topology factors a second time. The derived measurements, reconciliation and plot
> were regenerated with link-level alpha and beta after normalising those factors.
> A later audit found that the plotting path reused the ring fit for the
> recursive-doubling prediction and omitted its promised crossover marker. That stale
> PNG was removed; the maintained plotter now fits both algorithms separately and marks
> every finite fitted crossover.

Full sweep at 2, 4 and 8 ranks, 1 KiB to 16 MiB, gloo on localhost.

The byte counts came out exactly right and every timing prediction was badly
wrong. That split is the useful part: I can count messages, and I had no idea what
one costs.

## The counts

1.5 MiB per rank for a 1 MiB buffer at N=4, measured, against 1,572,864 predicted.
Exact. Same for the send counts and the broadcast counts, which are derivations
rather than measurements — the sweep does not record them, so they sit in the
reconciliation without recorded counterparts.

## The timings

|              | predicted | measured  |
|--------------|-----------|-----------|
| alpha        | 30 us     | 222 us    |
| beta         | 0.5 ns/B  | 1.73 ns/B |
| 1 MiB, N=4   | 0.97 ms   | 3.21 ms   |
| 16 MiB, N=4  | 12.8 ms   | 94.6 ms   |

Alpha is out by a factor of 7.4. I guessed 30 microseconds by adding up a syscall, a
trip through loopback and a scheduler wakeup, which would be about right for the
network part. Almost none of it is the network. It is Python: a `send` here goes
through the wrapper in `common/collectives`, a counter record, a pybind boundary and
a fresh tensor allocation, and that happens 2(N-1) times per all-reduce. The α–β
model is fine; I was fitting it to the wrong machine. My mental model had a NIC in
it and there isn't one.

Beta at 1.73 ns/byte is 578 MB/s effective, against the 2 GB/s I assumed. Loopback
should manage better than that, and I think the gap is the same story — per-chunk
Python overhead that does not vanish as the messages get larger, because the chunk
count is fixed by N rather than by size.

Crossover moved from the predicted 133 kB to 105 kB. The fitted ring and
recursive-doubling bandwidth terms put the crossover closer than the original
estimate, despite the larger measured latency.

## One thing I should not have called latency

The original `fit_alpha_beta` call fitted a line to the elapsed time of a
*complete all-reduce*, so its intercept was per-operation, not per-message. The
current benchmark divides out the collective rounds and fits bytes per round
before passing the result to `predict_allreduce`, so the reported 222 us is now
the link-level model parameter used consistently by the prediction.

## Left open

The sweep never compares the handwritten ring against `dist.all_reduce`, so
`ring_vs_torch_slowdown_factor` is still the 5x I guessed. Given how much of alpha
turned out to be interpreter overhead, and that torch's version does none of it, I
would now guess considerably worse than 5x.

The raw timings are lumpy between 256 KiB and 2 MiB — a few adjacent sizes go
backwards. Median of 12 with 3 warmup, so it is not a single outlier. Probably a
protocol or buffer-size threshold, which is item 2 on the list in THEORY section 1 of
things the straight line cannot capture.
