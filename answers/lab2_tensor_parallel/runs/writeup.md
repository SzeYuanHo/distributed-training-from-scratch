# Run notes

> **Maintenance status.** The rank-local finite-difference gradcheck described below was
> removed because it cannot validate a distributed conjugate mapping. The distributed
> conjugacy and end-to-end equivalence tests remain and pass. A local single-device
> hardware profile that caused the right-panel divide-by-zero was also removed; unknown
> device peaks are no longer guessed. The plotting code now skips profiles without a
> positive compute peak and suppresses misleading log-axis minor labels. The stale
> pre-fix PNG was removed; rerun the maintained benchmark to regenerate it.

## The flat line

TP=2 and TP=4 both recorded 4,721,664 collective payload bytes per rank per step.
Identical, not approximately. Four all-reduces per layer per step, as predicted.

That is the result the whole lab is built to produce and it is worth staring at for a
second: doubling the tensor-parallel degree halved the per-rank compute and did
nothing to the shape of the tensor being reduced: `(b, s, h)` does not depend on `t`.
The counter sums those logical tensor sizes. Literal ring link traffic also carries
the bounded `2(t-1)/t` factor, so it is not exactly constant. Per-rank compute still
falls as 1/t while communication does not, creating a degree past which additional TP
costs more than it saves. This is why TP favours a fast interconnect domain.

## The benchmark and prediction use different workload keys

The six-layer `small` prediction is intentionally not compared with the benchmark's
smaller default workload.

I computed 75,497,472 bytes for what I was picturing: the `small` preset, 6 layers,
batch 8, sequence 256. `tp_scaling.py` defaults to 4 layers, hidden 256, batch 4,
sequence 64. The maintained benchmark records those measurements under keys containing
that actual shape, so reconciliation no longer presents the two quantities as a miss.

Its own analytic line, `4·L·b·s·h·4`, gives 4,194,304 for the shape it actually ran,
against 4,721,664 measured. The 527,360-byte remainder is the embedding all-reduce
and the three in the loss, which the per-layer formula does not count. That gap I
believe.

The original run taught the useful data-contract lesson: put the shape in a measurement
key whenever the benchmark and named preset are not the same workload.

## Historical invalid gradcheck

The original `test_mappings_gradcheck[2]` failed on `copy_to` because it asked for a
rank-local property that contradicts the distributed mapping.

`f` is the identity in the forward and an all-reduce in the backward. That is the
definition — `test_f_and_g_are_conjugate` asserts exactly it, and requires `x.grad`
to come back as `world`, not 1.

`gradcheck` compares the analytic backward against a finite-difference estimate of
the *local* forward. The local forward is the identity, so the numerical Jacobian is
`I`. The analytic one is `N·I`. At two ranks that is a factor of 2, which is why the
local check failed.

Neither side is buggy. `f`'s backward is deliberately not the adjoint of its own
local forward — it is the adjoint of the global operation, where the input really is
replicated across the group and the gradients really do sum. That is the entire
content of the conjugate pair, and it is precisely the property gradcheck cannot see,
because gradcheck only knows about one process.

`reduce_from` passes, which fits: its forward is a real all-reduce and its backward
is the identity, so local and global agree.

The conjugate mapping was correct. A branch that made that local gradcheck happy would
have broken `test_f_and_g_are_conjugate` and every layer test downstream. The invalid
rank-local gradcheck has been removed; the distributed conjugacy and end-to-end
equivalence tests remain.

## The right panel

The scaling panel requires both a positive accelerator peak and a positive
interconnect bandwidth. Localhost communication profiles do not define an
accelerator compute ceiling, so the plotting code skips them explicitly instead
of inventing a peak or dividing by zero.

The panel is a derivation, not a measurement. I have no A100s. Working out where the
crossing point falls from published bandwidths and the alpha and beta I fitted in
Lab 0 is the point of the exercise, and the honest label on that chart is "predicted".
