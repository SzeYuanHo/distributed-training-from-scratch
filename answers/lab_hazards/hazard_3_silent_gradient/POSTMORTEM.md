# Hazard 3 — Postmortem

## 1. Symptom

The run completes. Nothing errors, the ranks agree about the model at the end, and
the loss falls. It just falls slightly short of the single-process reference the
script prints alongside it — a couple of percent by step 40, widening slowly.

Without that reference sitting next to it I would not have looked twice.

## 2. First hypothesis

Written before investigating: a learning-rate or scaling problem. A run that tracks
the reference and then lags it looked to me like gradients slightly too small, so I
guessed the divide by world size was being applied somewhere it should not be, or
applied twice.

Half right, and for the wrong reason. The divide is the problem, but not because it
happens twice.

## 3. How I confirmed it

The comm counter. `sync_gradients_fast` issued five async all-reduces and recorded
zero `wait[...]` events. That is the whole bug visible in one number, and it does not
require understanding the arithmetic at all.

Having seen that, the ordering in the loop is obvious on a reread: `param.grad /=
world` runs in the same iteration that issued the collective for that gradient.

## 4. Root cause

Two things, and the second is the one that bites. The handles are collected and never
waited on, so the gradients are read while still in flight. And the divide is applied
to a buffer mid-reduction, so it scales whatever has landed at that instant while the
contributions arriving afterwards are added at full weight.

The result is a gradient that is not a clean multiple of the right answer. It is a
weighted mixture whose weights depend on transport timing, which is why the lag is
small and drifts rather than being a tidy factor of N.

## 5. The fix

Three phases: issue every collective, wait on every handle, then divide.

Why that one: the obvious fix is to drop `async_op` and let each all-reduce block,
which is correct and gives up the concurrency the function was written for. Keeping
the issue loop separate from the wait loop keeps every collective in flight at once
and costs nothing — the reductions still overlap each other, I just stop reading the
buffers early.

## 6. The general lesson

Between `async_op=True` and `wait()` the tensor's contents are undefined. Not "mostly
finished", not "probably fine on a fast link". Undefined. The tensor has the right
shape and dtype and plausible magnitudes throughout, so nothing in the type system or
the runtime will tell you.

The habit: async collectives come in three phases and the phases do not interleave. If
I find myself touching a buffer inside the issue loop, that is the bug.

## 7. How would you catch this earlier?

The counter check generalises well: compare async operations issued against wait
events recorded, and assert they match at the end of every step. It is a couple of
lines, it is cheap, and it is a structural invariant rather than a numeric one, so it
does not need a reference run to be meaningful.

Beyond that, keep a single-process reference for the first few steps of any new
sync path. This bug is only visible as a comparison.

## 8. Time spent

Hypothesis to fix: about 20 minutes, and it would have been five if I had gone to the
counter before reasoning about the arithmetic.

## 9. A note on the loss I log

Not part of the bug. `worker` reported each rank's loss on its own quarter of the
batch, and `reference_losses` reports the whole batch, so the two curves compare
different quantities and disagree by more than the tolerance even when the gradients
are right. I all-reduce the logged value and divide by the world size so the curve
means what the comparison assumes it means. The gradients were already correct; only
the number being printed was misleading.
