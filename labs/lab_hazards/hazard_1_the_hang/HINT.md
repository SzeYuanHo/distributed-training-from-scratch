# Hazard 1 — Hints

Try for fifteen minutes first. Being stuck is where the learning is; a hint collected too
early buys you a fix and costs you the skill.

Have you written your hypothesis in `POSTMORTEM.md` yet? Do that before opening any of
these.

---

<details>
<summary><b>Hint 1 — where to look</b></summary>

Run it with the debug flag on:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL python labs/lab_hazards/hazard_1_the_hang/broken.py
```

Read the last collective each rank logged before the silence. Are they the same
collective? Is every rank in the same place?

The general principle: a collective is a rendezvous. Every rank has to arrive, and they
have to arrive at the *same* one in the *same order*. When they do not, the ones that
arrived wait forever for the ones that did not.

So: does every rank execute the same sequence of collective calls in this program?
</details>

---

<details>
<summary><b>Hint 2 — narrowing it down</b></summary>

Ranks 1, 2 and 3 are all blocked inside an `all_reduce`. Rank 0 is somewhere else.

That pattern is the signature. When most ranks are stuck in a collective and one is not,
the one that is not is the cause, and the others are its victims. Their stacks all point
at the collective, which makes it look like the collective is broken. It is not.

Now look for something in this program that only rank 0 does.

Count the collectives each rank issues in one call to `train_step`. Do the counts match?
</details>

---

<details>
<summary><b>Hint 3 — the answer</b></summary>

The logging block:

```python
if step % 3 == 0 and rank == 0:
    total_norm = torch.zeros(1)
    for param in model.parameters():
        total_norm += param.grad.norm() ** 2
    collectives.all_reduce(total_norm)  # <-- only rank 0 reaches this
    print(...)
```

There is an `all_reduce` inside `if rank == 0`. On step 0, rank 0 enters that block and
calls a collective that ranks 1, 2 and 3 never call. Rank 0 blocks waiting for three peers
who have already moved on to step 1. Everyone waits forever.

This is the most common hang in real distributed training, and it almost always arrives
the same way: someone adds logging, and the logging computes something that needs a
collective, and the logging is guarded by `if rank == 0` because you only want to print
once.

**Two ways to fix it, and they are not equivalent:**

*Every rank participates in the collective; only rank 0 prints.*

```python
if step % 3 == 0:
    total_norm = torch.zeros(1)
    for param in model.parameters():
        total_norm += param.grad.norm() ** 2
    collectives.all_reduce(total_norm)  # every rank
    if rank == 0:
        print(...)  # one rank
```

*Do not communicate at all; compute the norm locally.*

```python
if step % 3 == 0 and rank == 0:
    total_norm = sum(p.grad.norm() ** 2 for p in model.parameters())
    print(...)
```

The second is cheaper and reports rank 0's local norm rather than the global one. After
the gradient all-reduce every rank holds identical gradients, so here they are the same
number, and knowing *why* they are the same is the point.

The first is what you want when the quantity genuinely differs across ranks, such as a
per-rank loss or a data-dependent count.

Pick one. Say in the postmortem which and why.
</details>

---

## The general lesson

Write this one in your own words, because it generalises further than the fix does.

Collectives are a rendezvous. Any control flow that causes ranks to issue different
collectives, or the same collectives in a different order, hangs. Which means:

- `if rank == 0:` around anything that communicates is a bug waiting to happen
- so is `if loss < threshold:` when the loss differs across ranks
- so is `for batch in dataloader:` when the dataloaders have different lengths
- so is an early `break` or `continue` that only some ranks take

The habit worth building: whenever you write a branch in distributed code, ask whether
every rank takes the same side of it. If they might not, no collective may appear inside.
