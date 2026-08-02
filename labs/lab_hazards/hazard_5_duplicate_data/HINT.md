# Hazard 5 — Hints

Hypothesis in `POSTMORTEM.md` first.

---

<details>
<summary><b>Hint 1 — where to look</b></summary>

The gradient averaging is correct. The model initialisation is correct. The optimizer is
correct.

The script prints a batch fingerprint per rank per step. Look at what it printed. Are the
four columns different?

If four ranks compute gradients on identical data and then average them, what is the
average?
</details>

---

<details>
<summary><b>Hint 2 — narrowing it down</b></summary>

`get_my_batch` has a docstring that says it shards data across ranks. Read the arguments
it actually passes:

```python
return data_mod.get_batch(
    "train",
    batch_size=BATCH_PER_RANK,
    block_size=BLOCK,
    rank=0,  # <--
    world_size=1,  # <--
    step=step,
    dataset=dataset,
)
```

Now read the signature of `common.data.get_batch` and what it does with `rank` and
`world_size`. The docstring there spells out the property it guarantees.
</details>

---

<details>
<summary><b>Hint 3 — the answer</b></summary>

`rank=0, world_size=1` are hardcoded. Every rank asks for rank 0's slice of a
single-process batch, so every rank gets the same data.

```python
return data_mod.get_batch(
    "train",
    batch_size=BATCH_PER_RANK,
    block_size=BLOCK,
    rank=dist_utils.rank(),
    world_size=dist_utils.world_size(),
    step=step,
    dataset=dataset,
)
```

Two lines. That is the whole fix, and the docstring above it was describing what the
author intended rather than what the code did, which is why reading the docstring instead
of the arguments does not find it.

**What actually happens with the bug in place.** Four ranks compute gradients on identical
data. The all-reduce averages four identical tensors, which returns the same tensor. So
the job is a single-process run with four times the power draw, and the effective dataset
is one quarter of what you think, because in the time the correct version sees four
distinct batches this one sees one batch four times.

The training loss falls *faster*, which is the part that makes this hard to spot: seeing
the same examples repeatedly does improve the training loss quickly. The validation loss
then climbs, and it looks like a regularisation problem.

**Why hardcoded defaults are dangerous here specifically.** `get_batch` defaults to
`rank=0, world_size=1` because that is right for the single-process reference runs the
tests use. Copying a call from a test into distributed code and not changing them is a
completely natural way to arrive at this bug.
</details>

---

## The general lesson

**Verify your data sharding, do not assume it.** Hash each rank's batch and assert the
hashes differ. `common.data.batch_fingerprint` exists for this and it costs one line:

```python
prints = testing.gather_object(data_mod.batch_fingerprint(x))
assert len(set(prints)) == world_size, f"ranks share data: {prints}"
```

Put that behind a debug flag in your own training loops. It has caught this bug for people
more than once.

**The disjointness property is a contract worth writing down.** `common/data.py` guarantees
that the union of the ranks' batches at step `s` equals the single-process batch of `N`
times the size. That is what makes an `N`-rank run comparable to a 1-rank reference, and
it is why Lab 1's loss-curve equivalence test is meaningful at all. A loader that does not
guarantee it makes every equivalence test in this repo untestable.

**And the recurring theme.** Faster convergence is not evidence that anything is right.
Every hazard in this lab produces a metric that looks fine or better than fine:

- hazard 2: the loss falls smoothly
- hazard 3: the loss falls, slightly slower
- hazard 5: the loss falls *faster*

The one thing that separates them from a correct run is comparison against a reference,
or an invariant asserted directly. Which is the design principle the whole repository is
built on.
