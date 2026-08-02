# Hazard 3 — Hints

Hypothesis in `POSTMORTEM.md` first.

---

<details>
<summary><b>Hint 1 — where to look</b></summary>

The model initialisation is correct (that was hazard 2, and this script fixed it). The
data sharding is correct. The loss function is correct.

So look at `sync_gradients_fast`. Specifically, look at what it promises in its name and
what it actually guarantees.

Question to answer before reading on: after `sync_gradients_fast` returns, is
`param.grad` the averaged gradient?
</details>

---

<details>
<summary><b>Hint 2 — narrowing it down</b></summary>

`async_op=True` means the collective is *issued*, not *finished*. It returns a handle
immediately and the reduction happens in the background.

So between the `all_reduce` call and the handle's `wait()`, the contents of `param.grad`
are **undefined**. Some peers' contributions may have landed; some may not.

Two things read `param.grad` before anything waits:

```python
handles.append(handle)
param.grad /= world  # <-- reads and writes it, right now
```

and then `optimizer.step()` back in the caller.

There is no `wait()` anywhere in the function.

Run Lab 1's `tests/test_negative.py::test_reading_grad_before_wait_gives_a_partial_result`
if you have not. It prints the actual numbers for exactly this.
</details>

---

<details>
<summary><b>Hint 3 — the answer</b></summary>

Two bugs, and they compound.

**The missing wait.** The handles are collected and never waited on. `optimizer.step()`
runs against gradients that are in an indeterminate state.

**The division inside the loop.** `param.grad /= world` executes while the all-reduce for
that same tensor is in flight. That is a read-modify-write racing against the collective's
own writes. Whatever it produces is not the answer, and on some backends it is not even
deterministic.

The fix:

```python
def sync_gradients_fast(model):
    world = dist_utils.world_size()
    handles = []
    params = []

    for param in model.parameters():
        if param.grad is None:
            continue
        handles.append(collectives.all_reduce(param.grad, async_op=True))
        params.append(param)

    for handle in handles:
        handle.wait()  # every one, before touching any gradient

    for param in params:
        param.grad /= world  # now it is safe
```

The overlap survives. Every collective is still issued before any is waited on, so they
still run concurrently. What changed is that nothing reads a gradient until its collective
has completed.

**Why the symptom is so mild.** Partially reduced gradients are not random noise. They are
the sum of *some* ranks' contributions, which points in roughly the right direction with
roughly the wrong magnitude. So the model still learns, just less efficiently, and the
loss curve looks entirely normal.

That mildness is what makes this the most dangerous of the five. A hang gets fixed in an
hour. This ships.
</details>

---

## The general lesson

Asynchronous collectives have a contract, and the contract is:

**Between `async_op=True` and `handle.wait()`, the tensor's contents are undefined.**

Not "probably fine". Not "usually finished by then". Undefined. Code that reads it works on
your laptop with a small tensor, works in CI with a small tensor, and breaks in production
where the tensor is 200 MB and the network is busy.

The pattern that keeps you honest, and it is what Lab 1's `finalize()` is:

1. issue every collective
2. wait on every handle
3. only then read any of the tensors

Three phases, in that order, with no reads in phase 1.

And the meta-lesson, which is the whole reason this repository is built the way it is:
**you cannot find this bug by looking at the loss curve.** You find it by comparing against
a reference implementation. That is why every lab here has one, and why "the loss went
down" is never accepted as evidence anywhere in this repo.
