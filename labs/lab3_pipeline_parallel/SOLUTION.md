# Lab 3 — Solution notes

---

## Partitioning

Mostly bookkeeping. The two things worth care:

**Which stage owns what.** Stage 0 gets `wte`, `wpe`, `drop` and its blocks. The last
stage gets its blocks, `ln_f` and `lm_head`. Middle stages get blocks only. Encode that
in the `is_first` / `is_last` flags rather than by comparing indices inside `forward`, so
the branch is in one place.

**`activation_shape`.** Every receiver needs the sender's shape before it can allocate.
For a middle stage that is `(B, T, C)` and for stage 0's *input* it is `(B, T)` integers.
Compute it locally rather than sending metadata: every rank knows the config, so every
rank can work out what its neighbour will send. Metadata exchange is the general solution
and it costs a round trip per micro-batch, which at `m=16` is not free.

**Weight tying.** The reference model can tie `wte` and `lm_head`, and under a depth split
those land on different stages. `reference/single_process.py` builds its config with
`tie_weights=False` so the tests do not force a decision on you. In production you either
replicate the weight on both stages and all-reduce its gradient, or drop the tie. Say
which you would do and why.

---

## Point-to-point

Use `batch_isend_irecv`. The whole function:

```python
ops = []
if send_to is not None:
    ops.append(collectives.P2POp(dist.isend, send_tensor, send_to))
if recv_from is not None:
    buffer = torch.empty(recv_shape, dtype=recv_dtype)
    ops.append(collectives.P2POp(dist.irecv, buffer, recv_from))
reqs = collectives.batch_isend_irecv(ops)
for r in reqs:
    r.wait()
```

The backend sees every transfer at once and can order them however it likes, so the
symmetric-send deadlock cannot happen. Odd/even ordering also works and is free; it just
puts the correctness argument in your head instead of in the backend, and Lab 3 has enough
of those already.

**`requires_grad` on the received tensor is the thing to get right.** The autograd graph
stops at the process boundary. Stage 1 receives a plain tensor with no history. Marking it
as a leaf requiring grad means the backward pass accumulates into it, and *that* is what
you send upstream:

```python
x = recv_activation(shape, dtype, src=prev)
x.requires_grad_(True)
out = stage(x)
# ... later, in the backward phase ...
out.backward(gradient=received_grad)
send_activation(x.grad, dst=prev)
```

Miss it and stage 0's parameters never update. The loss falls, because stages 1 to 3 are
still learning, and you have silently frozen the embedding and the first blocks.
`test_gradients_reach_every_stage` is the guard.

---

## AFAB

The structure is two loops:

```python
outputs, inputs = [], []
for i in range(n_micro):
    x = batches[i] if stage.is_first else recv_activation(...)
    if not stage.is_first:
        x.requires_grad_(True)
    out = stage(x, targets[i] if stage.is_last else None)
    inputs.append(x); outputs.append(out)
    if not stage.is_last:
        send_activation(out, next_stage)

for i in reversed(range(n_micro)):
    ...backward, send input gradient upstream...
```

Keep both lists. `outputs` holds the graph, and dropping a reference frees it, after which
the backward for that micro-batch fails or silently produces nothing. That list is
literally the activation memory the schedule is criticised for, and seeing it as a python
list makes the `O(m)` claim concrete.

Scale each micro-batch loss by `1/n_micro` before `backward`. Same arithmetic as Lab 1's
gradient accumulation, same failure if you skip it.

Backward in reverse order. It works either way numerically; reverse order releases the
most recently allocated activations first, which is friendlier to the allocator.

---

## 1F1B

The warmup count is the whole schedule:

```python
def warmup_steps(stage_index, n_stages, n_micro):
    return min(n_stages - stage_index, n_micro)
```

Check it against the test's expectations: at `p=4`, stage 0 gets 4 and stage 3 gets 1.
Stage 3 can start its backward as soon as its own forward finishes, because nothing is
downstream of it. Stage 0 has to wait for the first micro-batch to travel all the way
down and its gradient all the way back, which is `p` steps.

The `min(..., n_micro)` matters. With fewer micro-batches than stages there is no steady
state, and a warmup longer than the available work schedules operations that never happen.

The schedule itself:

```python
warmup = warmup_steps(stage_index, n_stages, n_micro)
ops = [Op("F", i) for i in range(warmup)]
f, b = warmup, 0
while f < n_micro:
    ops.append(Op("B", b))
    b += 1
    ops.append(Op("F", f))
    f += 1
ops += [Op("B", i) for i in range(b, n_micro)]
```

Check the output by hand against the table in the starter docstring before running
anything. `test_00_math.py` checks it exactly, so an off-by-one costs you a millisecond
rather than a two-minute timeout.

In the executor, the activation store becomes a bounded queue instead of a growing list. A
forward pushes, a backward pops the oldest. The depth never exceeds the warmup count, and
that bound is the entire memory argument.

**In the steady state, issue both exchanges together.** A stage receives an activation from
upstream and a gradient from downstream in the same iteration. Two separate `exchange`
calls reintroduce an ordering dependency; one call with both `P2POp`s does not.

---

## Reading the results

**The elapsed idle fraction will usually measure worse than
`(p-1)/(m+p-1)`.** The related `(p-1)/m` quantity is overhead relative to ideal
work and is not directly comparable to `1 - compute_occupancy`. Three contributors
can widen the measured gap:

*Stage imbalance* is usually the largest. The formula assumes every stage takes the same
time. Run `stage_flops` on your split and you will see the last stage carrying the LM
head. At `V=72` the imbalance is mild; the test shows what happens at `V=128000`.

*Point-to-point latency* is second. The formula counts idle slots, not microseconds spent
moving tensors. On gloo over loopback that is tens of microseconds per exchange, times
`2·m·(p-1)` exchanges.

*Python overhead* is third and not negligible at these model sizes: one interpreter frame
per micro-batch per stage.

**The memory difference will look smaller than `m/p`.** The measurement includes
parameters, the interpreter, and torch itself, none of which change between schedules.
`test_00_math.py` checks the exact bound at the schedule level, where there is nothing else
in the number. Quote both: the schedule-level ratio is the claim, the measured ratio is the
claim diluted by everything else in the process.

**Question 1 has a precise answer** and it is worth writing carefully: between the two
charts, the *activation memory* changes and the *bubble* does not. People expect 1F1B to
be faster. It is not. It uses less memory, which lets you raise `m`, which then shrinks
the bubble. The speedup is second-order and the memory saving is the mechanism.

---

## Production API comparison

The direct mapping is `ScheduleGPipe` to the all-forward-all-backward schedule and
`Schedule1F1B` to the warmup/steady-state/cooldown schedule. Both preserve the same
micro-batch loss scaling and gradient accumulation rules as the implementations here.

The production API additionally owns stage metadata, static input/output shapes,
communication setup and schedule/runtime integration. This lab keeps those mechanics
visible by constructing point-to-point operations and autograd boundaries directly.
The optional API smoke test verifies that the two named production schedules exist on
the supported Torch line; it does not claim a version-independent numerical benchmark
against that evolving API.
