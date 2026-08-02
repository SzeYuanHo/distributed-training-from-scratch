# Lab 1 — Solution notes

Approach, not code. Read when stuck.

---

## Step 1: the flat buffer

Two ways to build a bucket, and the second is much better.

**Flatten on flush.** When the bucket fills, call `_flatten_dense_tensors([p.grad for p
in params])`, all-reduce the result, then `_unflatten_dense_tensors` and copy back:

```python
flat = torch._utils._flatten_dense_tensors(grads)
collectives.all_reduce(flat)
flat /= world_size
for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
    grad.copy_(synced)
```

Correct, and it allocates a full copy of the bucket on every flush.

**Preallocate and view.** At construction, allocate one contiguous buffer of
`total_bytes`, then point each parameter's `.grad` at a view of it:

```python
self.buffer = torch.zeros(total_numel, dtype=dtype)
offset = 0
for p in params:
    p.grad = self.buffer[offset : offset + p.numel()].view_as(p)
    offset += p.numel()
```

Now autograd writes gradients straight into the flat buffer and the flatten step costs
nothing. This is what `reducer.cpp` does, and it is why real DDP's memory profile is flat
during the backward pass.

The catch, and it is a real one: `optimizer.zero_grad(set_to_none=True)` sets `.grad` to
`None`, which throws away your view. Either use `set_to_none=False`, or re-establish the
views at the start of each iteration. The tests use `set_to_none=True` because that is
what everyone actually writes, so handle it.

Do not `torch.cat`. It allocates, it copies, and assigning back with slices detaches any
view someone else held.

---

## Step 2: bucket assignment

The loop is straightforward. The three details are not.

**Deduplicate by identity.** This repo's model ties `wte.weight` and `lm_head.weight`.
They are one tensor with two names. `list(model.parameters())` already deduplicates, but
`[p for _, p in model.named_parameters()]` does not, and a bucket list built from the
latter all-reduces the tied weight twice.

```python
seen = set()
unique = [p for p in params if p.requires_grad and id(p) not in seen and not seen.add(id(p))]
```

That `not seen.add(...)` trick is cute and unreadable; write the four-line loop instead.

**Filter frozen parameters.** `requires_grad=False` means no gradient ever, so a bucket
containing one never fills. The symptom is a hang in `finalize()`, and you will look for
it in the communication code.

**Reverse before you chunk, not after.** `assign_buckets` should iterate the reversed
list and fill buckets in that order, so `buckets[0]` genuinely holds the parameters that
arrive first. Reversing the finished bucket list gives you buckets whose *contents* are
in forward order, which loses most of the benefit.

---

## Steps 3 and 4: SimpleDDP and HookedDDP

`SimpleDDP` is twenty lines and the only thing to get wrong is the division. Do it.

For `HookedDDP`, the closure bug is worth stating explicitly because it is silent:

```python
# Wrong. Every hook sees the last p in the loop.
for p in self.module.parameters():
    p.register_post_accumulate_grad_hook(lambda param: self._sync(p))

# Right. Bind it.
for p in self.module.parameters():
    p.register_post_accumulate_grad_hook(self._make_hook(p))
```

`register_post_accumulate_grad_hook` passes the parameter to the callback, so you can
often use the argument and sidestep the capture entirely. `register_hook` passes the
gradient tensor instead, and then you do need the binding.

Keep the handles:

```python
self._handles.append(p.register_post_accumulate_grad_hook(fn))
```

and remove them in `close()`. The hook holds a reference to `self`, `self` holds the
module, and the tests build a lot of these.

---

## Steps 5 and 6: bucketing and overlap

The state machine is small. Write it out before you code it:

```
hook fires for param p
  -> if not syncing: return                     (no_sync)
  -> bucket = self._bucket_of[p]
  -> bucket.mark_ready(p)
  -> if bucket.is_full(): bucket.flush(async_op=self.overlap)
finalize()
  -> for each bucket with a handle: bucket.wait()
  -> for each bucket: bucket.reset()
  -> re-establish .grad views if you preallocated
```

`self._bucket_of` is a dict from `id(param)` to bucket. Keying on the parameter object
itself works too, since `nn.Parameter` hashes by identity, and reads better.

**Idempotent `finalize`.** A training loop with an early `continue` calls it twice.
Waiting on an already-waited handle should be a no-op, not an error.

**The `zero_grad` interaction, again.** If you preallocated views, `finalize()` is the
natural place to rebuild them, because it runs after `optimizer.step()` has consumed the
gradients and before the next backward writes new ones.

---

## Step 7: `no_sync`

Five lines and one trap.

```python
@contextlib.contextmanager
def no_sync(self):
    previous = self._syncing
    self._syncing = False
    try:
        yield
    finally:
        self._syncing = previous
```

Saving `previous` rather than setting `True` on exit makes nesting work. `try/finally` is
what makes an exception inside the body not leave syncing off forever, which is the case
`test_no_sync_restores_state_after_an_exception` checks.

The caller-side trap is in the README and worth repeating: the *last* micro-batch must
run outside the context. A loop where every iteration is inside `no_sync()` never
communicates at all, and each rank happily trains its own model.

---

## Step 8: unused parameters

Understand the failure before fixing it. Rank 0 uses `head_b`, rank 1 does not. Rank 0's
hook fires, its bucket fills, it issues an all-reduce, and it blocks waiting for rank 1
to join. Rank 1's hook never fires, so it never issues that collective, so it moves on to
whatever is next. Both ranks are now waiting for different things. Nothing errors.

Two strategies, and the cheap one is subtly wrong:

**Graph traversal (what real DDP does).** Before the backward pass, walk the autograd
graph from the output and collect every `AccumulateGrad` node you can reach. Any
parameter not in that set will not receive a gradient, so mark it ready immediately with
a zero gradient. Correct even when ranks disagree about which parameters are used,
because each rank discovers its own answer from its own graph, and the *bucket* still
gets flushed on every rank.

```python
def _find_unused(self, output):
    reachable = set()
    stack = [output.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen: continue
        ...walk node.next_functions...
    # parameters whose AccumulateGrad is not in `reachable` are unused
```

Fiddly, and worth doing once.

**Post-hoc bucket completion.** After `loss.backward()` returns, any bucket that did not
fill must be missing parameters that were unused, so mark them ready and flush. Cheaper.
And wrong in the case that matters: if rank 0 used `head_b` and rank 1 did not, rank 0
flushed that bucket during the backward pass and rank 1 flushes it in `finalize()`. The
collectives still pair up in this simple case, but the *ordering* relative to other
buckets differs between ranks, and with more than one such parameter you get a genuine
mismatch and a hang.

Either is acceptable for the lab. Say in a comment which failure you accepted.

---

## What the writeup should contain

Numbers, and an account of where they go.

Expect at Tier 1 with `tiny` on 2 ranks:

- Step time somewhere around 1.5 to 4 times the single-process time. The model is small
  enough that the backward pass is a few milliseconds and a single gloo all-reduce is
  comparable, so there is almost nothing to hide the communication behind.
- A bucket-size curve that is nearly flat above about 1 MB, because the whole `tiny`
  gradient set is roughly 3 MB and everything above that is one bucket anyway. Run the
  `small` preset if you want a curve with a visible minimum.
- Exposed communication somewhere between 40% and 90%. On a GPU cluster with a real model
  this number is under 10%, and the difference is entirely the compute-to-communication
  ratio.

The most valuable paragraph you can write is the one that answers question 3 honestly:
not "communication overhead" but "2 ranks, 3.1 MB of gradients, 4 all-reduces per step at
roughly 0.9 ms each, against a 2.3 ms backward pass, so there was never enough compute to
hide it behind." That sentence is the whole cost model applied to your own measurement,
and it is the thing that transfers to a cluster you have not seen yet.
