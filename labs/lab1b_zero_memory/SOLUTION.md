# Lab 1B — Solution notes

---

## The ledger

The row people get wrong is the fp32 master copy, and specifically where it lives when
you shard.

It is tempting to file it under "parameters" and shard it at stage 3. It belongs to the
optimizer group and shards at stage 1. The reason: the master copy exists so the
*optimizer* can accumulate updates too small to represent in bf16. Only the rank that
runs the optimizer step for a given parameter needs it, and under ZeRO-1 that is one
rank. The bf16 copy used in the forward pass is a different tensor with a different
purpose, and it stays replicated until stage 3.

Get this backwards and you compute `2 + 2 + 8/N + 4`, which at N=8 is 9 bytes per
parameter instead of 5.5. It looks like a plausible answer and the test names the row.

The fp32-versus-bf16 result is worth sitting with. Both come to 16 bytes per parameter,
because dropping to bf16 halves the parameter and gradient rows but adds the master copy
you did not previously need. What mixed precision actually buys is halved activation
memory, halved bytes on the wire, and tensor-core throughput. Not state memory. If you
have ever told someone mixed precision halves their memory, this is the correction.

---

## `shard_parameters`

Greedy largest-first, assign to whichever rank currently has the fewest bytes:

```python
order = sorted(params, key=lambda p: -p.numel())
shards = [[] for _ in range(world_size)]
loads = [0] * world_size
for p in order:
    target = loads.index(min(loads))
    shards[target].append(p)
    loads[target] += p.numel()
```

Six lines and it handles the awkward case, which is this repo's tied embedding holding a
large share of the total. Round-robin gives one rank that tensor and the imbalance test
catches it.

For a model with one tensor larger than `total/N`, perfect balance is impossible and the
test's tolerance reflects that. Note the imbalance in your writeup rather than fighting
it: it is exactly why real ZeRO shards flattened byte ranges instead of whole tensors.

---

## `ZeRO1Optimizer`

The three-step sequence is the whole class:

```python
def step(self):
    self._reduce_scatter_gradients()  # each rank gets averaged grads for its shard
    self.inner.step()  # AdamW over local_params only
    self._all_gather_parameters()  # everyone ends up with the full updated model
```

**Reduce-scatter.** The straightforward implementation, and the one that keeps the byte
count honest, is a single reduce-scatter over a flat buffer. Build a flat gradient buffer
in shard order, so rank `r`'s slice is contiguous, then one
`reduce_scatter_tensor(out=my_slice, input=flat)`.

Getting the layout right is the fiddly part: the flat buffer must be ordered by owning
rank, not by parameter registration order, or the scatter hands each rank the wrong
slice. Build the ordering once in `__init__` and keep it.

The lazy alternative, one `reduce` per parameter to its owner, is correct and issues `P`
collectives instead of 1. It will pass the loss test and look wrong in the counter.

**All-gather.** Symmetrically, one `all_gather_into_tensor` over the flat parameter
buffer, then scatter the results back into the parameter tensors. Or, simpler and almost
as good, one `broadcast` per parameter from its owning rank. That is `P` collectives
again but the bytes are right, and for a first implementation it is defensible.

**The inner optimizer must only see the local shard.**

```python
self.inner = torch.optim.AdamW(self.local_params, lr=lr, ...)
```

Not `self.inner = AdamW(all_params)` with a filter at step time. Adam allocates `m` and
`v` for every parameter it is given, on the first step, whether or not you touched it.
That version is correct and saves nothing, and `test_optimizer_state_shrinks_with_world_size`
is there specifically for it.

**Free the gradients you do not own.** After the reduce-scatter, gradients for parameters
belonging to other ranks are dead weight. Setting them to `None` is what makes the
gradient row of your ledger honest. It is also the first step toward ZeRO-2, which never
materialises them in the first place.

---

## Why the communication is unchanged

Worth being able to state in one breath, because it is the least intuitive true thing in
this lab.

DDP does one ring all-reduce: `2(N-1)/N · P` bytes per rank.

A ring all-reduce *is* a reduce-scatter followed by an all-gather. Each phase moves
`(N-1)/N · P` bytes per rank. ZeRO-1 does exactly those two phases, with an optimizer
step wedged between them.

So ZeRO-1 gives you `12Ψ/N` bytes of memory back for zero additional bandwidth. That is
an unusually good trade and it is why stage 1 is close to free.

The latency profile does differ, and the writeup should say so: DDP issues one collective
and ZeRO-1 issues two, separated by a compute phase, so the second cannot start until the
optimizer step finishes. Same bytes, one more synchronisation point, less opportunity to
overlap. On a slow interconnect that matters.

ZeRO-3 is a different bargain entirely: it adds a parameter all-gather in the forward pass
and another in the backward, so roughly `2Ψ` extra bytes per rank per step. You are
buying memory with bandwidth, and whether that is a good deal depends on your
interconnect.

---

## Reading the measurements

Two things will look wrong and are not.

**Measured peak far above the analytic total.** The ledger covers states only. Add
activations, the autograd graph holding saved tensors until backward completes, cuBLAS
workspaces, and the allocator's block rounding. On the `small` preset at batch 8 the
activations alone are comparable to the states.

**Measured optimizer state of exactly zero.** You measured before the first `step()`.
Adam allocates lazily. This is a real behaviour worth having tripped over once, because
it also means a model that fits during the first forward pass can OOM on the first
optimizer step, which is a confusing way to find out you were near the limit.

On CPU the peak numbers come from process RSS and include the interpreter and torch
itself. Compare trends against N, not absolutes. The `optimizer state` column, which is
computed by summing state tensors rather than asking the OS, is trustworthy on either
platform, and it is the one the tests use.
