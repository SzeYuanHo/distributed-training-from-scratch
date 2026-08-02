# Lab 2 — Solution notes

The longest of these, because the lab is the longest. Read the section you are stuck on.

---

## The mappings

Both `Function`s are four lines each. The difficulty is entirely in believing the answer.

`f` (`CopyToTensorParallelRegion`): forward returns the input untouched, backward
all-reduces the incoming gradient.

`g` (`ReduceFromTensorParallelRegion`): forward all-reduces, backward returns the
incoming gradient untouched.

If the backward of `f` feels arbitrary, work the column-parallel case explicitly.
`Y_i = X W_i`, so `∂L/∂X` gets a contribution `(∂L/∂Y_i) W_iᵀ` from rank `i`. Each rank
computes one term of a sum. The true `∂L/∂X` is the sum over all ranks. Something has to
add them up, and the only node in the graph that sits at the point where `X` was
replicated is `f`. So `f`'s backward is the all-reduce.

`g` is the mirror. Its forward *was* the sum, and `d(a+b)/da = 1`, so the gradient
arrives at every term unchanged.

Two implementation notes:

`backward` must return one value per input to `forward`. Both of these take
`(tensor, group)`, so return `(grad, None)`. Returning a bare tensor gives you a confusing
error about the number of gradients.

Do not use `ctx.save_for_backward` here. Neither backward needs the forward's input, and
saving it holds a reference to a tensor you do not need, which shows up as unexplained
memory growth in a deep model.

---

## The layers

**The layout confusion is real and it is worth naming.** THEORY.md writes `W` as
`(in, out)` because that is how linear algebra writes it. `torch.nn.Linear` stores
`weight` as `(out, in)` and computes `x @ weight.T`. So:

- column-parallel (split the *output* dimension) splits `weight` along **dim 0**
- row-parallel (split the *input* dimension) splits `weight` along **dim 1**

Which reads backwards from the names. `reference/plain_blocks.py` has
`split_weight_columns` and `split_weight_rows` with a comment about exactly this, and
using those helpers rather than inline `chunk` calls keeps the confusion in one place.

**Column-parallel forward:**

```
x -> f(x) -> F.linear(x, self.weight, self.bias) -> optional gather
```

The `f` looks like a no-op and it is, in the forward pass. Take it out and the outputs
still match, the gradients silently do not, and `test_column_parallel_linear` catches it
because it checks `x.grad` and not just the output.

**Row-parallel forward:**

```
x -> optional scatter -> F.linear(x, self.weight, bias=None) -> g -> + self.bias
```

`bias=None` in the linear call is the whole trick. The bias goes on after `g`. If you
pass it to `F.linear` you get `t` copies of it in the sum, and the Tier-0 test
`test_row_split_bias_added_once` shows exactly what that looks like: the output is off by
`(t-1)·b`, which for small biases is a small error that does not obviously scream "bug".

**`load_from_reference`** is where the sharded model meets the checkpoint. It is four
lines and worth an assertion on shapes, because a mismatch here produces a model that
runs and computes nonsense.

---

## The blocks

The MLP is ten lines. `ColumnParallelLinear(h, 4h, gather_output=False)`, GELU,
`RowParallelLinear(4h, h, input_is_parallel=True)`. Those two flags are the entire
argument from THEORY section 3, expressed as arguments.

Attention is where the time goes, and specifically `load_from_reference`.

`c_attn.weight` is `(3h, h)`. The rows are Q for all heads, then K for all heads, then V
for all heads. Rank `i` needs its heads' rows out of all three blocks. A plain
`chunk(t, dim=0)` gives rank 0 the whole of Q, which is wrong in a way that produces
correctly-shaped tensors and nonsense values.

The clean approach:

```python
w = reference.c_attn.weight  # (3h, h)
w = w.view(3, n_head, head_dim, hidden)
mine = w[:, rank * local_heads : (rank + 1) * local_heads]  # (3, local_heads, head_dim, h)
self.c_attn.weight.data.copy_(mine.reshape(-1, hidden))
```

The bias, if present, gets the same treatment with one fewer dimension:
`view(3, n_head, head_dim)`.

`tests/test_00_math.py::test_qkv_fused_weight_slicing_is_not_a_plain_chunk`
demonstrates the difference on a small example. Run it and look at the numbers before you
write this method.

The rest of attention is the reference implementation with `n_head` replaced by
`local_heads`. The causal mask is *not* sharded: every rank computes attention over the
full sequence for its own heads, so it needs the whole `(1, 1, T, T)` mask.

---

## Vocab parallelism

The embedding is a masking exercise:

```python
in_range = (ids >= self.vocab_start) & (ids < self.vocab_end)
local = (ids - self.vocab_start).clamp(0, self.num_local - 1)
out = F.embedding(local, self.weight)
out = out * in_range.unsqueeze(-1)  # <- the step people forget
reduce_from_tp_region(out)
```

The `clamp` is not cosmetic. Passing a negative index to `F.embedding` raises on CPU and
reads out of bounds on some CUDA builds, which is the worst kind of bug. Clamp to
something valid, look up, then zero the result.

Cross-entropy is the five steps from THEORY section 7, and the ordering matters:

```
1. local max -> all_reduce(MAX)  -> subtract
2. local sum of exp -> all_reduce(SUM)
3. masked target logit -> all_reduce(SUM)
4. loss = log(sum) - target
```

Two ways to handle the backward. Writing an explicit `torch.autograd.Function` with
`grad = softmax - onehot` is what Megatron does and it is faster. Building it from
differentiable ops and letting autograd figure it out is easier to get right, and every
collective you use is already a `Function`, so the graph is correct by construction. For
a first implementation, take the second.

Step 1 matters more than it looks. With toy logits in `[-5, 5]` you can skip the max
subtraction and pass every test. With real logits reaching 30, `exp(30)` is `1e13`, which
overflows fp16 and loses precision in fp32. `test_vocab_parallel_cross_entropy[True]`
puts a logit of 100 in the tensor for exactly this reason.

---

## The full model

`from_reference` is mechanical once the components have `load_from_reference`. The two
decisions worth stating:

**Weight tying survives sharding**, because the embedding is sharded by vocabulary row
and the LM head is column-parallel sharded by output dimension, and those are the same
axis. So after loading both, tie them: `self.lm_head.weight = self.wte.weight`. An untied
parallel model produces identical logits at step 0 and drifts after the first optimizer
step, which is a genuinely confusing failure to debug.

**The logits shape.** Each rank naturally holds `(b, s, V/t)`. Gathering them costs a
collective on the largest tensor in the model. The rule in the starter docstring is:
gather when `targets is None`, keep sharded when computing the loss. Document whichever
rule you pick, because the test has to know which one you chose.

---

## When the full model does not match and the layers do

Almost always one of three things, in order of likelihood:

1. **The QKV slicing.** See above. This accounts for most of it.
2. **Weight tying**, either lost or applied to the wrong sharding axis.
3. **The position embedding got sharded.** It should not be. It is
   `(block_size, n_embd)`, replicated, and sharding it produces the wrong positions on
   every rank but rank 0.

Bisect by comparing intermediate activations. Run the reference and the parallel model on
the same input, hook both after each block, and compare. The first block where they
diverge is your bug, and `common/testing.py::compare_to_reference` will tell you which
element and by how much.

---

## What the writeup should contain

The scaling-ceiling number is the deliverable, and it is the one thing here that would
survive being shown to someone outside the exercise.

Doing it properly: take `h=4096`, `b·s=8192`, fp16. Compute per layer per rank is
`24·b·s·h²/t` FLOPs. Communication per layer is two all-reduces of `b·s·h·2` bytes at
ring cost `2(t-1)α + 2(t-1)/t·M/BW`. Set them equal and solve for `t`.

On a T4 pair over the 12 GB/s PCIe planning assumption the answer is small, in the low
single digits. Under the 300 GB/s NVLink assumption it is comfortably past 8. This is
why tensor parallelism is usually kept within a fast interconnect domain; slower
cross-node links make it less attractive, although actual layouts depend on the model,
topology, and available parallelism dimensions.

That paragraph, with your own arithmetic in it, is the thing to be able to reproduce on a
whiteboard.

The other question worth taking seriously is number 3: why the loss curve would never
have revealed an `f`/`g` swap. The precise answer is that the swap leaves the forward
computation unchanged, because both orderings evaluate the same function; only the
gradient is wrong, and it is wrong by a factor that behaves like a modified learning rate
rather than like noise. So the loss falls, more slowly, and "more slowly" is
indistinguishable from a hyperparameter you have not tuned yet. `tests/test_negative.py`
prints the actual factor. Quote it.
