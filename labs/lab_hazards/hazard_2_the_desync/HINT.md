# Hazard 2 — Hints

Fifteen minutes first. Hypothesis in `POSTMORTEM.md` before you open anything below.

---

<details>
<summary><b>Hint 1 — where to look</b></summary>

The gradient averaging is correct. Check it if you like: every rank all-reduces every
gradient and divides by the world size, exactly as Lab 1 taught.

So if the gradients are identical across ranks at every step, and the optimizer is
deterministic, and the ranks still end up with different weights, then the difference
must have existed **before the first gradient was ever computed**.

Add this at the top of the training loop and see what it says:

```python
from common import testing

for name, p in model.named_parameters():
    testing.assert_close_across_ranks(p.data, name=name)
```
</details>

---

<details>
<summary><b>Hint 2 — narrowing it down</b></summary>

The ranks start from different weights.

Now, why? `build_model` seeds before constructing, so the construction is deterministic.
Read the seeding line carefully, and specifically the argument.

```python
dist_utils.seed_everything(1337, rank_offset=True)
```

What does `rank_offset=True` do? Read the docstring in `common/dist_utils.py`. It is
explicit about which situations want it and which do not.
</details>

---

<details>
<summary><b>Hint 3 — the answer</b></summary>

`rank_offset=True` seeds rank `r` with `1337 + r`. So each rank builds its layers from a
different random stream and gets different initial weights. Nothing broadcasts them, so
they stay different forever.

The comment above it is the trap, and it is the kind of comment that gets written in good
faith:

```python
# Seed per rank so that dropout masks and any other per-rank randomness
# differ. Standard practice.
```

That reasoning is correct for *data* and *dropout*. It is exactly wrong for *model
initialisation*. The two need opposite things from the RNG, and using one seeding call for
both is the bug.

**Two fixes, and you want both:**

```python
def build_model():
    dist_utils.seed_everything(1337, rank_offset=False)  # identical weights
    model = torch.nn.Sequential(...)
    dist_utils.sync_module_params(model)  # and prove it
    return model
```

`rank_offset=False` makes the construction identical. `sync_module_params` broadcasts rank
0's weights over the top, which makes it identical *even if something else consumed the
RNG differently on some rank*.

The second is not redundant. It is the one that keeps working when someone later adds a
layer whose initialisation depends on the data, or turns on a fused kernel that draws a
different number of random values, or loads a partial checkpoint. Every real DDP
implementation broadcasts at construction, and this is why.
</details>

---

## The general lesson

Write it in your own words. The shape of it:

The RNG has two jobs in distributed training and they want opposite things.

**Model initialisation must be identical across ranks.** Different weights means you are
training N unrelated models and averaging gradients between them, which converges to
something because gradient descent is robust, and produces a checkpoint that is the state
of one arbitrary rank.

**Data sampling and dropout must differ across ranks.** Identical data means your
effective dataset is 1/N of what you think, which is hazard 5. Identical dropout masks on
*sharded* activations means the effective dropout pattern is wrong, which is the RNG
problem Lab 2 documents.

So: seed identically, then diverge deliberately where you need to. And broadcast the model
anyway, because seeding is a promise about your own code and broadcasting is a fact.

The deeper point, and the one worth carrying: **a falling loss is not evidence of
correctness.** Every silent bug in this repository produces a falling loss. The only thing
that catches them is comparison against a reference, or an invariant check like
`assert_close_across_ranks`.
