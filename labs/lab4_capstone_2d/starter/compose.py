"""Step 2: run data parallelism and tensor parallelism at the same time.

Data-parallel gradient averaging over the DP group. Tensor-parallel layers within
the TP group. Two groups, two kinds of collective, and the interesting part is
keeping them from stepping on each other.

**The rule that keeps it correct:** each parameter is either *replicated* across
the TP group or *sharded* across it, and how you synchronise it depends on which.

* A **sharded** parameter, such as a column-parallel weight, holds different values
  on each TP rank. Its gradient is already correct within the TP group and must be
  averaged only over the DP group.
* A **replicated** parameter, such as a LayerNorm weight, holds identical values on
  every TP rank. Every TP rank computed a gradient from the same replicated
  activations, so those gradients are already identical, and averaging over DP
  alone is still correct.

So in both cases: **average over the DP group only.** Never over both. Averaging
over the world divides by ``dp * tp`` when you meant to divide by ``dp``, and the
resulting effective learning rate is ``tp`` times too small. The loss falls,
slowly, and it looks like you need to tune the learning rate.

There is a real exception, and it is worth knowing even though this lab avoids it:
a parameter that is replicated across TP but whose gradient differs between TP
ranks does need a TP all-reduce as well. The classic case is a bias in a
sequence-parallel region. Megatron handles it explicitly. Say in a comment whether
your model has any.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Parallel2DModel", "average_gradients_over_dp"]


def average_gradients_over_dp(model: nn.Module, layout) -> None:
    """All-reduce every gradient over the DP group, divide by ``dp_size``.

    Args:
        model: this rank's model.
        layout: the ``MeshLayout`` from ``mesh.build_mesh``.

    Use ``layout.dp_group``, not the default group. Passing ``group=None`` here
    silently averages over the world and is the single most likely bug in this
    file.

    A no-op when ``dp_size == 1``, which keeps the ``tp``-only configurations
    cheap and makes the collective-count test easy to read.
    """
    raise NotImplementedError("Step 2a: loop, all-reduce on the DP group, divide by dp_size.")


class Parallel2DModel(nn.Module):
    """A model that is tensor-parallel within a node and data-parallel across nodes.

    ::

        layout = mesh.build_mesh(dp_size=2, tp_size=2)
        model = Parallel2DModel(config, layout)

        loss = model(x, y)[1]
        loss.backward()
        model.sync_gradients()      # DP group only
        optimizer.step()

    Construction, in order, and the order matters:

    1. Build the reference model identically on every rank, from the same seed.
    2. Shard it across the **TP group** using Lab 2's ``TensorParallelGPT``.
    3. Broadcast across the **DP group** so ranks with the same TP coordinate hold
       identical shards.

    Step 3 is easy to skip and produces the desync bug in a form that is harder to
    spot than usual: the TP group is internally consistent, so the model runs and
    the logits look plausible, and only the DP replicas disagree.

    **Data loading.** Each rank's batch depends on its **DP coordinate only**.
    Every rank in a TP group must see the *same* data, because they are jointly
    computing one forward pass over it. So::

        get_batch(..., rank=layout.dp_rank, world_size=layout.dp_size, ...)

    Passing the global rank instead gives each TP rank a different batch, and the
    tensor-parallel arithmetic is then combining activations from unrelated
    examples. Nothing raises. The loss falls. It is nonsense.

    Args:
        config: a ``GPTConfig``.
        layout: the ``MeshLayout``.

    If Lab 2 is unfinished, this lab still works with plain replicated layers and
    ``tp_size=1``. The mesh construction and the planner are the parts that are
    unique to Lab 4, and neither depends on Lab 2.
    """

    def __init__(self, config, layout):
        super().__init__()
        raise NotImplementedError("Step 2b.")

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        raise NotImplementedError("Step 2c: pass through to the sharded model.")

    def sync_gradients(self) -> None:
        """Average gradients over the DP group. Call before ``optimizer.step()``."""
        raise NotImplementedError("Step 2d.")

    def parameters(self, recurse: bool = True):
        raise NotImplementedError("Step 2e.")
