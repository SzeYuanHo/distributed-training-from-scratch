"""Step 3: the simplest thing that is actually correct.

No bucketing, no overlap, no hooks. After ``loss.backward()`` returns, walk the
parameters and all-reduce each gradient. Slow, and correct, and the baseline every
later variant is checked against.

Get this one right before touching hooks. Every subsequent bug is easier to find
when you know the semantics were right at some point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SimpleDDP"]


class SimpleDDP(nn.Module):
    """Data-parallel wrapper. Averages gradients after the backward pass.

    ::

        model = SimpleDDP(reference_gpt.make_model("tiny"))
        for step in range(20):
            loss = model(x, y)[1]
            loss.backward()
            model.sync_gradients()      # the part this class adds
            optimizer.step()
            optimizer.zero_grad()

    Two things happen, and both are load-bearing.

    **At construction: broadcast rank 0's parameters to everyone.**
    ``nn.Module`` initialisation uses the global RNG, and although every rank
    seeded identically will produce identical weights *in this repo*, relying on
    that is exactly the desync bug in Lab H hazard 2. Any real model has a data
    order, a dropout mask, or a `torch.compile` decision that breaks the symmetry.
    Broadcast. ``dist_utils.sync_module_params`` does it, and buffers matter too:
    a BatchNorm running mean that differs per rank produces a model that evaluates
    differently depending on who saved it.

    **After backward: all-reduce each gradient and divide by the world size.**
    ``ReduceOp.SUM`` gives N times the mean. Dividing is what makes N-way DDP
    equivalent to one process with N times the batch, which is the property every
    test in this lab checks.

    Attributes:
        module: the wrapped model. ``self.module.parameters()`` is what the
            optimizer should be given, and ``SimpleDDP`` forwards ``parameters()``
            so either works.
    """

    def __init__(self, module: nn.Module, process_group=None):
        super().__init__()
        raise NotImplementedError(
            "Step 3a: store the module, broadcast its parameters and buffers from rank 0."
        )

    def forward(self, *args, **kwargs):
        """Pass through to the wrapped module. One line."""
        raise NotImplementedError("Step 3b.")

    def sync_gradients(self) -> None:
        """All-reduce every gradient and divide by the world size.

        Skip parameters whose ``.grad`` is None. That happens for genuinely unused
        parameters, and it also happens on the very first call if you managed to
        call this before any backward pass. Both cases should be a no-op rather
        than an ``AttributeError`` three frames deep.

        Use ``common.collectives.all_reduce``, not ``dist.all_reduce``, so the
        CommCounter can see it. The tests assert on those counts.
        """
        raise NotImplementedError("Step 3c: loop, all-reduce, divide.")

    def parameters(self, recurse: bool = True):
        """Forward to the wrapped module so the optimizer sees the real parameters."""
        raise NotImplementedError("Step 3d.")

    def state_dict(self, *args, **kwargs):
        """Return the *wrapped* module's state dict, without the ``module.`` prefix.

        Real DDP does not do this, which is why every checkpoint-loading utility
        in the world has a line that strips ``module.`` from the keys. Doing it
        here means a `SimpleDDP` checkpoint loads straight into a plain model,
        which makes the equivalence tests read better.
        """
        raise NotImplementedError("Step 3e.")
