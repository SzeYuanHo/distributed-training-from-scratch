"""Steps 3 to 5: ZeRO stage 1, optimizer-state sharding.

The idea in one sentence: every rank keeps optimizer state for only `1/N` of the
parameters, steps that slice, and all-gathers the result.

The step sequence, and the order is not negotiable:

1. **Reduce-scatter** the gradients, so rank `r` ends up with the averaged
   gradient for exactly the parameters it owns. Not an all-reduce: you would be
   paying to receive gradients you are not going to use.
2. **Step** the local optimizer on rank `r`'s shard only. Its `m` and `v` tensors
   cover `1/N` of the parameters, which is where the memory saving comes from.
3. **All-gather** the updated parameters, so every rank ends the step holding the
   full updated model, exactly as DDP would.

Steps 1 and 3 together move the same number of bytes as DDP's single all-reduce.
That is not a coincidence and THEORY section 3 asks you to show why.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

__all__ = ["ZeRO1Optimizer", "shard_parameters"]


def shard_parameters(params: list[nn.Parameter], world_size: int) -> list[list[nn.Parameter]]:
    """Assign each parameter to exactly one rank, balancing total bytes.

    Args:
        params: trainable parameters, deduplicated by identity.
        world_size: N.

    Returns:
        ``result[r]`` is rank r's parameters. Every parameter appears exactly once
        across the whole list.

    THEORY section 6 asked you to choose between sharding by parameter and
    sharding by flattened byte range. This signature commits you to by-parameter,
    which is what ``ZeroRedundancyOptimizer`` does. Real ZeRO does byte ranges and
    gets perfect balance at the cost of splitting tensors across ranks.

    Balance matters more than it looks. This repo's model ties the embedding and
    the LM head, so one tensor is a large fraction of the total. A naive
    round-robin gives one rank that tensor and everyone else the LayerNorms, and
    your memory saving evaporates. A greedy largest-first assignment to the
    currently-emptiest rank is a few lines and much better.

    Assert every rank gets at least one parameter, with a message naming the
    counts. A rank with an empty shard has no optimizer state, contributes nothing
    at step time, and will surface later as a confusing all-gather shape mismatch.
    """
    raise NotImplementedError("Step 3a: greedy balanced assignment.")


class ZeRO1Optimizer:
    """AdamW with optimizer state sharded across ranks.

    ::

        model = make_model("tiny")
        dist_utils.sync_module_params(model)
        optimizer = ZeRO1Optimizer(model.parameters(), lr=1e-3)

        loss.backward()
        optimizer.step()        # reduce-scatter, local step, all-gather
        optimizer.zero_grad()

    Note what this class replaces. In Lab 1 the DDP wrapper synchronised gradients
    and the optimizer was ordinary. Here the *optimizer* owns the communication,
    because the sharding and the stepping have to agree about who owns what. That
    inversion is why ZeRO is an optimizer in DeepSpeed and not a model wrapper.

    Args:
        params: iterable of parameters, as you would pass to any optimizer.
        lr, betas, eps, weight_decay: ordinary AdamW knobs.
        process_group: defaults to the world.

    Attributes the tests inspect:
        shards: ``list[list[nn.Parameter]]``, one entry per rank.
        local_params: this rank's shard.
        inner: the wrapped ``torch.optim.AdamW`` over ``local_params`` only.

    **The memory claim, and how to keep it true.** The saving comes entirely from
    ``inner`` holding state for `1/N` of the parameters. Build it over
    ``local_params``, not over everything. An implementation that constructs
    ``AdamW(all_params)`` and then only steps some of them allocates `m` and `v`
    for all of them on the first step, gives the right answer, and saves nothing.
    ``test_optimizer_state_shrinks_with_world_size`` exists to catch that.

    **A gradient your rank does not own is dead after the reduce-scatter.** Free it
    rather than leaving it attached, or the gradient row of the ledger never
    shrinks either.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        process_group=None,
    ):
        raise NotImplementedError(
            "Step 3b: shard the parameters, build AdamW over this rank's shard only."
        )

    def _reduce_scatter_gradients(self) -> None:
        """Average gradients and leave each rank holding only its own shard's.

        The straightforward implementation is one reduce-scatter over a flat
        buffer covering all parameters, with the shard layout deciding the split.
        A simpler one that also works: one ``reduce`` per parameter to its owning
        rank. Correct, and `P` collectives instead of 1, so the collective-count
        test will notice.

        Use ``common.collectives`` so the CommCounter sees it.
        """
        raise NotImplementedError("Step 3c.")

    def _all_gather_parameters(self) -> None:
        """Broadcast each rank's updated shard so everyone holds the full model.

        After this, ``assert_close_across_ranks`` on any parameter must pass. If it
        does not, the ranks have diverged and you have rebuilt Lab H hazard 2.
        """
        raise NotImplementedError("Step 3d.")

    def step(self, closure=None) -> Any:
        """Reduce-scatter, step the local shard, all-gather. In that order.

        Reversing the last two is the classic bug: all-gather stale parameters,
        then step locally, and every rank now holds a model that is one step behind
        everyone else's view of it. The loss still falls.
        """
        raise NotImplementedError("Step 3e.")

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on *every* parameter, not just this rank's shard.

        Autograd writes gradients for all parameters during the backward pass,
        whether or not this rank owns them, so all of them need clearing.
        """
        raise NotImplementedError("Step 3f.")

    def state_dict(self) -> dict:
        """This rank's shard of the optimizer state.

        Worth thinking about even though nothing tests it: a ZeRO checkpoint is
        distributed, and reloading it on a different world size means
        re-sharding. That is why every framework that implements ZeRO also ships
        a checkpoint-consolidation script.
        """
        raise NotImplementedError("Optional.")

    def local_state_bytes(self) -> int:
        """Bytes of optimizer state this rank holds. Used by the memory test.

        Sum over ``inner.state`` of every tensor's byte count. Should come out
        close to `2 * 4 * (Ψ/N)` for AdamW, plus the master copy if you keep one.
        Returns 0 before the first ``step()``, because Adam allocates lazily, and
        that is a real behaviour rather than a bug in the measurement.
        """
        raise NotImplementedError("Step 5: sum the state tensors.")
