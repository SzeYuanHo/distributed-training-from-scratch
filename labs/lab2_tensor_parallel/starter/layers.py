"""Step 3: the two parallel linear layers everything else is built from.

Get these right and the MLP, attention, and the full model are mostly plumbing.
Get the bias wrong and you will spend an hour on a factor of ``tp_size``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ColumnParallelLinear", "RowParallelLinear", "tp_rank", "tp_world_size"]


def tp_rank(group=None) -> int:
    """Rank within the tensor-parallel group. Rank in the world if group is None."""
    raise NotImplementedError("Step 3a.")


def tp_world_size(group=None) -> int:
    raise NotImplementedError("Step 3b.")


class ColumnParallelLinear(nn.Module):
    """``y = xA + b`` with ``A`` split along its output dimension.

    ``A = [A_1 | A_2 | ... | A_t]``, so rank ``i`` holds ``A_i`` of shape
    ``(in_features, out_features // t)`` and produces a shard of the output.

    Args:
        in_features: full input dimension. Not sharded.
        out_features: full output dimension. Sharded ``t`` ways; must divide.
        bias: whether to have one.
        gather_output: when True, all-gather the shards so the caller sees the
            full ``(..., out_features)`` output. Costs a collective. Set it False
            when the consumer is a row-parallel layer, which is the whole point of
            the column-then-row ordering, and True at the boundary of a
            tensor-parallel region.
        group: process group. None means the world.

    ⚠ **torch's weight layout is ``(out_features, in_features)``.** So splitting
    the *output* dimension means splitting ``weight`` along dimension **0**, not
    dimension 1. This trips people up because the maths in THEORY.md writes
    ``W`` as ``(in, out)`` and torch stores its transpose. Getting it backwards
    produces a shape error immediately if you are lucky, and a silently wrong
    answer if the dimensions happen to match.

    ⚠ **The bias is sharded.** Each rank holds ``out_features // t`` bias entries,
    matching its output shard. That is different from the row-parallel case and
    the difference is the source of a lot of confusion.

    **Forward structure:**

    1. Apply ``f`` to the input. Identity in the forward pass, and the thing that
       makes the input gradient correct in the backward.
    2. Local ``F.linear`` against this rank's shard.
    3. All-gather if ``gather_output``.

    Step 1 is the one that looks removable and is not. Take it out and the forward
    pass is still correct, the backward pass is silently wrong, and every test that
    only checks outputs still passes.

    Attributes the tests use:
        weight: ``(out_features // t, in_features)``
        bias: ``(out_features // t,)`` or None
        out_features_per_rank: for shape assertions

    Provide a ``load_from_reference(linear)`` method that takes a plain
    ``nn.Linear`` and copies this rank's slice out of it. That is how the tests
    build a sharded layer that is supposed to match a specific reference, and it is
    also how you would load a pretrained checkpoint into a tensor-parallel model.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        group=None,
    ):
        super().__init__()
        raise NotImplementedError(
            "Step 3c: assert divisibility, allocate this rank's weight shard and bias shard."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Step 3d: f, then local matmul, then optional gather.")

    def load_from_reference(self, linear: nn.Linear) -> None:
        """Copy this rank's slice out of an unsharded ``nn.Linear``.

        Assert the shapes line up, and put both in the message. A silent shape
        mismatch here produces a model that runs and computes nonsense.
        """
        raise NotImplementedError("Step 3e.")


class RowParallelLinear(nn.Module):
    """``y = xA + b`` with ``A`` split along its input dimension.

    ``A = [A_1; A_2; ...; A_t]`` stacked, so rank ``i`` holds ``A_i`` of shape
    ``(in_features // t, out_features)`` and consumes a shard of the input. Each
    rank's ``x_i A_i`` is a *partial* result, and the true output is their sum.

    Args:
        in_features: full input dimension. Sharded ``t`` ways; must divide.
        out_features: full output dimension. Not sharded.
        bias: whether to have one.
        input_is_parallel: True when the caller already holds a shard, which is
            the case when the previous layer was column-parallel with
            ``gather_output=False``. False means the caller holds the full tensor
            and this layer has to scatter it first.
        group: process group.

    ⚠ **In torch's ``(out, in)`` layout, splitting the input dimension means
    splitting ``weight`` along dimension 1.** The opposite of the column-parallel
    case.

    ⚠ **The bias is NOT sharded, and it is added after the all-reduce.**

    Every rank holds the full ``(out_features,)`` bias. Adding it before the
    reduce means every rank contributes a copy, and the sum contains ``t`` of
    them. The output is then ``xA + t·b``, which is a perfectly plausible-looking
    tensor of the right shape. At ``t=2`` on a model with small biases you might
    not notice for a while.

    In practice: hold the bias on every rank, and add it only after ``g``.

    **Forward structure:**

    1. Scatter the input if ``input_is_parallel`` is False.
    2. Local ``F.linear`` against this rank's shard, **without bias**.
    3. Apply ``g``: all-reduce, summing the partials.
    4. Add the bias now.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        group=None,
    ):
        super().__init__()
        raise NotImplementedError(
            "Step 3f: assert divisibility, allocate the input-sharded weight and the full bias."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Step 3g: optional scatter, local matmul without bias, g, then bias."
        )

    def load_from_reference(self, linear: nn.Linear) -> None:
        """Copy this rank's slice out of an unsharded ``nn.Linear``.

        The bias is copied whole, not sliced. If you find yourself slicing it, go
        back and reread why row-parallel bias is different.
        """
        raise NotImplementedError("Step 3h.")
