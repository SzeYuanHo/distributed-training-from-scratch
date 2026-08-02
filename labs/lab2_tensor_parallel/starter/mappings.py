"""Step 2: the four autograd Functions that make in-model collectives legal.

**Read this before writing anything.**

A collective inside a model is not a plain function call. Autograd has no idea what
``dist.all_reduce`` did, so if you call it on a tracked tensor the graph records
nothing and the gradient that flows back is the gradient of the *pre-reduce*
tensor. That is not an error and it is not the right answer, which is the worst
combination available.

So every collective inside the model goes through a ``torch.autograd.Function``
with a hand-written ``backward``.

The four you need, with the forward column filled in and the backward column
blank. Deriving the backward is the exercise, and THEORY sections 1 and 2 walk you
to it:

===================================  =============  ==================
Function                             forward        backward
===================================  =============  ==================
``CopyToTensorParallelRegion`` (f)   identity       ``____________``
``ReduceFromTensorParallelRegion``   all-reduce     ``____________``
``ScatterToTensorParallelRegion``    split          ``____________``
``GatherFromTensorParallelRegion``   all-gather     ``____________``
===================================  =============  ==================

Two of those backwards are the other two forwards. That is what "conjugate" means
here, and noticing it is most of the insight.

**The consequence of getting f and g backwards:** the forward output is still
correct, because both orderings compute the same value. Only the gradients differ.
The loss falls. The model trains. It is wrong. ``tests/test_negative.py`` proves
the suite catches it, and it is worth running that test and reading its output
before you start.
"""

from __future__ import annotations

import torch

__all__ = [
    "CopyToTensorParallelRegion",
    "ReduceFromTensorParallelRegion",
    "ScatterToTensorParallelRegion",
    "GatherFromTensorParallelRegion",
    "copy_to_tp_region",
    "reduce_from_tp_region",
    "scatter_to_tp_region",
    "gather_from_tp_region",
    "split_along_last_dim",
    "gather_along_last_dim",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def split_along_last_dim(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Return this rank's contiguous slice of the last dimension.

    No communication: every rank already has the whole tensor and simply keeps its
    own piece. Asserts the last dimension divides by the group size, with both
    numbers in the message.

    ``.contiguous()`` on the result matters. A view into a larger tensor keeps the
    whole thing alive, and some collectives require contiguous input and will
    either copy silently or raise depending on the backend.
    """
    raise NotImplementedError("Step 2a: chunk the last dim, keep index rank().")


def gather_along_last_dim(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """All-gather every rank's slice and concatenate along the last dimension.

    Inverse of :func:`split_along_last_dim`. Uses a real collective.
    """
    raise NotImplementedError("Step 2b: all_gather, then cat on the last dim.")


# --------------------------------------------------------------------------- #
# the conjugate pair
# --------------------------------------------------------------------------- #


class CopyToTensorParallelRegion(torch.autograd.Function):
    """``f``. Forward: identity. Backward: derive it.

    Sits at the *entry* to a tensor-parallel region, in front of a column-parallel
    layer. The forward pass does nothing at all, which makes it look pointless
    until you look at the backward.

    THEORY section 1 walked you through it: each rank's column-parallel weight
    shard produces a partial contribution to ``∂L/∂X``, and the true gradient is
    the sum over ranks. Something has to perform that sum, and this is the only
    place in the graph that can.

    A `Function` whose forward is the identity and whose backward is not is a
    strange object the first time you meet one. It is worth sitting with: it says
    "this tensor is replicated going forwards and needs summing coming back", which
    is a statement about the *parallelism layout*, not about the arithmetic.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        raise NotImplementedError("Step 2c: what does f do in the forward pass?")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError(
            "Step 2d: THEORY section 1. Return a tuple matching forward's inputs, "
            "with None for the non-tensor ones."
        )


class ReduceFromTensorParallelRegion(torch.autograd.Function):
    """``g``. Forward: all-reduce. Backward: derive it.

    Sits at the *exit* of a tensor-parallel region, after a row-parallel layer.
    The forward all-reduce turns the per-rank partial sums into the true output.

    The backward is the surprising half. THEORY section 2 has it. The one-line
    intuition: the forward was a sum over ranks, and the derivative of a sum with
    respect to each of its terms is 1, so the incoming gradient passes to every
    rank unchanged.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        raise NotImplementedError("Step 2e.")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Step 2f: THEORY section 2.")


# --------------------------------------------------------------------------- #
# the sequence-parallel pair
# --------------------------------------------------------------------------- #


class ScatterToTensorParallelRegion(torch.autograd.Function):
    """Forward: split, keeping this rank's slice. Backward: derive it.

    Needed for the sequence-parallel stretch goal, and useful now for
    ``ColumnParallelLinear(input_is_parallel=False)`` when you want to shard an
    input rather than replicate it.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        raise NotImplementedError("Step 2g.")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Step 2h: the inverse of a split.")


class GatherFromTensorParallelRegion(torch.autograd.Function):
    """Forward: all-gather. Backward: derive it.

    Used by ``ColumnParallelLinear(gather_output=True)`` when the consumer needs
    the full unsharded tensor.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group=None) -> torch.Tensor:
        raise NotImplementedError("Step 2i.")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Step 2j: the inverse of an all-gather.")


# --------------------------------------------------------------------------- #
# thin functional wrappers, so call sites read as maths
# --------------------------------------------------------------------------- #


def copy_to_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """``f(x)``."""
    return CopyToTensorParallelRegion.apply(tensor, group)


def reduce_from_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """``g(x)``."""
    return ReduceFromTensorParallelRegion.apply(tensor, group)


def scatter_to_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    return ScatterToTensorParallelRegion.apply(tensor, group)


def gather_from_tp_region(tensor: torch.Tensor, group=None) -> torch.Tensor:
    return GatherFromTensorParallelRegion.apply(tensor, group)
