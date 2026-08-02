"""Steps 1 and 2: flatten gradients into one buffer, and decide which go together.

A bucket exists for one reason: a single all-reduce of 4 MB costs one latency, and
a thousand all-reduces of 4 KB each cost a thousand. The bytes are identical. The
time is not.

Two pieces here. ``GradientBucket`` is the mechanism. ``assign_buckets`` is the
policy. Getting the policy right is what makes overlap possible in step 6, so do
not treat it as bookkeeping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GradientBucket", "assign_buckets", "param_bytes"]


def param_bytes(param: nn.Parameter) -> int:
    """Bytes a gradient for this parameter occupies. One line."""
    raise NotImplementedError("Step 1a.")


class GradientBucket:
    """A group of parameters whose gradients are all-reduced as one flat buffer.

    Lifecycle::

        bucket = GradientBucket(params, index=0)
        bucket.mark_ready(param)        # from an autograd hook, once per param
        if bucket.is_full():
            bucket.flush(async_op=True) # one all-reduce over the flat buffer
        ...
        bucket.wait()                   # before optimizer.step()

    ⚠ **Do not build the flat buffer with ``torch.cat``.** It allocates a second
    copy of every gradient in the bucket, so peak memory during the flush is twice
    what it should be, and copying results back with slice assignment silently
    detaches any view someone else holds on ``.grad``. Use
    ``torch._utils._flatten_dense_tensors`` and ``_unflatten_dense_tensors``, or
    allocate one contiguous buffer at construction and make each parameter's
    ``.grad`` a view into it. The second option is what real DDP does and it is
    strictly better: gradients are written directly into the buffer by autograd,
    so the flatten step costs nothing at all.

    Attributes you will need, and the test inspects some of them:
        params: the parameters in this bucket, in bucket order.
        index: position in the bucket list. Bucket 0 flushes first.
        ready: set of parameters whose gradients have arrived.
        handle: the async work handle after a flush, None before.
    """

    def __init__(self, params: list[nn.Parameter], index: int = 0):
        raise NotImplementedError(
            "Step 1b: store the params, allocate the flat buffer, track readiness."
        )

    def total_bytes(self) -> int:
        """Sum of gradient bytes across this bucket's parameters."""
        raise NotImplementedError("Step 1c.")

    def mark_ready(self, param: nn.Parameter) -> None:
        """Record that ``param.grad`` now exists.

        Called from an autograd hook. Must be idempotent: a shared parameter can
        fire its hook more than once, and counting it twice makes the bucket look
        full before it is.
        """
        raise NotImplementedError("Step 1d.")

    def is_full(self) -> bool:
        """True once every parameter in this bucket has a gradient."""
        raise NotImplementedError("Step 1e.")

    def flush(self, async_op: bool = False, world_size: int | None = None):
        """All-reduce the bucket and divide by the world size.

        Flatten the gradients, all-reduce through ``common.collectives.all_reduce``
        (so the CommCounter sees it), divide by the world size, unflatten back.

        **The division is not optional and it is not cosmetic.** ``all_reduce``
        with ``ReduceOp.SUM`` gives you N times the mean gradient. Skip the divide
        and your effective learning rate is N times too large: the loss still
        falls, everything looks fine, and the loss curve diverges from the
        reference around step 5 with no error anywhere.

        With ``async_op=True``, return immediately and store the handle. The
        caller is then responsible for ``wait()``, and reading ``.grad`` before
        that gives a partially reduced tensor. See ``tests/test_negative.py``.

        Returns:
            The async handle when ``async_op=True``, otherwise None.
        """
        raise NotImplementedError("Step 1f: flatten, all-reduce, divide, unflatten.")

    def wait(self) -> None:
        """Block until this bucket's collective has completed. Idempotent."""
        raise NotImplementedError("Step 1g.")

    def reset(self) -> None:
        """Clear readiness and the handle, ready for the next iteration."""
        raise NotImplementedError("Step 1h.")


def assign_buckets(
    params: list[nn.Parameter], bucket_bytes: int = 1 << 20, reverse: bool = True
) -> list[GradientBucket]:
    """Group parameters into buckets by size, in gradient-readiness order.

    Args:
        params: as returned by ``model.parameters()``, in registration order.
        bucket_bytes: soft cap. A parameter larger than the cap gets a bucket to
            itself rather than being split.
        reverse: iterate in reverse registration order. THEORY section 3 is the
            argument for why. ``reverse=False`` exists so you can measure the
            difference rather than take the argument on faith.

    Returns:
        Buckets in flush order. ``result[0]`` should be the one that fills first.

    Details the tests check:

    * Every parameter with ``requires_grad`` appears in exactly one bucket.
    * Parameters with ``requires_grad=False`` appear in none. They never get a
      gradient, so a bucket containing one never fills, and step 8's hang arrives
      early and for a boring reason.
    * Bucket sizes stay under the cap except when a single parameter exceeds it.
    * With ``reverse=True``, ``result[0]`` contains the *last* registered
      parameters.

    A tied weight, which this repo's reference model has between the token
    embedding and the LM head, appears twice in ``named_parameters()`` and is one
    object. Putting it in two buckets means all-reducing it twice. Deduplicate by
    identity, not by name.
    """
    raise NotImplementedError(
        "Step 2: walk the parameters in readiness order, filling buckets to the cap."
    )
