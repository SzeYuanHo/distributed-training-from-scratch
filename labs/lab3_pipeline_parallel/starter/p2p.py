"""Step 2: move activations between stages without deadlocking.

Short file, and the place this lab goes wrong. Everything below either avoids a
deadlock or avoids a shape mismatch, and both failures present as a hang rather
than an error.
"""

from __future__ import annotations

import torch

__all__ = ["send_activation", "recv_activation", "exchange", "send_recv_pairs"]


def send_activation(tensor: torch.Tensor, dst: int, group=None) -> None:
    """Send an activation to the next stage.

    Use ``common.collectives`` so the counter sees it.

    Blocking ``send`` is fine here *if* the peer is already receiving. In a
    pipeline it usually is not, which is what :func:`exchange` is for.
    """
    raise NotImplementedError("Step 2a.")


def recv_activation(
    shape: tuple, dtype: torch.dtype, src: int, group=None, requires_grad: bool = True
) -> torch.Tensor:
    """Receive an activation from the previous stage.

    Args:
        shape: must match what the sender sent. There is no negotiation.
        dtype: likewise.
        src: the sending rank.
        requires_grad: mark the received tensor as requiring grad.

    ⚠ **``requires_grad=True`` is what makes the backward pass work across a stage
    boundary.**

    The autograd graph does not cross a process. Stage 1 receives a plain tensor
    with no history: as far as its autograd is concerned, that tensor appeared
    from nowhere. Marking it as requiring grad makes it a leaf, so the backward
    pass accumulates a gradient into it, and *that gradient* is what you send back
    to stage 0 to continue the chain.

    Skip this and stage 1's backward pass stops at its own input, stage 0 never
    receives a gradient, and its parameters never update. The loss still falls,
    because the later stages are learning. Only the early layers are frozen, and
    you find out when the model plateaus at a loss that seems slightly too high.

    Returns:
        A newly allocated tensor holding the received data.
    """
    raise NotImplementedError("Step 2b.")


def exchange(
    send_tensor: torch.Tensor | None,
    send_to: int | None,
    recv_shape: tuple | None,
    recv_dtype: torch.dtype | None,
    recv_from: int | None,
    group=None,
) -> torch.Tensor | None:
    """Send and receive in one call, without deadlocking.

    Every rank calls this with its own arguments. A stage at the head of the
    pipeline passes ``recv_from=None``; one at the tail passes ``send_to=None``.

    **Use ``dist.batch_isend_irecv``.** Build a list of ``P2POp``s and hand the
    whole thing to the backend at once. The backend can then order the transfers
    however it likes, and the symmetric-send deadlock becomes impossible by
    construction rather than by convention.

    ``common.collectives`` exports ``batch_isend_irecv`` and ``P2POp``, and it
    records the traffic so the counter still sees it.

    The two alternatives, for the record and for the writeup:

    * **Odd/even ordering.** Even ranks send then receive, odd ranks receive then
      send. Correct, free, and it needs care when the world size is odd or when
      the communication pattern is not a simple chain.
    * **isend/irecv with explicit waits.** Post both operations, then wait on
      both. Also correct. More code, and the ordering of the waits matters in ways
      that are easy to get subtly wrong.

    Returns:
        The received tensor, or None when ``recv_from`` is None.
    """
    raise NotImplementedError("Step 2c: build the P2POp list and issue it as one batch.")


def send_recv_pairs(stage_index: int, n_stages: int) -> tuple[int | None, int | None]:
    """``(previous_stage, next_stage)`` for this position, with None at the ends.

    Trivial, and worth having in one place so the boundary conditions live
    somewhere they can be tested rather than scattered through the schedules.
    """
    raise NotImplementedError("Step 2d.")
