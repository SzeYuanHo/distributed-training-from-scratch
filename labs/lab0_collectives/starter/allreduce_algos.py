"""Steps 5 and 6: the same collective, three ways.

All three produce identical results. They differ only in how the cost is spent,
and that difference is the entire point of the lab. By the end you should be able
to look at a message size and a rank count and say which one a good library would
pick.

Same constraint as ``p2p_collectives``: point-to-point only, through
``common.collectives`` so the counter sees it.
"""

from __future__ import annotations

import torch

__all__ = ["all_reduce_naive", "all_reduce_ring", "all_reduce_recursive_doubling"]


def all_reduce_naive(tensor: torch.Tensor) -> torch.Tensor:
    """Gather everything to rank 0, sum it, broadcast the answer back.

    In place. Postcondition: every rank's tensor equals the elementwise sum of all
    N inputs.

    Build it from ``reduce_to_root`` and ``broadcast_linear``. Two lines.

    Its cost, which you derived in THEORY section 2: rank 0 receives (N-1)·M bytes
    and then sends (N-1)·M bytes, and it does both serially. Every other rank sits
    idle for almost the whole operation. This is the baseline the other two beat.

    Args:
        tensor: per-rank contribution.

    Returns:
        The same tensor object, modified in place.
    """
    raise NotImplementedError("Step 5a: reduce_to_root, then broadcast.")


def all_reduce_ring(tensor: torch.Tensor) -> torch.Tensor:
    """Ring all-reduce: reduce-scatter, then all-gather.

    In place. Postcondition: matches ``dist.all_reduce`` to within floating-point
    reordering.

    Two lines if steps 4a and 4b are correct.

    **The assertion that matters:** the test checks that this issues exactly
    ``2(N-1)`` sends per rank, and nothing else. Not `2N`, not `3(N-1)`. If your
    count is high, you have an extra pass somewhere, and the extra pass is moving
    real bytes even though the answer comes out right. That is the class of bug
    numeric tests never catch and this repo exists to teach you to look for.

    Args:
        tensor: per-rank contribution, same shape and dtype on every rank.

    Returns:
        The same tensor object, modified in place.
    """
    raise NotImplementedError("Step 5b: reduce_scatter_ring, then all_gather_ring, then copy back.")


def all_reduce_recursive_doubling(tensor: torch.Tensor) -> torch.Tensor:
    """Recursive halving-doubling all-reduce. Requires a power-of-two world size.

    In place. Postcondition: matches ``dist.all_reduce``.

    The structure, from THEORY section 3: in round k, rank r exchanges with the
    rank whose id differs from r in bit k. Reduce-scatter phase halves the
    exchanged data each round; all-gather phase reverses it.

    Latency: ``2·log2(N)`` rounds against the ring's ``2(N-1)``. At N=8 that is 6
    rounds instead of 14, and at N=64 it is 12 instead of 126. Bytes moved are the
    same. That is why this wins on small messages and loses on large ones.

    **Raise, do not silently misbehave.** If ``world_size`` is not a power of two,
    raise ``ValueError`` with the world size in the message and a note about what
    to use instead. The test checks that at N=3 you raise rather than return a
    wrong answer, because a wrong answer here would surface three labs later as an
    inexplicable gradient mismatch.

    A simpler variant exists, where every rank exchanges its *whole* buffer each
    round rather than halving. It is correct, it takes log2(N) rounds instead of
    2·log2(N), and it moves ``log2(N)·M`` bytes per rank instead of
    ``2(N-1)/N·M``. Work out at which N the simple variant stops being a good idea,
    and write the answer in your writeup. You may implement either; the test
    checks correctness and the byte count separately, and tells you which variant
    it thinks you wrote.

    Args:
        tensor: per-rank contribution.

    Returns:
        The same tensor object, modified in place.

    Raises:
        ValueError: if the world size is not a power of two.
    """
    raise NotImplementedError(
        "Step 6: log2(N) rounds of pairwise exchange with the rank differing in bit k."
    )
