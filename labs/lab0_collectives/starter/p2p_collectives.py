"""Steps 3, 4 and 7: collectives built from point-to-point messages only.

**The constraint.** Every function here may call only these, from
``common.collectives``::

    send, recv, isend, irecv, barrier

Calling ``all_reduce``, ``all_gather``, ``broadcast``, or any other collective
defeats the exercise. The tests enforce this by counting what the ``CommCounter``
saw: if a collective op shows up in a function that is supposed to be built from
sends, the assertion fails and names it.

Using the wrappers rather than ``torch.distributed`` directly is what makes that
counting possible, and it costs you nothing: the signatures match.

**Every rank runs the same code.** That is the mental shift this file exists to
force. You are not writing "the coordinator" and "the workers"; you are writing
one function that behaves differently depending on ``rank()``, and every rank
must issue the same number of operations in a compatible order or the job hangs.
"""

from __future__ import annotations

import torch

__all__ = [
    "broadcast_linear",
    "broadcast_tree",
    "reduce_to_root",
    "reduce_scatter_ring",
    "all_gather_ring",
    "all_to_all_p2p",
]


def broadcast_linear(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Rank ``src``'s tensor overwrites every other rank's, one message at a time.

    In place: modify and return ``tensor``.

    Postcondition: after this returns on every rank, all N tensors are equal to
    the value ``src`` held on entry.

    Message count, which the test asserts: rank ``src`` issues N-1 sends. Every
    other rank issues exactly 1 recv.

    Args:
        tensor: same shape and dtype on every rank. Contents matter only on ``src``.
        src: the rank whose value wins.

    Returns:
        The same tensor object, modified in place.
    """
    raise NotImplementedError(
        "Step 3a: if you are src, send to everyone else. Otherwise receive from src."
    )


def broadcast_tree(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Same postcondition as :func:`broadcast_linear`, in log2(N) rounds instead of N-1.

    The shape of the algorithm: in each round, every rank that already has the
    value sends it to exactly one rank that does not. The number of ranks holding
    the value doubles each round.

    Getting the peer arithmetic right is the work. Two things make it fiddly:

    * ``src`` is not necessarily 0. The clean trick is to relabel ranks so ``src``
      becomes 0, do the arithmetic in that space, and relabel back.
    * N is not necessarily a power of two. Ranks that would receive from a
      non-existent peer must simply not participate in that round, and they must
      not block waiting for a message nobody will send.

    Message counts the test asserts, at N=8 with src=0: total sends across all
    ranks is 7 (each rank except the source receives exactly once), and no single
    rank sends more than 3 times. Compare that to linear's 7 sends all from one
    rank. Same total, very different critical path.

    Args:
        tensor: same shape and dtype on every rank.
        src: the rank whose value wins.

    Returns:
        The same tensor object, modified in place.
    """
    raise NotImplementedError(
        "Step 3b: doubling rounds. Work out, for rank r in round k, whether it "
        "sends, receives, or sits out, and to or from whom."
    )


def reduce_to_root(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    """Sum every rank's tensor onto rank ``dst``. Naive version.

    Postcondition: on rank ``dst``, the tensor holds the elementwise sum of all N
    inputs. On every other rank the contents are unspecified, and the test does
    not look at them. Say so in your head: "unspecified" is a real part of the
    contract for a reduce, and it is why an all-reduce is not just a reduce.

    Args:
        tensor: per-rank contribution, same shape and dtype everywhere.
        dst: the rank that ends up with the sum.

    Returns:
        The same tensor object.
    """
    raise NotImplementedError("Step 3c: everyone sends to dst; dst receives N-1 times and sums.")


def reduce_scatter_ring(tensor: torch.Tensor) -> torch.Tensor:
    """Ring reduce-scatter. Rank r returns the reduced slice belonging to rank r.

    Same algorithm as ``ring_numpy.reduce_scatter_numpy``, now with real processes
    and real messages instead of a list of arrays. If step 1 is solid this is
    mostly translation, and if it is not, this is where you find out.

    Postcondition: concatenating the N returned tensors in rank order gives
    exactly what ``dist.all_reduce`` would have produced.

    Message count the test asserts: exactly N-1 sends and N-1 receives per rank.
    Not 2(N-1); that is the full all-reduce.

    Two things that bite here and did not in step 1:

    * A receive needs a buffer of the right shape allocated in advance. There is
      no negotiation. Allocate before you enter the loop.
    * Blocking ``send`` then blocking ``recv``, symmetrically on every rank,
      deadlocks once the message exceeds the transport's buffer. You have three
      ways out and `test_symmetric_send_deadlocks` demonstrates the failure.
      Choose one and note in a comment which and why.

    Args:
        tensor: the full-length buffer, identical shape on every rank.

    Returns:
        A new tensor holding this rank's reduced chunk. Length may differ by one
        element between ranks when the buffer does not divide evenly.
    """
    raise NotImplementedError(
        "Step 4a: N-1 rounds of send-one-chunk, receive-one-chunk, accumulate."
    )


def all_gather_ring(chunk: torch.Tensor, full_length: int) -> torch.Tensor:
    """Ring all-gather. Every rank ends up with every rank's chunk, concatenated.

    Postcondition: every rank returns an identical tensor of length
    ``full_length``, being the concatenation of all ranks' chunks in rank order.

    Message count: N-1 sends and N-1 receives per rank.

    The subtlety this function has and ``reduce_scatter_ring`` does not: chunks
    can be different lengths when the buffer does not divide evenly, so a rank
    receiving chunk ``j`` needs to know how long chunk ``j`` is before it can
    allocate the receive buffer. Every rank can compute that locally from
    ``full_length`` and the world size. Do that rather than sending a size first;
    metadata exchange is Lab 3's problem and you do not need it here.

    Args:
        chunk: this rank's piece.
        full_length: total length of the reassembled buffer.

    Returns:
        A new tensor of length ``full_length``, identical on every rank.
    """
    raise NotImplementedError("Step 4b: N-1 rounds circulating chunks around the ring.")


def all_to_all_p2p(tensor: torch.Tensor) -> torch.Tensor:
    """All-to-all: a distributed transpose.

    ``tensor`` is split into N equal chunks. Rank i's chunk j goes to rank j's
    slot i. After this, rank r holds ``[chunk_r_from_rank_0, chunk_r_from_rank_1,
    ...]`` concatenated in source-rank order.

    Concretely, at N=3 with each rank holding ``[a, b, c]``:

    =========  ==========  ==========
    rank       before      after
    =========  ==========  ==========
    0          a0 b0 c0    a0 a1 a2
    1          a1 b1 c1    b0 b1 b2
    2          a2 b2 c2    c0 c1 c2
    =========  ==========  ==========

    This is the collective behind expert parallelism in mixture-of-experts models
    and behind sequence-parallel attention. It is also the most expensive of the
    standard collectives: every rank talks to every other rank, so the message
    count is quadratic in N even though the bytes per rank are not.

    Postcondition: matches ``dist.all_to_all_single`` exactly.

    Requires ``tensor.numel() % world_size == 0``. Assert it, and put both numbers
    in the message.

    Args:
        tensor: length divisible by the world size.

    Returns:
        A new tensor of the same length, permuted as described.
    """
    raise NotImplementedError(
        "Step 7: N-1 exchanges plus a local copy of your own chunk. Watch the "
        "ordering: naive symmetric exchange deadlocks."
    )
