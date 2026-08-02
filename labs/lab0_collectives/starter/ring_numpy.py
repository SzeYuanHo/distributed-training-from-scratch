"""Step 1, Tier 0: ring all-reduce with no distributed infrastructure at all.

N ranks are simulated as a list of numpy arrays in one process. No ports, no
spawning, no hangs. If the algorithm is wrong here, it will be wrong later with
ten times the debugging cost, so prove it here first.

Run just this step::

    python tasks.py test LAB=0 K=ring_numpy
"""

from __future__ import annotations

import numpy as np

__all__ = ["ring_allreduce_numpy", "chunk_boundaries", "reduce_scatter_numpy", "all_gather_numpy"]


def chunk_boundaries(length: int, n_ranks: int) -> list[tuple[int, int]]:
    """Split ``length`` elements into ``n_ranks`` contiguous chunks.

    Returns ``[(start, stop), ...]`` with ``n_ranks`` entries covering ``[0, length)``
    with no gaps and no overlaps.

    Postconditions the test checks:

    * ``len(result) == n_ranks``, even when ``length < n_ranks`` (some chunks are
      then empty, and that has to work rather than crash)
    * ``result[0][0] == 0`` and ``result[-1][1] == length``
    * ``result[i][1] == result[i+1][0]`` for every i
    * chunk sizes differ by at most 1

    That last one is the whole exercise. THEORY section 4.2 asked you to choose
    between two remainder schemes; this is where the choice becomes code.

    Args:
        length: total number of elements.
        n_ranks: number of chunks.

    Returns:
        List of ``(start, stop)`` half-open index pairs.
    """
    raise NotImplementedError(
        "Step 1a: split `length` into `n_ranks` near-equal contiguous chunks. "
        "See THEORY.md section 4.2 for the two remainder schemes."
    )


def reduce_scatter_numpy(buffers: list[np.ndarray]) -> list[np.ndarray]:
    """Phase 1 of the ring: after this, rank r holds the fully-reduced chunk r.

    ``buffers[r]`` is rank r's full-length input. Do not modify it in place; the
    test reuses it.

    Postcondition: ``result[r]`` equals the elementwise sum, across all ranks, of
    the slice of each rank's buffer covering chunk r. So
    ``result[r] == sum(b[start_r:stop_r] for b in buffers)``.

    Implement it as the ring actually runs, not by calling ``sum``. You are
    simulating N-1 rounds of "send one chunk to your right neighbour, receive one
    chunk from your left neighbour, add it to what you have". Writing the shortcut
    passes the test and teaches you nothing, and step 5's collective-count
    assertion will catch it anyway.

    Args:
        buffers: one array per rank, all the same length and dtype.

    Returns:
        List of N arrays. ``result[r]`` has the length of chunk r.
    """
    raise NotImplementedError(
        "Step 1b: N-1 rounds. In round s, rank r sends one chunk and accumulates "
        "one. Work out which chunk index each rank sends in each round; THEORY.md "
        "section 4 has the blanks for it."
    )


def all_gather_numpy(chunks: list[np.ndarray], length: int) -> list[np.ndarray]:
    """Phase 2 of the ring: every rank ends up with every chunk.

    ``chunks[r]`` is rank r's reduced chunk, as returned by
    :func:`reduce_scatter_numpy`.

    Postcondition: every entry of the result is identical, and equals the
    concatenation of ``chunks`` in rank order.

    Again, simulate the N-1 circulation rounds rather than concatenating.

    Args:
        chunks: one reduced chunk per rank.
        length: total length of the reassembled buffer.

    Returns:
        List of N identical full-length arrays.
    """
    raise NotImplementedError("Step 1c: N-1 more rounds, circulating chunks instead of summing.")


def ring_allreduce_numpy(buffers: list[np.ndarray]) -> list[np.ndarray]:
    """Full ring all-reduce: reduce-scatter, then all-gather.

    Postcondition: every returned array equals ``sum(buffers)`` elementwise, and
    they are all equal to each other.

    Cases the test covers, and each one has caught a real bug:

    * ``n_ranks in {2, 3, 4, 8}``, including a non-power-of-two
    * ``length`` that does and does not divide by ``n_ranks`` (7 on 3 ranks)
    * ``length < n_ranks`` (3 elements on 4 ranks), where some chunks are empty
    * ``n_ranks == 1``, which should be a no-op returning a copy
    * float32 and float64

    Args:
        buffers: one input array per simulated rank.

    Returns:
        List of N arrays, each equal to the elementwise sum of all inputs.

    Raises:
        ValueError: if the buffers differ in length or dtype. The test checks the
            error message mentions both lengths, because a silent broadcast here
            would be worse than a crash.
    """
    raise NotImplementedError("Step 1: compose reduce_scatter_numpy and all_gather_numpy.")
