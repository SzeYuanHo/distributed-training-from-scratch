"""Step 8: fit the alpha-beta model to your hardware, then predict with it.

This is the part of the lab that transfers. Anyone can call ``dist.all_reduce``.
Being able to say "that will take 4 ms and here is why" before you run it is the
skill Lab 4 is built on.

No distributed code here. Pure numerics on measurements someone else collected,
which means these functions are Tier 0 and test in milliseconds.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fit_alpha_beta",
    "predict_allreduce",
    "algorithm_bandwidth",
    "bus_bandwidth",
    "crossover_bytes",
]


def fit_alpha_beta(sizes_bytes: np.ndarray, times_s: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``T = alpha + beta * m``.

    Args:
        sizes_bytes: message sizes actually measured.
        times_s: corresponding wall times, one per size.

    Returns:
        ``(alpha, beta)``: seconds, and seconds per byte.

    Two things to think about before you write the four lines this takes.

    **The fit is dominated by the large sizes.** Times span four or five orders of
    magnitude across a 2^10 to 2^26 sweep, so an ordinary least-squares fit on the
    raw values fits the big messages well and ignores the small ones entirely,
    which means alpha comes out as noise. Alpha is the number you care about for
    predicting small-message behaviour. Decide what to do about it, and put your
    reasoning in a comment: weighting, fitting in log space, or fitting alpha
    separately from the smallest few points are all defensible.

    **A negative alpha means something.** If the fit returns alpha < 0 the model is
    wrong for this data, usually because a protocol switch or a cache effect put a
    kink in the curve. Clamp it to zero and print a warning rather than returning
    a physically impossible latency.

    Raises:
        ValueError: if fewer than two points, or lengths differ.
    """
    raise NotImplementedError(
        "Step 8a: fit a line. The interesting decision is what you weight it by."
    )


def predict_allreduce(
    algorithm: str, n_ranks: int, n_bytes: int, alpha: float, beta: float
) -> float:
    """Predicted seconds for one all-reduce, from the model.

    Args:
        algorithm: ``"naive"``, ``"ring"``, or ``"recursive_doubling"``.
        n_ranks: world size.
        n_bytes: buffer size.
        alpha: fitted latency, seconds.
        beta: fitted inverse bandwidth, seconds per byte.

    Returns:
        Predicted seconds.

    The three expressions are the ones you filled in in THEORY sections 2, 3 and
    4. Transcribe them; do not re-derive them here.

    ``recursive_doubling`` is only defined for power-of-two ``n_ranks``. Raise
    ``ValueError`` otherwise, matching what
    ``allreduce_algos.all_reduce_recursive_doubling`` does, so the predicted and
    measured paths agree about what is legal.

    Raises:
        ValueError: unknown algorithm, or non-power-of-two N for recursive doubling.
    """
    raise NotImplementedError("Step 8b: transcribe the three cost expressions from THEORY.md.")


def algorithm_bandwidth(n_bytes: int, seconds: float) -> float:
    """Bytes per second, the naive way: buffer size over elapsed time.

    This is the number that answers "how fast did my data get reduced". It falls
    as N grows even on perfect hardware, because a larger ring takes more hops.

    Returns:
        Bytes per second.
    """
    raise NotImplementedError("Step 8c: one line.")


def bus_bandwidth(n_bytes: int, seconds: float, n_ranks: int) -> float:
    """Derived all-reduce bus bandwidth as ``nccl-tests`` reports it.

    Scaled so that a perfect flat-ring implementation reports roughly the
    hardware's link bandwidth regardless of N. It is a correction factor, not
    literal wire rate for every topology or algorithm. THEORY section 6 derives
    it.

    Returns:
        Bytes per second.
    """
    raise NotImplementedError("Step 8d: one line, once you have the factor from THEORY section 6.")


def crossover_bytes(
    n_ranks: int, alpha: float, beta_ring: float, beta_rd: float | None = None
) -> float:
    """Message size where ring overtakes recursive doubling.

    Args:
        n_ranks: world size, a power of two.
        alpha: fitted latency.
        beta_ring: fitted inverse bandwidth for the ring implementation.
        beta_rd: same for recursive doubling. When None, assume both achieve the
            same effective bandwidth, which makes the comparison degenerate.
            THEORY section 5 asks you to work out what that degeneracy means and
            what the honest model is.

    Returns:
        Crossover size in bytes. Return ``float("inf")`` when the ring never wins,
        and ``0.0`` when it always does. Both are real outcomes at some parameter
        values, and returning a negative number instead is how you find out later
        that you did not handle them.
    """
    raise NotImplementedError("Step 8e: set the two cost expressions equal and solve for M.")
