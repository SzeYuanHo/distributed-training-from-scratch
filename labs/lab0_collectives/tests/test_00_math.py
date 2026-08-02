"""Tier 0: the ring algorithm and the cost model, in one process.

No ``torch.distributed`` anywhere in this file. That is enforced by
``test_no_distributed_import`` at the bottom, and it matters: these tests should
run in milliseconds and should never hang, so that when they fail you know the
problem is your arithmetic and not your environment.

Run them alone::

    python tasks.py test LAB=0 K=test_00
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from labs.lab0_collectives.starter import cost_model, ring_numpy

pytestmark = pytest.mark.tier0


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "length,n_ranks",
    [
        (8, 4),  # divides evenly
        (7, 3),  # does not: 3, 2, 2
        (100, 8),  # does not: 13, 13, 13, 13, 12, 12, 12, 12
        (1, 4),  # more ranks than elements; three chunks are empty
        (5, 1),  # degenerate world size
        (0, 3),  # empty buffer, which should not crash
    ],
)
def test_chunk_boundaries_partition_exactly(length, n_ranks):
    bounds = ring_numpy.chunk_boundaries(length, n_ranks)

    assert len(bounds) == n_ranks, f"expected {n_ranks} chunks, got {len(bounds)}"
    assert bounds[0][0] == 0, "first chunk must start at 0"
    assert bounds[-1][1] == length, f"last chunk must end at {length}"

    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0], (
            f"gap or overlap between chunk {i} and {i + 1}: {bounds[i]} then {bounds[i + 1]}"
        )

    sizes = [stop - start for start, stop in bounds]
    assert sum(sizes) == length
    assert all(s >= 0 for s in sizes), f"negative chunk size in {sizes}"
    if length >= n_ranks:
        assert max(sizes) - min(sizes) <= 1, (
            f"chunks are unbalanced: {sizes}. The remainder should be spread one "
            f"element at a time, not dumped on the last rank."
        )


# --------------------------------------------------------------------------- #
# the ring itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("length", [16, 7, 1, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ring_matches_sum(n_ranks, length, dtype):
    """The whole point: every rank ends up with the elementwise total."""
    rng = np.random.default_rng(0)
    buffers = [rng.standard_normal(length).astype(dtype) for _ in range(n_ranks)]
    expected = np.sum(buffers, axis=0)

    result = ring_numpy.ring_allreduce_numpy([b.copy() for b in buffers])

    assert len(result) == n_ranks
    tol = 1e-5 if dtype is np.float32 else 1e-12
    for r, got in enumerate(result):
        assert got.shape == expected.shape, f"rank {r} shape {got.shape} != {expected.shape}"
        np.testing.assert_allclose(
            got,
            expected,
            rtol=tol,
            atol=tol,
            err_msg=f"rank {r} disagrees with the reference sum",
        )


def test_ring_does_not_mutate_inputs():
    """A ring that scribbles on its inputs breaks the moment you reuse a buffer."""
    rng = np.random.default_rng(1)
    buffers = [rng.standard_normal(12).astype(np.float32) for _ in range(4)]
    snapshot = [b.copy() for b in buffers]

    ring_numpy.ring_allreduce_numpy(buffers)

    for r, (before, after) in enumerate(zip(snapshot, buffers)):
        np.testing.assert_array_equal(
            before, after, err_msg=f"input buffer for rank {r} was modified in place"
        )


def test_ring_rejects_mismatched_lengths():
    """Silently broadcasting a length mismatch would be worse than crashing."""
    bad = [np.ones(8, dtype=np.float32), np.ones(4, dtype=np.float32)]
    with pytest.raises(ValueError) as excinfo:
        ring_numpy.ring_allreduce_numpy(bad)
    message = str(excinfo.value)
    assert "8" in message and "4" in message, (
        f"the error should name both lengths so the caller can see which rank is wrong; "
        f"got: {message!r}"
    )


@pytest.mark.parametrize("n_ranks,length", [(4, 16), (3, 7), (8, 100)])
def test_reduce_scatter_phase_alone(n_ranks, length):
    """Phase 1 in isolation, so a bug in phase 2 does not mask a bug in phase 1."""
    rng = np.random.default_rng(2)
    buffers = [rng.standard_normal(length).astype(np.float64) for _ in range(n_ranks)]
    total = np.sum(buffers, axis=0)
    bounds = ring_numpy.chunk_boundaries(length, n_ranks)

    chunks = ring_numpy.reduce_scatter_numpy([b.copy() for b in buffers])

    assert len(chunks) == n_ranks
    for r, (start, stop) in enumerate(bounds):
        np.testing.assert_allclose(
            chunks[r],
            total[start:stop],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"rank {r} should hold the reduced chunk [{start}:{stop}]",
        )


@pytest.mark.parametrize("n_ranks,length", [(4, 16), (3, 7)])
def test_all_gather_phase_alone(n_ranks, length):
    bounds = ring_numpy.chunk_boundaries(length, n_ranks)
    full = np.arange(length, dtype=np.float64)
    chunks = [full[start:stop].copy() for start, stop in bounds]

    gathered = ring_numpy.all_gather_numpy(chunks, length)

    assert len(gathered) == n_ranks
    for r, got in enumerate(gathered):
        np.testing.assert_array_equal(
            got, full, err_msg=f"rank {r} did not reassemble the full buffer"
        )


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #


def test_cost_model_recovers_a_known_line():
    """Fit synthetic data with no noise. If this fails the fit is broken, not noisy."""
    alpha_true, beta_true = 3e-5, 1e-9
    sizes = np.array([2**k for k in range(10, 27)], dtype=np.float64)
    times = alpha_true + beta_true * sizes

    alpha, beta = cost_model.fit_alpha_beta(sizes, times)

    assert beta == pytest.approx(beta_true, rel=0.05), (
        f"beta off: fitted {beta:.3e}, true {beta_true:.3e}"
    )
    assert alpha == pytest.approx(alpha_true, rel=0.30), (
        f"alpha off: fitted {alpha:.3e}, true {alpha_true:.3e}. "
        "A plain unweighted least-squares fit over five orders of magnitude of "
        "message size will do this. See the docstring."
    )


def test_cost_model_survives_noise():
    """Ten percent multiplicative noise should still give the right order of magnitude."""
    alpha_true, beta_true = 5e-5, 4e-10
    rng = np.random.default_rng(7)
    sizes = np.array([2**k for k in range(10, 27)], dtype=np.float64)
    clean = alpha_true + beta_true * sizes
    times = clean * (1 + 0.1 * rng.standard_normal(len(sizes)))

    alpha, beta = cost_model.fit_alpha_beta(sizes, times)

    assert 0 <= alpha < 10 * alpha_true
    assert beta == pytest.approx(beta_true, rel=0.25)


def test_cost_model_never_returns_negative_alpha():
    """Curved data can fit a negative intercept. A negative latency is not a thing."""
    sizes = np.array([2**k for k in range(10, 20)], dtype=np.float64)
    times = 1e-9 * sizes**1.15  # superlinear, so a line fits with alpha < 0

    alpha, _beta = cost_model.fit_alpha_beta(sizes, times)

    assert alpha >= 0, f"alpha must be clamped at zero, got {alpha:.3e}"


@pytest.mark.parametrize("n_ranks", [2, 4, 8])
def test_predicted_ring_volume_matches_theory(n_ranks):
    """The prediction should encode 2(N-1)/N * M, not something close to it."""
    alpha, beta, n_bytes = 0.0, 1e-9, 1 << 20  # alpha=0 isolates the bandwidth term

    predicted = cost_model.predict_allreduce("ring", n_ranks, n_bytes, alpha, beta)
    expected = 2 * (n_ranks - 1) / n_ranks * n_bytes * beta

    assert predicted == pytest.approx(expected, rel=1e-9), (
        f"ring bandwidth term wrong at N={n_ranks}: got {predicted:.6e}, "
        f"expected {expected:.6e} = 2*(N-1)/N * M * beta"
    )


@pytest.mark.parametrize("n_ranks", [2, 4, 8, 16])
def test_predicted_ring_latency_matches_theory(n_ranks):
    alpha, beta, n_bytes = 1e-5, 0.0, 1 << 20  # beta=0 isolates the latency term

    predicted = cost_model.predict_allreduce("ring", n_ranks, n_bytes, alpha, beta)

    assert predicted == pytest.approx(2 * (n_ranks - 1) * alpha, rel=1e-9), (
        f"ring latency term wrong at N={n_ranks}: expected 2(N-1) hops"
    )


@pytest.mark.parametrize("n_ranks", [2, 4, 8, 16])
def test_predicted_recursive_doubling_latency(n_ranks):
    alpha, beta, n_bytes = 1e-5, 0.0, 1 << 20

    predicted = cost_model.predict_allreduce("recursive_doubling", n_ranks, n_bytes, alpha, beta)

    assert predicted == pytest.approx(2 * math.log2(n_ranks) * alpha, rel=1e-9), (
        "recursive doubling should cost 2*log2(N) latency hops"
    )


def test_predicted_recursive_doubling_rejects_non_power_of_two():
    with pytest.raises(ValueError) as excinfo:
        cost_model.predict_allreduce("recursive_doubling", 6, 1024, 1e-5, 1e-9)
    assert "6" in str(excinfo.value), "the error should name the offending world size"


def test_bus_bandwidth_reaches_line_rate_for_a_perfect_ring():
    """The claim from THEORY section 6, as a test.

    With zero latency and a perfect ring, bus bandwidth should equal 1/beta no
    matter how many ranks there are. That is the whole reason the scaled metric
    exists.
    """
    beta, n_bytes = 1e-9, 1 << 24
    for n_ranks in (2, 4, 8, 64):
        seconds = 2 * (n_ranks - 1) / n_ranks * n_bytes * beta
        bus = cost_model.bus_bandwidth(n_bytes, seconds, n_ranks)
        assert bus == pytest.approx(1 / beta, rel=1e-6), (
            f"at N={n_ranks}, bus bandwidth came out {bus:.3e} instead of {1 / beta:.3e}. "
            "Check the scaling factor in THEORY section 6."
        )


def test_algorithm_bandwidth_falls_off_with_rank_count():
    """The complement: the unscaled metric does degrade with N, and should."""
    beta, n_bytes = 1e-9, 1 << 24
    values = []
    for n_ranks in (2, 4, 8, 64):
        seconds = 2 * (n_ranks - 1) / n_ranks * n_bytes * beta
        values.append(cost_model.algorithm_bandwidth(n_bytes, seconds))

    assert values == sorted(values, reverse=True), (
        f"algorithm bandwidth should fall as N grows; got {values}"
    )


def test_crossover_handles_the_degenerate_cases():
    """Infinite and zero crossovers are real answers, not bugs to paper over."""
    # Ring bandwidth much worse than recursive doubling: ring never wins.
    never = cost_model.crossover_bytes(8, alpha=1e-5, beta_ring=1e-8, beta_rd=1e-10)
    assert never == float("inf") or never > 1e12, (
        f"expected the ring never to win here, got crossover at {never}"
    )

    # Ring bandwidth much better: ring always wins.
    always = cost_model.crossover_bytes(8, alpha=1e-5, beta_ring=1e-10, beta_rd=1e-8)
    assert always >= 0, "a negative crossover is not a meaningful answer"


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #


def test_tier0_module_imports_nothing_distributed():
    """Tier 0 means Tier 0: this file and the modules it exercises stay serial.

    The point is not purity. It is that when a Tier-0 test fails you should never
    have to wonder whether the problem is your maths or your environment. Parsing
    the imports beats grepping the text, which would trip over this docstring.
    """
    import ast

    files = [
        Path(__file__),
        Path(ring_numpy.__file__),
        Path(cost_model.__file__),
    ]

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        offenders = [
            m
            for m in imported
            if m.startswith(("torch.distributed", "torch.multiprocessing"))
            or m in ("common.collectives", "common.dist_utils")
        ]
        assert not offenders, (
            f"{path.name} imports {offenders}, which makes it not a Tier-0 module. "
            "Tier 0 runs in one process with no communication at all."
        )
