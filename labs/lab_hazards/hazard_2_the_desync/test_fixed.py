"""Passes once every rank holds the same model."""

from __future__ import annotations

import pytest

from common import dist_utils, testing  # noqa: E402

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(120)]


def _w_check_initial_weights():
    """Check before any training. The divergence is there from step zero."""
    from . import broken

    model = broken.build_model()
    for name, param in model.named_parameters():
        testing.assert_close_across_ranks(param.data, rtol=1e-7, atol=1e-9, name=f"initial {name}")
    return "identical"


def test_ranks_start_from_identical_weights():
    """The root cause, tested directly rather than through its consequences."""
    results = dist_utils.run_distributed(
        _w_check_initial_weights, world_size=4, backend="gloo", timeout_s=60
    )
    assert all(v == "identical" for v in results.values())


def test_ranks_end_with_identical_weights():
    """The consequence. Should follow from the cause being fixed."""
    from . import broken

    results = dist_utils.run_distributed(broken.worker, world_size=4, backend="gloo", timeout_s=90)

    checksums = [info["weight_checksum"] for info in results.values()]
    spread = max(checksums) - min(checksums)

    assert spread < 1e-5, (
        f"after {broken.STEPS} steps the ranks' first-layer weight sums differ by "
        f"{spread:.6e}.\n"
        f"  Per-rank sums: {dict(sorted((r, round(i['weight_checksum'], 6)) for r, i in results.items()))}\n"
        "  Gradients are being averaged correctly, so the ranks must have started "
        "from different weights."
    )


def test_the_loss_still_falls():
    """The fix must not break training.

    Worth checking, because "make every rank identical" has a degenerate solution
    where nothing updates at all.
    """
    from . import broken

    results = dist_utils.run_distributed(broken.worker, world_size=4, backend="gloo", timeout_s=90)
    losses = results[0]["losses"]

    assert losses[-1] < losses[0] * 0.5, (
        f"the loss went from {losses[0]:.4f} to {losses[-1]:.4f}, which is not "
        "convergence. The fix should keep training working."
    )


def test_per_rank_randomness_is_still_available():
    """The fix should not be "never use rank_offset".

    Data sampling and dropout genuinely need to differ per rank. The lesson is
    that initialisation and sampling want opposite things from the RNG, not that
    one of them is always wrong.
    """
    results = dist_utils.run_distributed(
        _w_rank_offset_still_differs, world_size=4, backend="gloo", timeout_s=60
    )
    values = [info for info in results.values()]
    assert len(set(values)) == len(values), (
        f"seed_everything(rank_offset=True) gave every rank the same value {values}. "
        "That would break data sharding. The point is to use it for sampling and "
        "not for initialisation."
    )


def _w_rank_offset_still_differs():
    import torch

    dist_utils.seed_everything(1337, rank_offset=True)
    return float(torch.randn(1))
