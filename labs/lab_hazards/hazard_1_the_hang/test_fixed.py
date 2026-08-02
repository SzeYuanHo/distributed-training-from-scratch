"""Passes once ``broken.py`` no longer hangs.

The timeout is the assertion. A deadlock has to fail loudly here, or this test is
worth nothing.
"""

from __future__ import annotations

import pytest

from common import comm_counter, dist_utils  # noqa: E402

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(120)]


def _w_counts_match():
    """Every rank must issue the same number of collectives.

    That is the invariant hazard 1 violates, and checking it directly is more
    useful than checking that the program finished: a fix that happens to work by
    timing luck fails here.
    """
    import torch

    from . import broken

    torch.manual_seed(1337)
    model = torch.nn.Linear(16, 4)
    dist_utils.sync_module_params(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with comm_counter.CommCounter() as counter:
        for step in range(6):
            broken.train_step(model, optimizer, step)

    return {"rank": dist_utils.rank(), "counts": counter.counts()}


@pytest.mark.timeout(90)
def test_the_job_completes():
    """The direct check: four ranks, ten steps, no hang."""
    from . import broken

    results = dist_utils.run_distributed(
        broken.worker,
        world_size=4,
        backend="gloo",
        timeout_s=40,
        join_timeout_s=70,
    )
    assert all(v == "completed" for v in results.values()), f"not every rank finished: {results}"


@pytest.mark.timeout(90)
def test_every_rank_issues_the_same_collectives():
    """The invariant, not just the outcome.

    A fix that makes the program finish but still has ranks issuing different
    collective counts is fixing the symptom. This catches that.
    """
    results = dist_utils.run_distributed(
        _w_counts_match,
        world_size=4,
        backend="gloo",
        timeout_s=40,
        join_timeout_s=70,
    )

    reference = results[0]["counts"]
    for rank, info in results.items():
        assert info["counts"] == reference, (
            f"rank {rank} issued {info['counts']} collectives, rank 0 issued "
            f"{reference}.\n"
            "  Every rank has to call the same collectives in the same order. "
            "A difference here is a hang waiting for a slightly different program.\n"
            "  The usual cause is a collective inside `if rank == 0:`."
        )


@pytest.mark.timeout(90)
def test_the_logging_still_works():
    """The fix should keep the logging, not delete it.

    Removing the print makes the test pass and misses the point. The lesson is
    that logging *and* collectives can coexist, as long as every rank reaches the
    collective.
    """
    import inspect

    from . import broken

    source = inspect.getsource(broken.train_step)
    assert "print" in source, (
        "the logging was removed rather than fixed. The exercise is to keep the "
        "diagnostic and make it safe, not to delete it."
    )
