"""Passes once the distributed run matches the single-process reference."""

from __future__ import annotations

import pytest

from common import comm_counter, dist_utils, testing  # noqa: E402

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(180)]


def test_distributed_matches_the_reference():
    """The only test that can find this bug. Everything else says the run is fine."""
    from . import broken

    results = dist_utils.run_distributed(broken.worker, world_size=4, backend="gloo", timeout_s=120)
    reference = broken.reference_losses()

    for rank, info in results.items():
        testing.assert_curves_close(
            info["losses"],
            reference,
            rtol=3e-2,
            name=f"rank {rank} loss curve against the single-process reference",
        )


def _w_every_handle_is_waited():
    """The invariant, not the outcome.

    The CommCounter records a `wait[...]` event for every handle that gets waited
    on. Comparing that count against the number of async collectives issued says
    directly whether anything was left in flight.
    """
    from . import broken

    model = broken.build_model()

    import torch

    x = torch.randn(4, broken.HIDDEN)
    y = torch.randn(4, 8)
    torch.nn.functional.mse_loss(model(x), y).backward()

    with comm_counter.CommCounter() as counter:
        broken.sync_gradients_fast(model)

    counts = counter.counts()
    issued = counts.get("all_reduce", 0)
    waited = sum(v for k, v in counts.items() if k.startswith("wait["))

    return {"rank": dist_utils.rank(), "issued": issued, "waited": waited}


def test_every_async_collective_is_waited_on():
    results = dist_utils.run_distributed(
        _w_every_handle_is_waited, world_size=2, backend="gloo", timeout_s=90
    )

    for rank, info in results.items():
        assert info["waited"] >= info["issued"], (
            f"rank {rank} issued {info['issued']} async collectives and waited on "
            f"{info['waited']} of them.\n"
            "  Between async_op=True and handle.wait(), the tensor's contents are "
            "undefined. Not 'probably finished'. Undefined.\n"
            "  Issue every collective, then wait on every handle, then read the "
            "gradients. Three phases, in that order."
        )


def test_the_overlap_was_not_thrown_away():
    """The fix should keep the collectives concurrent.

    Making it correct by switching to synchronous all-reduces passes the numeric
    test and gives up the optimisation the function was written for. The point is
    that overlap and correctness are compatible.
    """
    import inspect

    from . import broken

    source = inspect.getsource(broken.sync_gradients_fast)
    assert "async_op" in source, (
        "the async path was removed. Correctness by giving up the optimisation "
        "misses the lesson: issue everything, then wait on everything, and the "
        "collectives still run concurrently."
    )
    assert "wait" in source, "no wait() anywhere; see HINT.md"


def test_ranks_still_agree_about_the_model():
    """Regression guard: fixing hazard 3 must not reintroduce hazard 2."""
    from . import broken

    results = dist_utils.run_distributed(broken.worker, world_size=4, backend="gloo", timeout_s=120)
    weights = [info["final_weight"] for info in results.values()]
    assert max(weights) - min(weights) < 1e-4, (
        f"the ranks' final weights differ by {max(weights) - min(weights):.3e}"
    )
