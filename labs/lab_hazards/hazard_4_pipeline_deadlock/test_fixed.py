"""Passes once the exchange survives a large message.

The timeout is the assertion. Without it a deadlock stalls the suite instead of
failing it.
"""

from __future__ import annotations

import pytest

from common import dist_utils  # noqa: E402

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(150)]


@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.timeout(60)
def test_exchange_survives_every_size(batch):
    """The direct check. 64 is the one that used to hang."""
    from . import broken

    results = dist_utils.run_distributed(
        broken.worker,
        world_size=4,
        backend="gloo",
        args=(batch,),
        timeout_s=25,
        join_timeout_s=45,
    )
    assert len(results) == 4
    for rank, info in results.items():
        assert info["correct"], (
            f"rank {rank} received the wrong data at batch {batch}. It should hold "
            f"the value sent by rank {(rank - 1) % 4}."
        )


@pytest.mark.timeout(90)
def test_exchange_survives_a_genuinely_large_message():
    """Past any plausible eager threshold.

    Passing at 64 could be luck if your transport buffers generously. Four
    megabytes will not be buffered anywhere.
    """
    from . import broken

    batch = 4096  # 4096 * 256 * 4 bytes = 4 MB per message

    results = dist_utils.run_distributed(
        broken.worker,
        world_size=4,
        backend="gloo",
        args=(batch,),
        timeout_s=40,
        join_timeout_s=70,
    )
    for rank, info in results.items():
        assert info["correct"], f"rank {rank} got the wrong data at 4 MB"
        assert info["bytes"] > 4_000_000


@pytest.mark.timeout(60)
def test_it_works_at_an_odd_world_size():
    """Three ranks in a ring.

    The odd/even fix has a hole here: in a ring of three, ranks 0 and 2 are
    adjacent and both even, so both send first and neither receives. If you took
    that route, this is where it shows.
    """
    from . import broken

    results = dist_utils.run_distributed(
        broken.worker,
        world_size=3,
        backend="gloo",
        args=(64,),
        timeout_s=25,
        join_timeout_s=45,
    )
    assert len(results) == 3
    for rank, info in results.items():
        assert info["correct"], (
            f"rank {rank} of 3 got the wrong data.\n"
            "  If you used odd/even ordering: in a ring of three, ranks 0 and 2 are "
            "neighbours and both even, so both send first. Use isend/irecv or "
            "batch_isend_irecv instead."
        )


def test_the_exchange_is_still_a_ring():
    """The fix should not turn the point-to-point exchange into a collective.

    Replacing the whole thing with an all-gather makes the test pass and discards
    the topology that pipeline parallelism depends on.
    """
    import inspect

    from . import broken

    source = inspect.getsource(broken.exchange_activations)
    for forbidden in ("all_gather", "all_reduce", "all_to_all"):
        assert forbidden not in source, (
            f"{forbidden} appeared in the exchange. A pipeline stage talks to its two "
            "neighbours, not to everyone; using a collective here moves N times the "
            "bytes and defeats the point of the topology."
        )
