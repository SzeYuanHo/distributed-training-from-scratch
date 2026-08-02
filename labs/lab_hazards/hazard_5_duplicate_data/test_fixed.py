"""Passes once each rank trains on its own slice of the data."""

from __future__ import annotations

import pytest

from common import data as data_mod  # noqa: E402
from common import dist_utils  # noqa: E402

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(180)]


def _w_batches_are_disjoint():
    from . import broken

    prints = []
    for step in range(5):
        x, _y = broken.get_my_batch(step)
        prints.append(data_mod.batch_fingerprint(x))
    return {"rank": dist_utils.rank(), "fingerprints": prints}


@pytest.mark.parametrize("world", [2, 4])
def test_every_rank_gets_different_data(world):
    """The direct check, and the one worth copying into your own training loops."""
    results = dist_utils.run_distributed(
        _w_batches_are_disjoint, world_size=world, backend="gloo", timeout_s=90
    )

    for step in range(5):
        prints = [results[r]["fingerprints"][step] for r in sorted(results)]
        assert len(set(prints)) == world, (
            f"step {step}: {world} ranks produced {len(set(prints))} distinct batches.\n"
            f"  fingerprints: {prints}\n"
            "  Identical batches mean the all-reduce is averaging identical gradients, "
            "so this is a single-process run with extra electricity."
        )


def _w_union_covers_the_global_batch():
    """The stronger property, and the one Lab 1's equivalence tests depend on.

    The union of the ranks' batches at step s must equal the single-process batch
    of N times the size at the same step. Disjointness alone is not enough:
    ranks could be reading disjoint but arbitrary slices and the comparison
    against a single-process reference would still be meaningless.
    """
    from . import broken

    x, _y = broken.get_my_batch(0)
    return {"rank": dist_utils.rank(), "batch": x.tolist()}


@pytest.mark.parametrize("world", [2, 4])
def test_the_union_reconstructs_the_global_batch(world):
    import torch

    results = dist_utils.run_distributed(
        _w_union_covers_the_global_batch, world_size=world, backend="gloo", timeout_s=90
    )

    from . import broken

    dataset = data_mod.load_dataset()
    expected, _ = data_mod.get_batch(
        "train",
        batch_size=broken.BATCH_PER_RANK * world,
        block_size=broken.BLOCK,
        rank=0,
        world_size=1,
        step=0,
        dataset=dataset,
    )

    pieces = [torch.tensor(results[r]["batch"]) for r in sorted(results)]
    union = torch.cat(pieces, dim=0)

    assert torch.equal(union, expected), (
        f"the {world} ranks' batches do not reassemble the global batch.\n"
        "  common.data.get_batch guarantees this when you pass the real rank and "
        "world size. It is what makes an N-rank run comparable to a 1-rank "
        "reference, and therefore what makes every equivalence test in this repo "
        "meaningful."
    )


def test_the_loss_still_falls():
    """The fix must not break training. Convergence will look slower now, and should."""
    from . import broken

    results = dist_utils.run_distributed(broken.worker, world_size=4, backend="gloo", timeout_s=150)
    losses = results[0]["losses"]
    assert losses[-1] < losses[0], f"the loss went from {losses[0]:.4f} to {losses[-1]:.4f}"


def test_no_hardcoded_rank_remains():
    """Guard against fixing it in the caller and leaving the trap in place."""
    import inspect

    from . import broken

    source = inspect.getsource(broken.get_my_batch)
    assert "rank=0" not in source.replace(" ", ""), (
        "get_my_batch still passes rank=0. Pass dist_utils.rank()."
    )
    assert "world_size=1" not in source.replace(" ", ""), (
        "get_my_batch still passes world_size=1. Pass dist_utils.world_size()."
    )
