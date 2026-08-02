"""Tier 1: the 2D mesh and the composed model.

Needs 4 ranks for a 2x2 mesh. On a machine with fewer than 4 cores these will run
and be slow.
"""

from __future__ import annotations

import pytest
import torch

from common import comm_counter, dist_utils, testing
from labs.lab4_capstone_2d.starter import compose, mesh

pytestmark = [pytest.mark.tier1, pytest.mark.timeout(300)]


# --------------------------------------------------------------------------- #
# step 1: the mesh
# --------------------------------------------------------------------------- #


def _w_mesh(dp_size: int, tp_size: int):
    layout = mesh.build_mesh(dp_size, tp_size)
    return {
        "rank": dist_utils.rank(),
        "dp_rank": layout.dp_rank,
        "tp_rank": layout.tp_rank,
        "dp_ranks": list(layout.dp_ranks),
        "tp_ranks": list(layout.tp_ranks),
        "text": str(layout),
    }


@pytest.mark.parametrize("dp_size,tp_size", [(2, 2), (1, 4), (4, 1)])
def test_mesh_coordinates(dp_size, tp_size):
    """TP innermost: contiguous ranks share a TP group.

    On a real machine, contiguous ranks share a node, so this is what puts
    tensor-parallel traffic on the fast link. Getting it backwards produces a
    working job that runs at a fraction of the speed, which is much harder to
    notice than a crash.
    """
    world = dp_size * tp_size
    results = dist_utils.run_distributed(
        _w_mesh,
        world_size=world,
        backend="gloo",
        args=(dp_size, tp_size),
        timeout_s=180,
    )

    for rank, info in results.items():
        assert info["tp_rank"] == rank % tp_size, (
            f"rank {rank}: tp_rank is {info['tp_rank']}, expected {rank % tp_size}. "
            "TP is the innermost dimension, so tp_rank = global_rank % tp_size."
        )
        assert info["dp_rank"] == rank // tp_size, (
            f"rank {rank}: dp_rank is {info['dp_rank']}, expected {rank // tp_size}"
        )

    print()
    for rank in sorted(results):
        print(f"  {results[rank]['text']}")


def test_tp_groups_are_contiguous():
    """The property that matters on real hardware.

    At dp=2, tp=2 the TP groups must be [0,1] and [2,3], not [0,2] and [1,3]. On a
    machine where ranks 0 and 1 share a node, the first layout keeps TP on NVLink
    and the second sends it over the network.
    """
    results = dist_utils.run_distributed(
        _w_mesh, world_size=4, backend="gloo", args=(2, 2), timeout_s=180
    )

    tp_groups = {tuple(sorted(info["tp_ranks"])) for info in results.values()}
    assert tp_groups == {(0, 1), (2, 3)}, (
        f"TP groups came out as {tp_groups}, expected {{(0,1), (2,3)}}. "
        "Contiguous ranks must share a TP group."
    )

    dp_groups = {tuple(sorted(info["dp_ranks"])) for info in results.values()}
    assert dp_groups == {(0, 2), (1, 3)}, (
        f"DP groups came out as {dp_groups}, expected {{(0,2), (1,3)}}"
    )


def _w_mesh_rejects_bad_sizes():
    try:
        mesh.build_mesh(3, 3)  # 9 != world size
    except ValueError as exc:
        if "9" not in str(exc) and str(dist_utils.world_size()) not in str(exc):
            raise AssertionError(f"the error should name the mismatched sizes; got: {exc}") from exc
        return "raised"
    raise AssertionError("dp*tp not equal to the world size was accepted")


def test_mesh_rejects_mismatched_sizes():
    results = dist_utils.run_distributed(
        _w_mesh_rejects_bad_sizes, world_size=4, backend="gloo", timeout_s=180
    )
    assert all(v == "raised" for v in results.values())


def _w_verify_against_torch(dp_size: int, tp_size: int):
    layout = mesh.build_mesh(dp_size, tp_size)
    return mesh.verify_against_device_mesh(layout)


def test_matches_torch_device_mesh():
    """Cross-check the hand-built groups against init_device_mesh.

    Skips when the build cannot make a DeviceMesh on gloo, which happens on some
    torch versions and is not your problem.
    """
    results = dist_utils.run_distributed(
        _w_verify_against_torch,
        world_size=4,
        backend="gloo",
        args=(2, 2),
        timeout_s=180,
    )
    values = set(results.values())
    if values == {False}:
        pytest.skip("DeviceMesh is unavailable on this backend or torch build")
    assert values == {True}, (
        "the hand-built mesh disagrees with torch's. Torch puts the last dimension "
        "innermost too, so passing (dp, tp) should match what you built."
    )


# --------------------------------------------------------------------------- #
# step 2: the composed model
# --------------------------------------------------------------------------- #


def _w_elementwise_rank_comparison():
    rank = dist_utils.rank()
    tensor = torch.tensor([1_000_000.0, rank * 1e-2])
    try:
        testing.assert_close_across_ranks(tensor, rtol=1e-6, atol=1e-8)
    except AssertionError:
        return "detected"
    return "missed"


def test_rank_comparison_uses_elementwise_tolerances():
    """A large element must not loosen the tolerance for every small element."""
    results = dist_utils.run_distributed(
        _w_elementwise_rank_comparison,
        world_size=2,
        backend="gloo",
        timeout_s=120,
    )
    assert set(results.values()) == {"detected"}


def _w_compose(dp_size: int, tp_size: int, steps: int):
    from common import data as data_mod
    from common import reference_gpt

    layout = mesh.build_mesh(dp_size, tp_size)
    dataset = data_mod.load_dataset()

    torch.manual_seed(1337)
    config = reference_gpt.GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=32,
        vocab_size=dataset.vocab_size,
        dropout=0.0,
        max_tp=4,
    )

    model = compose.Parallel2DModel(config, layout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    losses = []
    counts: dict[str, int] = {}

    for step in range(steps):
        # Data depends on the DP coordinate only. Every rank in a TP group sees
        # the same batch, because they are jointly computing one forward pass.
        x, y = data_mod.get_batch(
            "train",
            batch_size=4,
            block_size=32,
            rank=layout.dp_rank,
            world_size=layout.dp_size,
            step=step,
            dataset=dataset,
        )

        with comm_counter.CommCounter() as counter:
            _logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.sync_gradients()
            if step == 0:
                counts = dict(counter.counts())
            optimizer.step()

        losses.append(float(loss.detach()))

    for name, param in model.named_parameters():
        if "ln_" in name:  # LayerNorms are replicated across TP
            testing.assert_close_across_ranks(
                param.data, rtol=1e-5, atol=1e-7, name=name, group=layout.tp_group
            )

    return {
        "rank": dist_utils.rank(),
        "dp_rank": layout.dp_rank,
        "tp_rank": layout.tp_rank,
        "losses": losses,
        "counts": counts,
    }


@pytest.mark.slow
def test_2d_composition_matches_reference():
    """A 2x2 mesh should train the same model a single process would.

    The comparison is against a 1x1 run of the same code, so the test isolates
    "does composing the two axes work" from "is Lab 2's tensor parallelism
    correct", which Lab 2's own tests already cover.
    """
    single = dist_utils.run_distributed(
        _w_compose, world_size=1, backend="gloo", args=(1, 1, 8), timeout_s=280
    )
    composed = dist_utils.run_distributed(
        _w_compose, world_size=4, backend="gloo", args=(2, 2, 8), timeout_s=280
    )

    expected = single[0]["losses"]
    for rank, info in composed.items():
        testing.assert_curves_close(
            info["losses"],
            expected,
            rtol=5e-2,
            name=f"2x2 mesh loss curve (rank {rank})",
        )


def test_ranks_in_a_tp_group_see_the_same_data():
    """Every rank in a TP group must see the same batch.

    Different batches inside a TP group means the tensor-parallel arithmetic is
    combining activations from unrelated examples. Nothing raises, the loss falls,
    the result is nonsense.
    """
    from common import data as data_mod

    dataset = data_mod.load_dataset()
    a, _ = data_mod.get_batch(
        "train", batch_size=4, block_size=32, rank=0, world_size=2, step=0, dataset=dataset
    )
    b, _ = data_mod.get_batch(
        "train", batch_size=4, block_size=32, rank=0, world_size=2, step=0, dataset=dataset
    )
    assert torch.equal(a, b), (
        "two calls with the same dp coordinate returned different batches, so "
        "indexing by dp_rank would not give TP peers the same data"
    )

    c, _ = data_mod.get_batch(
        "train", batch_size=4, block_size=32, rank=1, world_size=2, step=0, dataset=dataset
    )
    assert not torch.equal(a, c), "different dp coordinates must see different data"


@pytest.mark.slow
def test_collective_counts_split_across_the_two_groups():
    """DP and TP traffic should be separable and should match your prediction."""
    results = dist_utils.run_distributed(
        _w_compose, world_size=4, backend="gloo", args=(2, 2, 2), timeout_s=280
    )

    for rank, info in results.items():
        total = info["counts"].get("all_reduce", 0)
        assert total > 0, f"rank {rank} issued no all-reduces at all"
        print(f"\n  rank {rank} (dp={info['dp_rank']} tp={info['tp_rank']}): {info['counts']}")
