"""Negative tests: the bugs that do not raise.

These pass by *demonstrating* a failure mode. They exist because the dangerous
bugs in distributed training all share a shape: nothing errors, the loss still
falls, and the model is quietly worse than it should be. If you have never seen
one produce a plausible-looking loss curve, you will not recognise it in the wild.

Each test here answers "would my test suite have caught this?"
"""

from __future__ import annotations

import pytest
import torch

from common import collectives, dist_utils
from labs.lab1_ddp_bucketing.starter import bucket

pytestmark = [pytest.mark.tier1, pytest.mark.negative, pytest.mark.timeout(180)]


# --------------------------------------------------------------------------- #
# reading .grad before waiting on the async handle
# --------------------------------------------------------------------------- #


def _w_read_before_wait():
    """Issue an async all-reduce, read the tensor immediately, then wait.

    The value read early is *partially reduced*: some peers' contributions have
    landed and some have not. Right shape, right dtype, plausible magnitude,
    wrong number.

    Returns both readings so the test can show they differ.
    """
    rank, world = dist_utils.rank(), dist_utils.world_size()

    tensor = torch.full((2_000_000,), float(rank + 1))
    expected = float(sum(range(1, world + 1)))

    handle = collectives.all_reduce(tensor, async_op=True)
    early = float(tensor[0])  # read before wait: undefined by contract
    handle.wait()
    late = float(tensor[0])

    return {"rank": rank, "early": early, "late": late, "expected": expected}


def test_reading_grad_before_wait_gives_a_partial_result():
    """The value you get before waiting is not the value you get after.

    On gloo with a large buffer this reliably differs. If it does not on your
    machine, the collective completed before the read, which is timing luck rather
    than correctness: the contract says the tensor is undefined until `wait()`
    returns, and code that relies on luck breaks on a different backend or a
    different buffer size.

    The lesson for ``BucketedDDP``: ``finalize()`` before ``optimizer.step()``,
    always. Not "usually", not "when it matters".
    """
    results = dist_utils.run_distributed(
        _w_read_before_wait, world_size=4, backend="gloo", timeout_s=120
    )

    for rank, info in results.items():
        assert info["late"] == pytest.approx(info["expected"], rel=1e-6), (
            f"rank {rank}: even after wait() the value is wrong. That is a different "
            f"bug from the one this test is about."
        )

    differed = [r for r, info in results.items() if info["early"] != info["late"]]

    print("\n  reading an async all-reduce before wait():")
    for rank, info in sorted(results.items()):
        marker = "  <- partial" if info["early"] != info["late"] else ""
        print(
            f"    rank {rank}: early={info['early']:.1f}  "
            f"after wait={info['late']:.1f}  expected={info['expected']:.1f}{marker}"
        )

    if not differed:
        pytest.skip(
            "the collective completed before the read on every rank, so the partial "
            "state was not observable this run. The contract still says the tensor is "
            "undefined until wait() returns; increase the buffer size to see it."
        )


# --------------------------------------------------------------------------- #
# summing without averaging
# --------------------------------------------------------------------------- #


def _w_sum_instead_of_average():
    """The most common Lab 1 bug: all-reduce with SUM and forget to divide."""
    rank, world = dist_utils.rank(), dist_utils.world_size()

    params = [torch.nn.Parameter(torch.ones(64))]
    for p in params:
        p.grad = torch.full_like(p, float(rank + 1))

    b = bucket.GradientBucket(params, index=0)
    b.mark_ready(params[0])
    b.flush(world_size=world)
    correct = float(params[0].grad[0])

    # Now the buggy version, inline, so the comparison is unambiguous.
    grad = torch.full((64,), float(rank + 1))
    collectives.all_reduce(grad)  # no division
    buggy = float(grad[0])

    return {"rank": rank, "correct": correct, "buggy": buggy, "world": world}


def test_forgetting_to_divide_scales_gradients_by_world_size():
    """Off by exactly N, and the loss still falls.

    What actually happens downstream: the optimizer sees gradients N times too
    large, which is indistinguishable from an N-times-larger learning rate. At
    N=2 on a small model you may not notice for a hundred steps. At N=8 the run
    diverges and you blame the data.

    The diagnostic signature is that *every* element is off by the same integer
    factor. ``common.testing.compare_to_reference`` looks for that and says so.
    """
    results = dist_utils.run_distributed(
        _w_sum_instead_of_average, world_size=4, backend="gloo", timeout_s=120
    )

    for rank, info in results.items():
        expected_mean = sum(range(1, info["world"] + 1)) / info["world"]
        assert info["correct"] == pytest.approx(expected_mean, rel=1e-6), (
            f"rank {rank}: GradientBucket.flush did not divide by world_size"
        )
        assert info["buggy"] == pytest.approx(expected_mean * info["world"], rel=1e-6)

    print(f"\n  averaged correctly: {results[0]['correct']:.2f}")
    print(f"  summed, not divided: {results[0]['buggy']:.2f}  ({results[0]['world']}x too large)")


# --------------------------------------------------------------------------- #
# parameters never broadcast
# --------------------------------------------------------------------------- #


def _w_never_broadcast():
    """Each rank initialises independently. Gradients average fine. Model is junk."""
    rank = dist_utils.rank()

    torch.manual_seed(500 + rank)  # deliberately different per rank
    model = torch.nn.Linear(16, 1, bias=False)

    losses = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))
    y = torch.randn(8, 1, generator=torch.Generator().manual_seed(1))

    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        for p in model.parameters():  # correct averaging, wrong starting point
            collectives.all_reduce(p.grad)
            p.grad /= dist_utils.world_size()
        optimizer.step()
        losses.append(float(loss.detach()))

    return {"rank": rank, "losses": losses, "weight": model.weight.detach().flatten().tolist()}


def test_unbroadcast_parameters_still_produce_a_falling_loss():
    """The point of the whole repo, in one test.

    Every rank runs a different model. The gradients are averaged correctly. The
    loss falls on every rank. And the ranks disagree about what the weights are,
    so whichever checkpoint you save is a model that was optimised by gradients
    computed on a different model.

    No exception. No warning. A loss curve you would happily put in a report.

    The only thing that catches it is comparing against a reference, or
    ``assert_close_across_ranks`` on the parameters, which is exactly what
    ``test_01_semantics.py`` does.
    """
    results = dist_utils.run_distributed(
        _w_never_broadcast, world_size=2, backend="gloo", timeout_s=120
    )

    for rank, info in results.items():
        assert info["losses"][-1] < info["losses"][0], (
            f"rank {rank}: the loss did not fall, so this test is not demonstrating what it claims"
        )

    w0 = torch.tensor(results[0]["weight"])
    w1 = torch.tensor(results[1]["weight"])
    divergence = float((w0 - w1).abs().max())

    assert divergence > 1e-3, (
        "the ranks converged to the same weights, so the demonstration failed. "
        "Increase the seed spread or shorten the run."
    )

    print(
        f"\n  loss fell from {results[0]['losses'][0]:.4f} to "
        f"{results[0]['losses'][-1]:.4f} on rank 0"
    )
    print(
        f"  loss fell from {results[1]['losses'][0]:.4f} to "
        f"{results[1]['losses'][-1]:.4f} on rank 1"
    )
    print(f"  and the two ranks' weights differ by {divergence:.4f}")
    print("  Nothing raised. This is why a falling loss is not evidence.")
