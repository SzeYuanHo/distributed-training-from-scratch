"""Tier 0: bucket mechanics and accumulation algebra, in one process.

No spawning, no gloo. These run in under a second and they catch the bugs that
would otherwise show up as a mysterious loss-curve divergence at world size 4.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from labs.lab1_ddp_bucketing.starter import bucket

pytestmark = pytest.mark.tier0


def _toy_params(sizes: list[int], requires_grad: bool = True) -> list[nn.Parameter]:
    return [nn.Parameter(torch.randn(n), requires_grad=requires_grad) for n in sizes]


def _with_grads(params: list[nn.Parameter]) -> list[nn.Parameter]:
    for p in params:
        p.grad = torch.randn_like(p)
    return params


# --------------------------------------------------------------------------- #
# flatten / unflatten
# --------------------------------------------------------------------------- #


def test_param_bytes():
    p = nn.Parameter(torch.zeros(100, dtype=torch.float32))
    assert bucket.param_bytes(p) == 400
    q = nn.Parameter(torch.zeros(100, dtype=torch.float64))
    assert bucket.param_bytes(q) == 800


def test_bucket_roundtrip_is_exact():
    """Flatten then unflatten must be bit-identical. Not close: identical.

    A round trip that loses bits means you went through a dtype conversion
    somewhere, and every gradient in the run is then quietly quantised.
    """
    torch.manual_seed(0)
    params = _with_grads(_toy_params([10, 3, 47, 1]))
    originals = [p.grad.clone() for p in params]

    b = bucket.GradientBucket(params, index=0)
    for p in params:
        b.mark_ready(p)
    b.flush(world_size=1)

    for i, (p, original) in enumerate(zip(params, originals)):
        assert torch.equal(p.grad, original), (
            f"parameter {i}: the flatten/unflatten round trip changed the gradient. "
            f"max diff {(p.grad - original).abs().max():.3e}"
        )


def test_bucket_flush_divides_by_world_size():
    """world_size=4 with no actual peers should still divide by 4.

    Skipping the division is the single most common Lab 1 bug and it is silent:
    the loss falls, just with an effective learning rate N times too high.
    """
    params = _with_grads(_toy_params([8]))
    original = params[0].grad.clone()

    b = bucket.GradientBucket(params, index=0)
    b.mark_ready(params[0])
    b.flush(world_size=4)

    torch.testing.assert_close(
        params[0].grad,
        original / 4,
        rtol=1e-6,
        atol=1e-8,
        msg="flush must divide by world_size; all_reduce with SUM gives N times the mean",
    )


def test_bucket_total_bytes():
    params = _toy_params([100, 200, 50])
    b = bucket.GradientBucket(params, index=0)
    assert b.total_bytes() == 350 * 4


def test_mark_ready_is_idempotent():
    """A tied parameter can fire its hook twice. Counting it twice fills the bucket early."""
    params = _with_grads(_toy_params([4, 4]))
    b = bucket.GradientBucket(params, index=0)

    b.mark_ready(params[0])
    b.mark_ready(params[0])  # same parameter again

    assert not b.is_full(), (
        "the bucket reported full after one parameter arrived twice. "
        "Track readiness by identity, not by a counter."
    )
    b.mark_ready(params[1])
    assert b.is_full()


def test_reset_clears_state():
    params = _with_grads(_toy_params([4, 4]))
    b = bucket.GradientBucket(params, index=0)
    for p in params:
        b.mark_ready(p)
    assert b.is_full()

    b.reset()
    assert not b.is_full(), "reset() must clear readiness for the next iteration"


# --------------------------------------------------------------------------- #
# bucket assignment
# --------------------------------------------------------------------------- #


def test_bucket_assignment_covers_every_trainable_parameter():
    params = _toy_params([100, 100, 100, 100])
    buckets = bucket.assign_buckets(params, bucket_bytes=1000)

    seen = [p for b in buckets for p in b.params]
    assert len(seen) == len(params), f"expected {len(params)} parameters, saw {len(seen)}"
    assert {id(p) for p in seen} == {id(p) for p in params}


def test_bucket_assignment_excludes_frozen_parameters():
    """A frozen parameter never gets a gradient, so a bucket holding one never fills.

    That is step 8's hang arriving early and for a boring reason. Filter them out.
    """
    trainable = _toy_params([100, 100])
    frozen = _toy_params([100], requires_grad=False)
    buckets = bucket.assign_buckets(trainable + frozen, bucket_bytes=10_000)

    seen = {id(p) for b in buckets for p in b.params}
    assert id(frozen[0]) not in seen, (
        "a parameter with requires_grad=False was put in a bucket. It will never "
        "receive a gradient, so that bucket can never fill and finalize() will hang."
    )


def test_bucket_assignment_respects_the_cap():
    params = _toy_params([100] * 10)  # 400 bytes each, 4000 total
    buckets = bucket.assign_buckets(params, bucket_bytes=1000)

    assert len(buckets) >= 4, f"4000 bytes at a 1000-byte cap needs 4+ buckets, got {len(buckets)}"
    for i, b in enumerate(buckets):
        if len(b.params) > 1:
            assert b.total_bytes() <= 1000, (
                f"bucket {i} holds {b.total_bytes()} bytes against a 1000-byte cap"
            )


def test_a_parameter_larger_than_the_cap_gets_its_own_bucket():
    params = _toy_params([10_000, 10, 10])  # 40 KB, then two tiny ones
    buckets = bucket.assign_buckets(params, bucket_bytes=1000)

    big = [b for b in buckets if any(p.numel() == 10_000 for p in b.params)]
    assert len(big) == 1
    assert len(big[0].params) == 1, (
        "an oversized parameter should get a bucket to itself rather than dragging "
        "small parameters into a flush that is already over budget"
    )


def test_reverse_order_puts_the_last_parameter_first():
    """The whole argument for reverse order, as an assertion.

    Gradients arrive in reverse forward order, so bucket 0 should hold the
    parameters that arrive first, which are the last-registered ones.
    """
    params = _toy_params([100, 100, 100, 100])
    buckets = bucket.assign_buckets(params, bucket_bytes=400, reverse=True)

    assert id(buckets[0].params[0]) == id(params[-1]), (
        "with reverse=True, bucket 0 should start with the LAST registered parameter, "
        "because that is the first gradient the backward pass produces. See THEORY.md "
        "section 3."
    )

    forward = bucket.assign_buckets(params, bucket_bytes=400, reverse=False)
    assert id(forward[0].params[0]) == id(params[0])


def test_tied_weights_are_not_bucketed_twice():
    """The reference model ties wte and lm_head. One object, two names.

    Bucketing it twice means all-reducing it twice: correct answer, double the
    bytes, and a collective count that quietly disagrees with your prediction.
    """
    shared = nn.Parameter(torch.randn(50))
    others = _toy_params([50, 50])
    params = [shared, *others, shared]  # appears twice, as named_parameters would

    buckets = bucket.assign_buckets(params, bucket_bytes=10_000)

    ids = [id(p) for b in buckets for p in b.params]
    assert len(ids) == len(set(ids)), (
        "a parameter appears in more than one bucket. Deduplicate by identity: "
        "tied weights show up twice in named_parameters() but are one tensor."
    )


# --------------------------------------------------------------------------- #
# gradient accumulation, no distribution involved
# --------------------------------------------------------------------------- #


def test_accumulation_equals_one_large_batch():
    """THEORY section 4, as a test. Pure autograd, no communication.

    If this fails, the problem is your understanding of loss scaling, not your
    DDP. Better to find that out here.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)

    # One batch of 16.
    model.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    large = [p.grad.clone() for p in model.parameters()]

    # Four micro-batches of 4, each loss scaled by 1/4.
    model.zero_grad()
    for i in range(4):
        chunk = slice(i * 4, (i + 1) * 4)
        (torch.nn.functional.mse_loss(model(x[chunk]), y[chunk]) / 4).backward()
    accumulated = [p.grad.clone() for p in model.parameters()]

    for i, (a, b) in enumerate(zip(large, accumulated)):
        torch.testing.assert_close(
            a,
            b,
            rtol=1e-5,
            atol=1e-7,
            msg=f"parameter {i}: accumulation does not match the large batch",
        )


def test_accumulation_without_scaling_is_wrong_by_exactly_k():
    """The negative case: forget the 1/k and the gradients come out k times too big.

    Worth seeing as a number. It is the same failure shape as forgetting to divide
    an all-reduce by the world size, and recognising the shape is what lets you
    diagnose it in ten seconds instead of an hour.
    """
    torch.manual_seed(0)
    model = nn.Linear(8, 1)
    x, y = torch.randn(16, 8), torch.randn(16, 1)

    model.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    correct = model.weight.grad.clone()

    model.zero_grad()
    for i in range(4):
        chunk = slice(i * 4, (i + 1) * 4)
        torch.nn.functional.mse_loss(model(x[chunk]), y[chunk]).backward()  # no /4
    unscaled = model.weight.grad.clone()

    ratio = (unscaled / correct).mean()
    assert abs(float(ratio) - 4.0) < 0.01, (
        f"expected exactly 4x, measured {float(ratio):.3f}. If this is not 4, the "
        "test itself is wrong."
    )
