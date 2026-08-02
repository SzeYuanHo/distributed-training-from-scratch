"""The f/g swap: correct forward, wrong gradients, falling loss.

This is the signature bug of tensor parallelism and it deserves a test of its own.

What the swap does: put ``g`` (all-reduce forward) where ``f`` (identity forward)
belongs and vice versa. Both orderings produce the *same forward output*, because
an all-reduce of a replicated tensor is just a scale by the world size and the
structure absorbs it. Only the backward differs, and it differs by a factor that
is close enough to right that training still appears to work.

If you have never seen a wrong model produce a convincing loss curve, this is your
chance.
"""

from __future__ import annotations

import pytest
import torch

from common import dist_utils
from labs.lab2_tensor_parallel.reference import plain_blocks
from labs.lab2_tensor_parallel.starter import layers, mappings

pytestmark = [pytest.mark.tier1, pytest.mark.negative, pytest.mark.timeout(240)]


class _SwappedColumnParallel(torch.nn.Module):
    """A column-parallel layer with f and g the wrong way round.

    Deliberately broken. Built here rather than in ``starter/`` so it cannot be
    mistaken for a starting point.
    """

    def __init__(self, correct: layers.ColumnParallelLinear):
        super().__init__()
        self.inner = correct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The bug: g where f belongs.
        x = mappings.ReduceFromTensorParallelRegion.apply(x, None)
        return torch.nn.functional.linear(x, self.inner.weight, self.inner.bias)


def _w_swap_breaks_gradients_only():
    rank, world = dist_utils.rank(), dist_utils.world_size()

    reference = plain_blocks.make_linear(32, 64, bias=True, seed=0)
    x_base = torch.randn(6, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    expected = reference(x_base)
    x_ref = x_base.clone().requires_grad_(True)
    reference(x_ref).sum().backward()
    expected_grad = x_ref.grad.clone()

    # Correct implementation.
    correct = layers.ColumnParallelLinear(32, 64, bias=True, gather_output=True).double()
    correct.load_from_reference(reference)

    x_ok = x_base.clone().requires_grad_(True)
    out_ok = correct(x_ok)
    out_ok.sum().backward()

    forward_ok = torch.allclose(out_ok.detach(), expected, rtol=1e-9, atol=1e-11)
    grad_ok = torch.allclose(x_ok.grad, expected_grad, rtol=1e-9, atol=1e-11)

    # Swapped implementation.
    broken = _SwappedColumnParallel(correct).double()
    x_bad = x_base.clone().requires_grad_(True)
    out_bad = broken(x_bad)
    out_bad.sum().backward()

    grad_ratio = float((x_bad.grad / x_ok.grad).abs().median())

    return {
        "rank": rank,
        "world": world,
        "forward_ok": bool(forward_ok),
        "grad_ok": bool(grad_ok),
        "grad_ratio": grad_ratio,
        "swapped_grad_matches": bool(
            torch.allclose(x_bad.grad, expected_grad, rtol=1e-6, atol=1e-8)
        ),
    }


@pytest.mark.parametrize("world", [2, 4])
def test_swapping_f_and_g_breaks_only_the_gradients(world):
    """The suite has to catch this, and a forward-only test would not.

    The assertion is not "the swapped version is wrong" alone. It is that the
    swapped version is wrong *in the backward pass specifically*, so any test that
    checks outputs and not gradients would pass it happily.
    """
    results = dist_utils.run_distributed(
        _w_swap_breaks_gradients_only, world_size=world, backend="gloo", timeout_s=200
    )

    for rank, info in results.items():
        assert info["forward_ok"], (
            f"rank {rank}: the correct implementation's forward pass does not match "
            "the reference, so this test cannot demonstrate anything yet. Finish "
            "step 3 first."
        )
        assert info["grad_ok"], (
            f"rank {rank}: the correct implementation's gradients do not match. "
            "Finish step 3 first."
        )
        assert not info["swapped_grad_matches"], (
            f"rank {rank}: the swapped version produced the right gradients, which "
            "means f and g are currently doing the same thing. Check that one of them "
            "is an identity forward and the other an all-reduce forward."
        )

    print(f"\n  world {world}: with f and g swapped,")
    print("    forward output   still correct")
    print(f"    input gradient   off by a factor of about {results[0]['grad_ratio']:.2f}")
    print("  A model built this way trains and shows a falling loss. The only thing")
    print("  that catches it is a gradient comparison against a reference.")


def _w_bare_allreduce_breaks_the_graph():
    """A collective outside an autograd Function silently detaches the backward.

    ``dist.all_reduce`` is not differentiable. Autograd records nothing, so the
    gradient that flows back is the gradient of the *pre-reduce* tensor. No error,
    no warning, wrong answer.
    """
    import torch.distributed as dist

    world = dist_utils.world_size()

    x = torch.ones(4, requires_grad=True)
    y = x * 2
    z = y.clone()
    dist.all_reduce(z)  # bare collective on a tracked tensor
    z.sum().backward()
    bare_grad = x.grad.clone()

    x2 = torch.ones(4, requires_grad=True)
    y2 = x2 * 2
    z2 = mappings.ReduceFromTensorParallelRegion.apply(y2, None)
    z2.sum().backward()
    proper_grad = x2.grad.clone()

    return {
        "rank": dist_utils.rank(),
        "world": world,
        "bare": bare_grad.tolist(),
        "proper": proper_grad.tolist(),
    }


@pytest.mark.parametrize("world", [2, 4])
def test_bare_allreduce_produces_a_detached_gradient(world):
    """Why every in-model collective goes through a Function.

    Both paths compute the same forward value. Only the one wrapped in a
    ``torch.autograd.Function`` produces a gradient that accounts for the
    reduction.
    """
    results = dist_utils.run_distributed(
        _w_bare_allreduce_breaks_the_graph,
        world_size=world,
        backend="gloo",
        timeout_s=200,
    )

    info = results[0]
    print(f"\n  world {world}:")
    print(f"    bare dist.all_reduce  -> grad {info['bare']}")
    print(f"    wrapped in a Function -> grad {info['proper']}")
    print("  Same forward value, different gradients. Nothing raised either way.")

    assert info["proper"] == pytest.approx([2.0] * 4), (
        "g's backward is the identity, so the gradient should be d(2x)/dx = 2 on every element"
    )
