"""Comparison helpers with failure messages worth reading.

``AssertionError: Tensors are not close!`` costs ten minutes every time it
appears. These helpers say which element, how far off, and where in the model,
so you can go straight to the layer instead of bisecting.

Two rules that the whole test suite depends on:

**Never assert bitwise equality across different world sizes.** Floating-point
addition is not associative. A 4-rank ring all-reduce sums the same numbers in a
different order from a 2-rank one and from a single process, so the last few
bits differ. That is arithmetic, not a bug. Use ``rtol=1e-5`` for fp32, and when
you genuinely need tighter, run the comparison in float64 rather than tightening
below what fp32 can represent.

**A falling loss is never evidence of correctness.** Every silent bug in
``docs/DEBUGGING.md`` produces a falling loss: unbroadcast weights, gradients
summed but not averaged, ``f`` and ``g`` swapped, ranks reading the same data.
The only thing that catches them is comparison against a frozen single-process
reference, which is why every lab has one.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch

__all__ = [
    "assert_close_across_ranks",
    "compare_to_reference",
    "assert_curves_close",
    "deterministic_context",
    "requires_cuda_devices",
    "requires_min_ranks",
    "gather_object",
    "TOLERANCES",
]

# Sensible defaults by dtype. fp32 accumulated over a few hundred ops loses
# roughly five decimal digits, so 1e-5 relative is about as tight as honest.
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float64: (1e-12, 1e-14),
    torch.float32: (1e-5, 1e-6),
    torch.bfloat16: (2e-2, 1e-3),
    torch.float16: (1e-3, 1e-4),
}


def _tolerances(
    tensor: torch.Tensor, rtol: float | None, atol: float | None
) -> tuple[float, float]:
    default_r, default_a = TOLERANCES.get(tensor.dtype, (1e-5, 1e-6))
    return (rtol if rtol is not None else default_r, atol if atol is not None else default_a)


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #


def compare_to_reference(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    name: str = "tensor",
    raise_on_fail: bool = True,
) -> bool:
    """Compare two tensors and, on failure, say exactly what went wrong.

    The report names the worst element, its index, both values, and the
    relative error, plus the fraction of elements that disagree. That last
    number is diagnostic on its own:

    * a handful of elements off by a hair is floating-point noise
    * everything off by a constant factor usually means a missing division by
      world size
    * exactly half the elements wrong usually means a sharding bug where one
      rank's half never arrived
    """
    if actual.shape != expected.shape:
        message = (
            f"{name}: shape mismatch. Got {tuple(actual.shape)}, expected {tuple(expected.shape)}."
        )
        if raise_on_fail:
            raise AssertionError(message)
        print(message)
        return False

    a = actual.detach().to(torch.float64).cpu()
    e = expected.detach().to(torch.float64).cpu()
    rtol_, atol_ = _tolerances(expected, rtol, atol)

    finite = torch.isfinite(a) & torch.isfinite(e)
    if not finite.all():
        nonfinite = ~finite
        flat_first = int(nonfinite.reshape(-1).nonzero()[0])
        index = torch.unravel_index(torch.tensor(flat_first), a.shape)
        index = tuple(int(i) for i in index)
        message = (
            f"{name}: {int(nonfinite.sum())} of {nonfinite.numel()} elements are non-finite.\n"
            f"  first at index {index}\n"
            f"    got      {float(a[index]):.10g}\n"
            f"    expected {float(e[index]):.10g}\n"
            "  NaN and infinity are always treated as failures, even when both "
            "tensors contain them at the same position."
        )
        if raise_on_fail:
            raise AssertionError(message)
        print(message)
        return False

    diff = (a - e).abs()
    tolerance = atol_ + rtol_ * e.abs()
    bad = diff > tolerance

    if not bad.any():
        return True

    flat_worst = int(diff.argmax())
    index = torch.unravel_index(torch.tensor(flat_worst), a.shape)
    index = tuple(int(i) for i in index)
    got, want = float(a[index]), float(e[index])
    rel = abs(got - want) / max(abs(want), 1e-30)

    n_bad, n_total = int(bad.sum()), bad.numel()
    ratio_hint = ""
    if want != 0:
        ratio = got / want
        if abs(ratio - round(ratio)) < 1e-6 and round(ratio) not in (0, 1):
            ratio_hint = (
                f"\n  Every bad element looks like a factor of {round(ratio)}. "
                f"Missing a division by world_size?"
            )

    message = (
        f"{name}: {n_bad} of {n_total} elements ({n_bad / n_total:.1%}) outside tolerance "
        f"rtol={rtol_:g} atol={atol_:g}\n"
        f"  worst at index {index}\n"
        f"    got      {got:.10g}\n"
        f"    expected {want:.10g}\n"
        f"    relative error {rel:.3e}\n"
        f"  max abs diff {float(diff.max()):.3e}, mean abs diff {float(diff.mean()):.3e}"
        f"{ratio_hint}"
    )
    if raise_on_fail:
        raise AssertionError(message)
    print(message)
    return False


def assert_close_across_ranks(
    tensor: torch.Tensor,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    name: str = "tensor",
    group=None,
) -> None:
    """Verify that a tensor which is supposed to be replicated really is.

    Use this on model parameters after construction, and on any activation you
    believe is replicated across a tensor-parallel group. It is the direct
    detector for the desync bug in Lab H hazard 2: weights initialised
    independently per rank, never broadcast, loss falls anyway, model is
    nonsense.

    Group rank 0's copy is the reference. Every other member compares against
    it and the failure names the offending global rank.
    """
    from common import dist_utils

    if not dist_utils.is_initialized():
        return

    import torch.distributed as dist

    group_world = dist.get_world_size(group=group)
    if group_world == 1:
        return

    source = 0 if group is None else dist.get_global_rank(group, 0)
    reference = tensor.detach().clone()
    dist.broadcast(reference, src=source, group=group)

    diff = (tensor.detach() - reference).abs()
    if tensor.is_floating_point() or tensor.is_complex():
        finite = torch.isfinite(tensor.detach()) & torch.isfinite(reference)
        has_nonfinite = not bool(finite.all())
        finite_diff = diff[finite]
    else:
        has_nonfinite = False
        finite = torch.ones_like(tensor, dtype=torch.bool)
        finite_diff = diff.reshape(-1)
    max_diff = float(finite_diff.max()) if finite_diff.numel() else 0.0
    tolerances = atol + rtol * reference.abs()
    bad_elements = torch.zeros_like(finite)
    bad_elements[finite] = diff[finite] > tolerances[finite]

    disagrees = has_nonfinite or not math.isfinite(max_diff) or bool(bad_elements.any())
    backend = str(dist.get_backend(group=group)).lower()
    control_device = tensor.device if backend == "nccl" else torch.device("cpu")
    local_bad = torch.tensor(
        [1.0 if disagrees else 0.0],
        dtype=torch.float32,
        device=control_device,
    )
    dist.all_reduce(local_bad, op=dist.ReduceOp.SUM, group=group)

    if local_bad.item() > 0:
        difference = "non-finite values" if has_nonfinite else f"{max_diff:.3e}"
        raise AssertionError(
            f"{name} differs across ranks relative to global rank {source}. "
            f"On global rank {dist_utils.rank()}, the maximum difference is {difference} "
            f"(elementwise rtol={rtol:g}, atol={atol:g}); "
            f"{int(local_bad.item())} of {group_world} group members disagree.\n"
            "  If this is a model parameter, you probably forgot "
            "dist_utils.sync_module_params(model) in your wrapper's constructor.\n"
            "  Every rank has to start from identical weights or you are training N "
            "different models and averaging their gradients."
        )


def assert_curves_close(
    actual: list[float],
    expected: list[float],
    *,
    rtol: float = 2e-2,
    atol: float = 1e-3,
    name: str = "loss curve",
    first_n: int | None = None,
) -> None:
    """Compare two loss curves with a tolerance that grows with the step index.

    Early steps should match tightly: the models start identical and have seen
    identical data. Later steps drift because floating-point error compounds
    through the optimizer, so the tolerance loosens linearly. That is not
    hand-waving, it is the actual behaviour of two numerically-equivalent
    training runs, and pretending otherwise produces a flaky test.
    """
    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: {len(actual)} steps vs {len(expected)}. "
            "Both runs need the same number of optimizer steps to be comparable."
        )

    n = first_n if first_n is not None else len(actual)
    for step in range(min(n, len(actual))):
        got, want = actual[step], expected[step]
        if not math.isfinite(got) or not math.isfinite(want):
            raise AssertionError(
                f"{name} contains a non-finite value at step {step}: "
                f"got {got!r}, expected {want!r}."
            )
        drift = 1.0 + step * 0.05  # 5% looser per step
        tolerance = atol + rtol * drift * abs(want)
        if abs(got - want) > tolerance:
            preview = min(8, len(actual))
            raise AssertionError(
                f"{name} diverges at step {step}: got {got:.6f}, expected {want:.6f} "
                f"(tolerance {tolerance:.6f}).\n"
                f"  actual   {[round(v, 4) for v in actual[:preview]]}\n"
                f"  expected {[round(v, 4) for v in expected[:preview]]}\n"
                "  Diverging at step 0 means the models started different: check that you "
                "broadcast parameters from rank 0.\n"
                "  Matching at step 0 and diverging later means the gradients are wrong: "
                "check the division by world size, and check that every async handle was "
                "waited on before optimizer.step()."
            )


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def deterministic_context(seed: int = 1337, *, warn_only: bool = True) -> Iterator[None]:
    """Make a comparison run reproducible, and say so when it cannot be.

    Sets seeds, turns off TF32, and asks torch for deterministic kernels. Some
    ops have no deterministic implementation at all. Rather than dying with a
    cryptic ``RuntimeError`` deep inside a backward pass, this catches that and
    prints which op is responsible, so you know to loosen the tolerance rather
    than hunt a bug that is not there.

    TF32 deserves a note. On Ampere and later, torch defaults matmul to TF32,
    which keeps 10 mantissa bits instead of 23. That is fine for training and
    fatal for a 1e-6 comparison. Any test comparing against a reference should
    be inside this context.
    """
    from common import dist_utils

    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    # Required by torch for deterministic cuBLAS reductions.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    dist_utils.seed_everything(seed, rank_offset=False)

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as exc:  # noqa: BLE001
        print(f"note: deterministic algorithms unavailable ({exc}). Loosen tolerances.")

    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(
                prev_deterministic, warn_only=prev_deterministic_warn_only
            )
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        if prev_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas


# --------------------------------------------------------------------------- #
# pytest helpers
# --------------------------------------------------------------------------- #


def requires_cuda_devices(n: int = 2) -> Callable:
    """Skip a test unless ``n`` CUDA devices are visible, with a useful reason."""
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return pytest.mark.skipif(
        available < n,
        reason=(
            f"needs {n} separately addressable CUDA devices, found {available}. "
            "These exercises assign one process rank to each CUDA device. "
            "Use the Kaggle notebook when a two-GPU accelerator is available, "
            "or run in another multi-GPU environment."
        ),
    )


def requires_min_ranks(n: int) -> Callable:
    """Skip when spawning ``n`` CPU ranks would oversubscribe the machine badly."""
    cores = os.cpu_count() or 1
    return pytest.mark.skipif(
        n > cores * 2,
        reason=f"{n} ranks on {cores} cores would thrash; skipping rather than timing out",
    )


def gather_object(obj: Any) -> list[Any]:
    """All-gather an arbitrary picklable object. Returns one entry per rank."""
    from common import dist_utils

    if not dist_utils.is_initialized() or dist_utils.world_size() == 1:
        return [obj]

    import torch.distributed as dist

    buckets: list[Any] = [None] * dist_utils.world_size()
    dist.all_gather_object(buckets, obj)
    return buckets


def spawn_case(world_size: int, *, timeout_s: int = 120) -> Callable:
    """Decorator: run a module-level worker across ``world_size`` gloo ranks.

    ::

        def _worker():
            ...assertions...

        @spawn_case(2)
        def test_thing():
            return _worker

    The indirection exists because ``spawn`` pickles the target, so the worker
    has to be a module-level function rather than the test body itself.
    """

    def decorator(fn: Callable[[], Callable]) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from common import dist_utils

            worker = fn(*args, **kwargs)
            return dist_utils.run_distributed(
                worker, world_size=world_size, backend="gloo", timeout_s=timeout_s
            )

        return wrapper

    return decorator
