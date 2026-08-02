#!/usr/bin/env python3
"""HAZARD 3 — The silent wrong gradient.

Symptom
-------
Training works. The loss falls. Every rank agrees about the model. Nothing errors.

The loss just falls a bit more slowly than it should, and you would never notice
unless you had a correct run to compare against.

::

    python labs/lab_hazards/hazard_3_silent_gradient/broken.py

This one runs both the broken version and a correct single-process reference, so
the gap is visible. In real life nobody hands you the reference, which is why
this hazard is the one that survives longest in production code.

Your job
--------
Find why the gradients are wrong, fix it in place, and make ``test_fixed.py``
pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common import repo_env  # noqa: E402

repo_env.apply()

import torch  # noqa: E402

from common import collectives, dist_utils  # noqa: E402

STEPS = 40
HIDDEN = 512  # large enough that the all-reduce takes real time


def build_model() -> torch.nn.Module:
    dist_utils.seed_everything(1337, rank_offset=False)
    model = torch.nn.Sequential(
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 8),
    )
    dist_utils.sync_module_params(model)
    return model


def sync_gradients_fast(model: torch.nn.Module) -> None:
    """Overlap the gradient all-reduces with each other for speed.

    Issuing every collective asynchronously and letting them run in parallel is a
    real optimisation and this is roughly how you would start writing it.
    """
    world = dist_utils.world_size()
    handles = []

    for param in model.parameters():
        if param.grad is None:
            continue
        handle = collectives.all_reduce(param.grad, async_op=True)
        handles.append(handle)
        param.grad /= world

    # All collectives are in flight. Carry on.


def worker() -> dict:
    rank, world = dist_utils.rank(), dist_utils.world_size()

    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    generator = torch.Generator().manual_seed(11)
    x_all = torch.randn(64, HIDDEN, generator=generator)
    y_all = torch.randn(64, 8, generator=generator)

    per_rank = 64 // world
    x = x_all[rank * per_rank : (rank + 1) * per_rank]
    y = y_all[rank * per_rank : (rank + 1) * per_rank]

    losses = []
    for _step in range(STEPS):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sync_gradients_fast(model)
        optimizer.step()
        losses.append(float(loss.detach()))

    return {"rank": rank, "losses": losses, "final_weight": float(model[0].weight.detach().sum())}


def reference_losses() -> list[float]:
    """The same training in one process on the full batch. Ground truth."""
    dist_utils.seed_everything(1337, rank_offset=False)
    model = torch.nn.Sequential(
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 8),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    generator = torch.Generator().manual_seed(11)
    x = torch.randn(64, HIDDEN, generator=generator)
    y = torch.randn(64, 8, generator=generator)

    losses = []
    for _step in range(STEPS):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> int:
    print(__doc__.split("Your job")[0])
    print("running 4 distributed ranks and a single-process reference ...\n")

    results = dist_utils.run_distributed(worker, world_size=4, backend="gloo", timeout_s=90)
    reference = reference_losses()
    distributed = results[0]["losses"]

    print(f"  {'step':>6}{'distributed':>14}{'reference':>14}{'gap':>12}")
    for step in (0, 5, 10, 20, STEPS - 1):
        gap = distributed[step] - reference[step]
        print(f"  {step:>6}{distributed[step]:>14.6f}{reference[step]:>14.6f}{gap:>+12.6f}")

    final_gap = abs(distributed[-1] - reference[-1])
    relative = final_gap / max(reference[-1], 1e-12)

    print(f"\n  final relative gap: {relative:.1%}")

    if relative > 0.02:
        print("\n" + "=" * 70)
        print("THE GRADIENTS ARE WRONG.")
        print("=" * 70)
        print("\n  Both runs converge. Neither raises. The distributed one is just")
        print("  slightly worse, forever, and the curve looks completely normal on")
        print("  its own.")
        print("\n  Note how small the signal is. If you did not have the reference")
        print("  sitting next to it you would call this a healthy run, and you would")
        print("  spend the next week tuning the learning rate.")
        print("\n  Hypothesis in POSTMORTEM.md, then investigate.")
        return 1

    print("\n  the two curves agree. Fixed. Now run:")
    print("    python -m pytest labs/lab_hazards/hazard_3_silent_gradient/test_fixed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
