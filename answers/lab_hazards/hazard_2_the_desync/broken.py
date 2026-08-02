#!/usr/bin/env python3
"""HAZARD 2 — The desync.

Symptom
-------
This one runs to completion. The loss falls smoothly from about 1.0 to about 0.05.
It looks like a healthy training run and you would put the curve in a report.

The saved model is garbage.

::

    python labs/lab_hazards/hazard_2_the_desync/broken.py

Run it. Look at the loss curve. Then look at the final check the script prints,
which is the only thing that reveals the problem.

Your job
--------
Work out why a run that converges produces a useless model. Fix it in place and
make ``test_fixed.py`` pass.

This is the most dangerous shape of bug in distributed training: no exception, no
warning, and every metric you would normally look at says everything is fine.
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

STEPS = 60


def build_model() -> torch.nn.Module:
    """Build the model this rank will train."""
    # Init wants the same seed everywhere. Sampling and dropout want rank_offset,
    # and mixing the two up is the whole hazard. Broadcast anyway: seeding alike
    # is an assumption, sync_module_params is a guarantee.
    dist_utils.seed_everything(1337, rank_offset=False)

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    )
    dist_utils.sync_module_params(model)
    return model


def worker() -> dict:
    rank, world = dist_utils.rank(), dist_utils.world_size()

    model = build_model()
    # 0.05 only reached 0.59 of the starting loss once the ranks agreed, and
    # test_fixed wants half. Nudged up; see POSTMORTEM.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)

    # Fixed data, identical on every rank, so the only thing that can differ
    # between ranks is the model itself.
    x = torch.randn(32, 16, generator=torch.Generator().manual_seed(7))
    y = torch.randn(32, 4, generator=torch.Generator().manual_seed(8))

    losses = []
    for _step in range(STEPS):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Average gradients across ranks. This part is correct.
        for param in model.parameters():
            collectives.all_reduce(param.grad)
            param.grad /= world

        optimizer.step()
        losses.append(float(loss.detach()))

    first_weight = model[0].weight.detach()
    return {
        "rank": rank,
        "losses": losses,
        "weight_checksum": float(first_weight.sum()),
        "weight_head": first_weight.flatten()[:4].tolist(),
    }


def main() -> int:
    print(__doc__.split("Your job")[0])
    print("training 4 ranks for 60 steps ...\n")

    results = dist_utils.run_distributed(worker, world_size=4, backend="gloo", timeout_s=60)

    losses = results[0]["losses"]
    print(f"  loss:  {losses[0]:.4f} -> {losses[len(losses) // 2]:.4f} -> {losses[-1]:.4f}")
    print("  a clean, healthy-looking convergence curve.\n")

    print("  now the part nobody looks at:\n")
    checksums = {r: info["weight_checksum"] for r, info in sorted(results.items())}
    for rank, checksum in checksums.items():
        print(f"    rank {rank} first-layer weight sum: {checksum:+.6f}")

    spread = max(checksums.values()) - min(checksums.values())
    print(f"\n    spread across ranks: {spread:.6f}")

    if spread > 1e-5:
        print("\n" + "=" * 70)
        print("THE RANKS DISAGREE ABOUT WHAT THE MODEL IS.")
        print("=" * 70)
        print("\n  Four ranks. Four different models. Gradients averaged correctly")
        print("  across all of them, every step, which is exactly why the loss fell.")
        print("\n  Whichever rank saves the checkpoint saves a model that was")
        print("  optimised by gradients computed on three other models.")
        print("\n  Nothing raised. No warning. The loss curve is beautiful.")
        print("\n  Write your hypothesis in POSTMORTEM.md, then find the cause.")
        print("  common.testing.assert_close_across_ranks is the direct detector.")
        return 1

    print("\n  ranks agree. The hazard is fixed. Now run:")
    print("    python -m pytest labs/lab_hazards/hazard_2_the_desync/test_fixed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
