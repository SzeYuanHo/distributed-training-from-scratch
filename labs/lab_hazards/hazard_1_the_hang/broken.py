#!/usr/bin/env python3
"""HAZARD 1 — The hang.

Symptom
-------
You run this. It prints the startup lines. Then nothing. No error, no traceback,
no CPU activity. Ten minutes later it is still there.

::

    python labs/lab_hazards/hazard_1_the_hang/broken.py

Run it. Watch the silence. Then form a hypothesis and write it down in
POSTMORTEM.md *before* you read further.

Your job
--------
Find the cause with a tool rather than by reading this file, fix it in place, and
make ``test_fixed.py`` pass.

If you are stuck after fifteen minutes, HINT.md has three progressively less
subtle hints behind collapsible sections. Fifteen minutes of being stuck is worth
more than the hint.
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

STEPS = 10
TIMEOUT_S = 30  # short on purpose: a hang should fail, not sit there


def train_step(model: torch.nn.Module, optimizer, step: int) -> float:
    """One training step with gradient averaging across ranks."""
    rank = dist_utils.rank()

    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(step))
    y = torch.randn(8, 4, generator=torch.Generator().manual_seed(step + 1000))

    loss = torch.nn.functional.mse_loss(model(x), y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    for param in model.parameters():
        collectives.all_reduce(param.grad)
        param.grad /= dist_utils.world_size()

    # Log the gradient norm every few steps so we can see training progress.
    if step % 3 == 0 and rank == 0:
        total_norm = torch.zeros(1)
        for param in model.parameters():
            total_norm += param.grad.norm() ** 2
        collectives.all_reduce(total_norm)
        print(f"  step {step}: loss {float(loss):.4f}  grad norm {float(total_norm) ** 0.5:.4f}")

    optimizer.step()
    return float(loss.detach())


def worker() -> str:
    rank = dist_utils.rank()
    dist_utils.printr(f"starting, world size {dist_utils.world_size()}", only=0)

    torch.manual_seed(1337)
    model = torch.nn.Linear(16, 4)
    dist_utils.sync_module_params(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(STEPS):
        train_step(model, optimizer, step)

    dist_utils.printr("finished all steps", only=0)
    return "completed"


def main() -> int:
    print(__doc__.split("Your job")[0])
    print("launching 4 ranks with a 30 second timeout ...\n")

    try:
        results = dist_utils.run_distributed(
            worker,
            world_size=4,
            backend="gloo",
            timeout_s=TIMEOUT_S,
            join_timeout_s=TIMEOUT_S + 20,
        )
    except dist_utils.DistributedTimeout as exc:
        print("\n" + "=" * 70)
        print("TIMED OUT. This is the symptom.")
        print("=" * 70)
        print(f"\n{exc}\n")
        print("What to do next, in order:")
        print("  1. Write your hypothesis in POSTMORTEM.md. Now, before investigating.")
        print("  2. Re-run with TORCH_DISTRIBUTED_DEBUG=DETAIL and read the last")
        print("     collective each rank logged.")
        print("  3. Or run it again in another terminal and py-spy dump every rank.")
        print("     Find the rank that is NOT inside a collective.")
        print("  4. Only then read the source.")
        return 1

    print(f"\ncompleted: {results}")
    print("\nIf you are seeing this, the hazard is fixed. Now run:")
    print("  python -m pytest labs/lab_hazards/hazard_1_the_hang/test_fixed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
