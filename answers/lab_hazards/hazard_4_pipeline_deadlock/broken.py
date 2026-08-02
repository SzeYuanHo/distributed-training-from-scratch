#!/usr/bin/env python3
"""HAZARD 4 — The pipeline deadlock.

Symptom
-------
The pipeline stages exchange activations and the job never finishes. Depending on
your transport, it either hangs until something times out or a rank reports
"Timed out waiting Nms for send operation".

::

    python labs/lab_hazards/hazard_4_pipeline_deadlock/broken.py

The script tries three message sizes in order and reports where it stops.

**Watch which size it stops at, and note it.** The result is specific to the
backend, build and transport. The local gloo configuration used while preparing
this lab blocked at the first size. A transport with an *eager* protocol may copy
a small payload into an internal buffer and let that case pass before a larger
payload exposes the same unsafe ordering. There is no portable threshold, and
this exercise should not be used to infer NCCL's host-side completion semantics.

That second case is what makes this bug dangerous in the wild: the same code
passes every test you wrote against small tensors and fails the first time you
run it on real data.

Your job
--------
Explain why the deadlock happens, why some transports hide it at small sizes, fix
it, and make ``test_fixed.py`` pass.

The "why do some transports hide it" half is the one worth thinking about. It
generalises well past this bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common import repo_env  # noqa: E402

repo_env.apply()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from common import collectives, dist_utils  # noqa: E402

HIDDEN = 256
SIZES_TO_TRY = [1, 8, 64]


def exchange_activations(tensor: torch.Tensor) -> torch.Tensor:
    """Pass an activation to the next stage and receive from the previous one.

    Each rank is a pipeline stage. Stage i sends downstream to i+1 and receives
    upstream from i-1, wrapping around so every rank does both.
    """
    rank, world = dist_utils.rank(), dist_utils.world_size()
    send_to = (rank + 1) % world
    recv_from = (rank - 1) % world

    inbox = torch.empty_like(tensor)

    # Hand both halves to the backend at once so it can order them. Odd/even
    # ordering would also work at world 4 and breaks at world 3, where ranks 0
    # and 2 are neighbours and both even.
    ops = [
        collectives.P2POp(dist.isend, tensor, send_to),
        collectives.P2POp(dist.irecv, inbox, recv_from),
    ]
    for work in collectives.batch_isend_irecv(ops):
        work.wait()

    return inbox


def worker(batch: int) -> dict:
    rank = dist_utils.rank()

    activation = torch.full((batch, HIDDEN), float(rank + 1))
    received = exchange_activations(activation)

    expected_sender = (rank - 1) % dist_utils.world_size()
    ok = bool(torch.all(received == expected_sender + 1))

    return {
        "rank": rank,
        "batch": batch,
        "correct": ok,
        "bytes": activation.numel() * activation.element_size(),
    }


def main() -> int:
    print(__doc__.split("Your job")[0])

    for batch in SIZES_TO_TRY:
        payload = batch * HIDDEN * 4
        print(
            f"\n  batch {batch:>3}  ({payload / 1024:.0f} KB per message) ... ", end="", flush=True
        )

        try:
            results = dist_utils.run_distributed(
                worker,
                world_size=4,
                backend="gloo",
                args=(batch,),
                timeout_s=20,
                join_timeout_s=35,
            )
        except Exception as exc:  # noqa: BLE001
            # A deadlock reaches us in one of two shapes depending on the
            # backend and the platform. Either the join deadline fires and we
            # get DistributedTimeout, or gloo's own send timeout trips first and
            # a child raises "Timed out waiting Nms for send operation". Same
            # bug, two messages, and neither is worth making the reader
            # distinguish.
            message = str(exc)
            looks_like_a_hang = isinstance(exc, dist_utils.DistributedTimeout) or (
                "timed out" in message.lower()
            )
            if not looks_like_a_hang:
                raise

            print("HUNG")
            print("\n" + "=" * 70)
            print(f"DEADLOCKED AT {payload / 1024:.0f} KB PER MESSAGE.")
            print("=" * 70)
            if payload > 1024:
                print("\n  Smaller messages went through. This one did not, and the")
                print("  code did not change between the runs. That size dependence")
                print("  is the clue.")
            else:
                print("\n  It stopped at the very first size, so this transport has no")
                print("  eager buffer at all: every send waits for a matching receive.")
                print("  On a transport that does buffer small messages, the first two")
                print("  sizes would have passed and only the third would hang. Same")
                print("  bug, and much harder to catch.")
            print("\n  What the backend reported:")
            for line in message.strip().splitlines()[-3:]:
                print(f"    {line.strip()[:100]}")
            print("\n  Hypothesis in POSTMORTEM.md, then investigate.")
            print("\n  Useful: labs/lab0_collectives/tests/test_01_semantics.py has a")
            print("  test called test_symmetric_send_deadlocks that demonstrates the")
            print("  same thing on purpose. Read it.")
            return 1

        if all(r["correct"] for r in results.values()):
            print("ok")
        else:
            print("WRONG DATA")
            return 1

    print("\n  every size completed. Fixed. Now run:")
    print("    python -m pytest labs/lab_hazards/hazard_4_pipeline_deadlock/test_fixed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
