#!/usr/bin/env python3
"""Deprecated compatibility entry point.

The repository's tests and benchmark drivers call
``common.dist_utils.run_distributed`` themselves.  Wrapping one of those files
in ``torchrun`` creates a second launch layer: every outer rank starts another
world of ranks, with conflicting rendezvous state and device ownership.

Use the task runner instead:

* ``python tasks.py bench LAB=1 NPROC=2`` for a benchmark;
* ``python tasks.py test LAB=1`` for Tier 0/1 tests; or
* ``python tasks.py test-tier2 LAB=1`` for Tier 2 tests.

For a custom file whose top-level code is intentionally rank-aware, invoke
``torchrun`` directly.  The standalone NCCL probe in the Kaggle notebook is one
such file; the bundled lab drivers are not.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lab", help="legacy option; use tasks.py bench/test instead")
    parser.add_argument("--script", help="legacy option; run self-spawning drivers directly")
    parser.add_argument("--nproc", type=int, default=2, help="legacy option")
    parser.add_argument("--tier", choices=["cpu", "gpu"], default="cpu", help="legacy option")
    parser.add_argument("--timeout", type=int, default=None, help="legacy option")
    parser.add_argument("--dry-run", action="store_true", help="legacy option")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    parser.parse_args(argv)

    parser.exit(
        2,
        "error: scripts/launch.py is deprecated because bundled tests and benchmarks "
        "spawn their own workers.\n"
        "  benchmark:  python tasks.py bench LAB=1 NPROC=2\n"
        "  Tier 0/1:   python tasks.py test LAB=1\n"
        "  Tier 2:     python tasks.py test-tier2 LAB=1\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
