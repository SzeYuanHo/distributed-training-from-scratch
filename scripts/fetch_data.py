#!/usr/bin/env python3
"""Download the dataset. Run by ``tasks.py setup``; safe to run again any time.

Default is tiny-shakespeare: 1.1 MB, one HTTP request, and enough to train
every model in this repo. TinyStories is available behind a flag with a hard
byte cap, because a teaching repo has no business filling your disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

repo_env.apply()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tinystories",
        action="store_true",
        help="also fetch the TinyStories validation split (500 MiB cap)",
    )
    parser.add_argument("--stats", action="store_true", help="print corpus statistics and exit")
    args = parser.parse_args(argv)

    from common import data

    path = data.ensure_shakespeare()
    dataset = data.load_dataset()
    print(f"  ready: {path}")
    print()
    print(data.dataset_stats(dataset))

    if args.stats:
        return 0

    if args.tinystories:
        import os

        os.environ["DTFS_ALLOW_TINYSTORIES"] = "1"
        data.ensure_tinystories()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
