#!/usr/bin/env python3
"""Check whether defined global cache locations grew during a managed run.

The managed workflow directs its known caches into the repository. One library
that ignores its cache variable can still seed a user-level cache, so this
script snapshots the usual fallback locations and checks them for growth.

::

    python scripts/verify_clean.py --snapshot     # before
    python -m pytest labs -m "not tier2"          # do the work
    python scripts/verify_clean.py --check        # after; nonzero on watched growth

Running it with no arguments just reports what currently exists in the usual
global locations, which is also useful the first time you clone the repo and
want to know what was already there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

HOME = Path.home()

# Places these libraries write when nobody redirects them.
WATCHED: list[tuple[str, Path]] = [
    ("huggingface hub", HOME / ".cache" / "huggingface"),
    ("torch hub", HOME / ".cache" / "torch"),
    ("triton", HOME / ".triton"),
    ("triton cache", HOME / ".cache" / "triton"),
    ("torchinductor", HOME / ".cache" / "torchinductor"),
    ("matplotlib", HOME / ".cache" / "matplotlib"),
    ("matplotlib config", HOME / ".config" / "matplotlib"),
    ("pip", HOME / ".cache" / "pip"),
    ("uv", HOME / ".cache" / "uv"),
    ("wandb", HOME / ".wandb"),
    ("conda envs", HOME / "miniconda3" / "envs"),
    ("conda pkgs", HOME / "miniconda3" / "pkgs"),
]

if sys.platform == "win32":
    local = Path(os.environ.get("LOCALAPPDATA", HOME / "AppData" / "Local"))
    WATCHED += [
        ("windows torch cache", local / "torch"),
        ("windows huggingface", local / "huggingface"),
        ("windows pip", local / "pip" / "Cache"),
        ("windows uv", local / "uv" / "cache"),
        ("windows matplotlib", local / "matplotlib"),
    ]


def snapshot_path() -> Path:
    path = repo_env.repo_root() / ".cache" / "verify_clean_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scan() -> dict[str, dict[str, int]]:
    """Record size and file count for each watched location."""
    out: dict[str, dict[str, int]] = {}
    for label, path in WATCHED:
        if not path.exists():
            out[label] = {"exists": 0, "bytes": 0, "files": 0}
            continue
        size = files = 0
        for dirpath, _dirs, names in os.walk(path):
            for name in names:
                try:
                    size += (Path(dirpath) / name).stat().st_size
                    files += 1
                except OSError:
                    pass
        out[label] = {"exists": 1, "bytes": size, "files": files}
    return out


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"


def report(state: dict[str, dict[str, int]]) -> None:
    print(f"\nGlobal cache locations (outside {repo_env.repo_root()})")
    print("-" * 68)
    any_present = False
    for label, path in WATCHED:
        info = state.get(label, {})
        if not info.get("exists"):
            continue
        any_present = True
        print(f"  {label:<24}{human(info['bytes']):>10}  {info['files']:>6} files   {path}")
    if not any_present:
        print("  none of the watched locations exist.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snapshot", action="store_true", help="record the current state")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the snapshot and exit nonzero on growth",
    )
    parser.add_argument(
        "--tolerance-bytes",
        type=int,
        default=1024 * 1024,
        help="ignore growth below this (default 1 MiB)",
    )
    args = parser.parse_args(argv)

    current = scan()

    if args.snapshot:
        snapshot_path().write_text(json.dumps(current, indent=2), encoding="utf-8")
        report(current)
        print(f"  snapshot written to {snapshot_path()}")
        return 0

    if not args.check:
        report(current)
        print("  Run with --snapshot before a test run and --check after it to see")
        print("  whether anything leaked out of the repository.")
        return 0

    path = snapshot_path()
    if not path.exists():
        print(f"  no snapshot at {path}. Run --snapshot first.")
        return 2

    before = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for label, after in current.items():
        prior = before.get(label, {"bytes": 0, "files": 0, "exists": 0})
        grew = after["bytes"] - prior.get("bytes", 0)
        new_files = after["files"] - prior.get("files", 0)
        if grew > args.tolerance_bytes or new_files > 0:
            problems.append((label, grew, new_files))

    if not problems:
        print("\n  No watched global cache location grew beyond the configured tolerance.")
        print("  The managed-cache check passed on this machine.\n")
        return 0

    print("\n  WATCHED CACHE GROWTH DETECTED:\n")
    for label, grew, new_files in problems:
        print(f"    {label:<24} +{human(grew):>10}  +{new_files} files")
    print(
        "\n  Usual cause: a library that ignores the variables set in "
        "common/repo_env.py.\n"
        "  Find which one by re-running with that library's own cache variable set to\n"
        "  a temporary directory, then add it to cache_env().\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
