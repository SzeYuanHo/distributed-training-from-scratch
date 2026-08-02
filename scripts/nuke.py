#!/usr/bin/env python3
"""Delete everything this repository generated, and say how much came back.

Removes ``.venv/``, ``.cache/``, ``runs/``, ``data/``, and every
``__pycache__``. Leaves source, published ``answers/`` and private ``workspace/`` files.

Managed project artefacts and known third-party caches live in this folder.
System tools and unmanaged software are outside its scope.

::

    python tasks.py nuke
    python tasks.py nuke YES=1
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

TARGET_DIRS = [
    ".venv",
    ".cache",
    "runs",
    "traces",
    "data",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
]


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument(
        "--keep-data", action="store_true", help="keep data/ so you do not re-download 1.1 MB"
    )
    args = parser.parse_args(argv)

    root = repo_env.repo_root()

    # Refuse to run anywhere that does not look like this repo. A path bug here
    # would be unrecoverable, so the guard is worth the four lines.
    if not (root / "pyproject.toml").exists() or not (root / "tasks.py").exists():
        print(f"  {root} does not look like this repository. Refusing to delete anything.")
        return 2

    try:
        running_from_venv = (
            Path(sys.executable).resolve().is_relative_to((root / ".venv").resolve())
        )
    except OSError:
        running_from_venv = False
    if running_from_venv:
        print("  refusing to delete .venv while its Python interpreter is running.")
        print("  Deactivate it, then run with a system Python:")
        print("    python tasks.py nuke")
        return 2

    targets = [d for d in TARGET_DIRS if not (args.keep_data and d == "data")]
    plan = [(name, root / name, dir_bytes(root / name)) for name in targets]
    plan = [p for p in plan if p[1].exists()]

    pycaches = [p for p in root.rglob("__pycache__") if ".venv" not in p.parts]
    pycache_bytes = sum(dir_bytes(p) for p in pycaches)

    total = sum(size for _n, _p, size in plan) + pycache_bytes
    if total == 0 and not pycaches:
        print("  nothing to remove; already clean.")
        return 0

    print(f"\n  About to delete from {root}:\n")
    for name, _path, size in plan:
        print(f"    {name + '/':<20}{human(size):>12}")
    if pycaches:
        print(f"    {'__pycache__ x' + str(len(pycaches)):<20}{human(pycache_bytes):>12}")
    print(f"    {'-' * 32}")
    print(f"    {'total':<20}{human(total):>12}\n")
    print("  Kept: source, docs, labs/, published answers/, and private workspace/.")
    if not args.keep_data:
        print("  data/ goes too; re-download is one HTTP request and 1.1 MB.")

    if not args.yes:
        try:
            answer = input("\n  proceed? [y/N] ").strip().lower()
        except EOFError:
            print("  no tty; re-run with --yes.")
            return 1
        if answer not in ("y", "yes"):
            print("  cancelled.")
            return 0

    removed = 0
    failures: list[str] = []
    for name, path, size in plan:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            failures.append(f"{name}/: {exc}")
        remaining = dir_bytes(path)
        removed += max(0, size - remaining)
        if path.exists():
            print(f"    incomplete {name}/")
        else:
            print(f"    removed {name}/")
    for cache in pycaches:
        if not cache.exists():
            continue
        before = dir_bytes(cache)
        try:
            shutil.rmtree(cache)
        except OSError as exc:
            failures.append(f"{cache.relative_to(root)}: {exc}")
        removed += max(0, before - dir_bytes(cache))

    print(f"\n  reclaimed {human(removed)}")
    if failures:
        print("  Cleanup was incomplete:")
        for failure in failures:
            print(f"    - {failure}")
        print("  Close processes using those paths and retry; no clean-state claim was made.")
        return 1
    print("  To check watched global caches, bracket a managed workload with:")
    print("    python scripts/verify_clean.py --snapshot")
    print("    # run the workload")
    print("    python scripts/verify_clean.py --check")
    print("  Rebuild:")
    print("    python tasks.py setup-cpu\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
