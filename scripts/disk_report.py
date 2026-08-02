#!/usr/bin/env python3
"""Per-directory footprint, checked against the budget in docs/FOOTPRINT.md.

The budget table in the docs starts out as estimates. ``--write`` replaces the
measured column with real numbers from this machine, which turns the modularity
claim from a promise into a record.

::

    python tasks.py disk
    python tasks.py disk WRITE=1
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

SKIP_DIRS = {
    ".venv",
    ".cache",
    "runs",
    "data",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".agents",
    ".codex",
    "plans",
    "workspace",
    "graphify-out",
    "traces",
    "build",
    "dist",
}


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def dir_bytes(path: Path, *, exclude: set[str] | None = None) -> int:
    if not path.exists():
        return 0
    exclude = exclude or set()
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".venv-")]
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def _venv_row() -> tuple[str, int, int, str]:
    """Describe the installed environment rather than assuming it is CUDA."""
    root = repo_env.repo_root()
    venv = root / ".venv"
    size = dir_bytes(venv)
    if not venv.exists():
        return (".venv", 0, 6 * 1024**3, "not created")

    python = repo_env.venv_python()
    if python.is_file():
        code = "import torch; print(torch.__version__); print(torch.version.cuda or 'cpu-only')"
        try:
            result = subprocess.run(
                [str(python), "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            details = result.stdout.strip().splitlines()
            version = details[0] if details else "unknown"
            runtime = details[1] if len(details) > 1 else "unknown"
            if runtime != "cpu-only":
                return (
                    ".venv, GPU path",
                    size,
                    6 * 1024**3,
                    f"torch {version}, CUDA runtime {runtime}",
                )
            return (
                ".venv, CPU-only path",
                size,
                1024**3,
                f"CPU-only torch {version}",
            )

    return (".venv", size, 6 * 1024**3, "torch profile could not be identified")


def _matching_bytes(path: Path, prefix: str) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.iterdir()
        if item.is_file() and item.name.startswith(prefix)
    )


def measure() -> list[tuple[str, int, int, str]]:
    root = repo_env.repo_root()
    data = root / "data"
    # The generic generated `runs/` directory is excluded, but completed
    # reference runs under answers/ are publishable source artefacts and must
    # count toward the source budget.
    source_bytes = dir_bytes(root, exclude=SKIP_DIRS)
    source_bytes += sum(dir_bytes(answer_runs) for answer_runs in (root / "answers").glob("*/runs"))
    # workspace/ is learner-owned and ignored except for its published guide
    # and one placeholder per expected lab.
    workspace = root / "workspace"
    source_bytes += sum(
        path.stat().st_size
        for path in [workspace / "README.md", *workspace.glob("*/.gitkeep")]
        if path.is_file()
    )
    return [
        (
            "source + docs",
            source_bytes,
            10 * 1024**2,
            "project files, including published answers",
        ),
        _venv_row(),
        (
            "data/",
            _matching_bytes(data, "tinyshakespeare"),
            6 * 1024**2,
            "tiny-shakespeare source and content-addressed token caches",
        ),
        (
            "data/ with TinyStories",
            _matching_bytes(data, "tinystories"),
            500 * 1024**2,
            "optional corpus; zero when it has not been downloaded",
        ),
        (
            "runs/",
            dir_bytes(root / "runs"),
            700 * 1024**2,
            "checkpoints, plots, and profiler traces",
        ),
        (".cache/", dir_bytes(root / ".cache"), 1024**3, "purgeable at any time"),
    ]


def write_footprint_doc(rows: list[tuple[str, int, int, str]]) -> Path | None:
    """Update only table rows that exactly match a measured item.

    Exact matching matters because the document has separate CPU/GPU venv and
    checkpoint/trace rows. A local run can measure only the active venv profile,
    and the filesystem does not reliably distinguish arbitrary run artefacts.
    """
    doc = repo_env.repo_root() / "docs" / "FOOTPRINT.md"
    if not doc.exists():
        print(f"  {doc} does not exist; nothing to update.")
        return None

    text = doc.read_text(encoding="utf-8")

    def normalise(label: str) -> str:
        return label.replace("`", "").replace("**", "").strip().lower()

    measured = {normalise(label): human(size) for label, size, _b, _n in rows}
    total = sum(size for _label, size, _budget, _note in rows)
    if any(label == ".venv, GPU path" for label, *_rest in rows):
        measured["total, gpu path"] = human(total)
    elif any(label == ".venv, CPU-only path" for label, *_rest in rows):
        measured["total, cpu-only path"] = human(total)

    def replace_row(match: re.Match) -> str:
        label = normalise(match.group(1))
        value = measured.get(label)
        if value is not None:
            cells = match.group(0).split("|")
            if len(cells) >= 4:
                cells[3] = f" {value} "
                return "|".join(cells)
        return match.group(0)

    updated = re.sub(r"^\|\s*([^|]+?)\s*\|[^\n]*\|$", replace_row, text, flags=re.MULTILINE)
    import datetime

    stamp = datetime.date.today().isoformat()
    updated = re.sub(
        r"(<!-- measured:).*?(-->)",
        rf"\1 {stamp} on {sys.platform} \2",
        updated,
    )
    doc.write_text(updated, encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="update the measured column in docs/FOOTPRINT.md"
    )
    args = parser.parse_args(argv)

    root = repo_env.repo_root()
    rows = measure()

    print(f"\nFootprint of {root}")
    print("-" * 74)
    print(f"{'item':<24}{'measured':>12}{'budget':>12}{'':>6}  note")
    print("-" * 74)

    total = 0
    over = []
    for label, size, budget, note in rows:
        total += size
        status = "ok" if size <= budget else "OVER"
        if size > budget:
            over.append(label)
        print(f"{label:<24}{human(size):>12}{human(budget):>12}{status:>6}  {note}")

    print("-" * 74)
    print(f"{'total':<24}{human(total):>12}")

    usage = shutil.disk_usage(root)
    print(f"\n  free on this filesystem   {human(usage.free)}")
    print(f"  repo is {total / usage.total * 100:.3f}% of the volume")

    if over:
        print("\n  Over budget: " + ", ".join(over))
        print("  `python tasks.py clean` drops runs/ and .cache/ without touching .venv.")

    print(
        "\n  Every measured path above is inside the repository. Managed commands"
        "\n  also direct their known third-party caches into .cache/."
    )

    if args.write:
        written = write_footprint_doc(rows)
        if written:
            print(f"\n  updated {written.relative_to(root)}")

    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
