#!/usr/bin/env python3
"""Run the lab suite against the published answers in an isolated repository copy.

The exercise tests intentionally import ``labs/*/starter``.  The completed
implementations intentionally live under ``answers/``.  This script reconciles
those two contracts without overwriting a learner's working tree:

1. copy the repository to a temporary directory;
2. overlay each published answer onto its matching exercise file;
3. run pytest inside the temporary copy; and
4. delete the copy on exit.

Arguments after the script name are forwarded to pytest.  With no arguments it
runs every non-Tier-2 lab test.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB_NAMES = (
    "lab0_collectives",
    "lab1_ddp_bucketing",
    "lab1b_zero_memory",
    "lab2_tensor_parallel",
    "lab3_pipeline_parallel",
    "lab4_capstone_2d",
)
COPY_EXCLUDES = {
    ".git",
    ".venv",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "data",
    "graphify-out",
    "plans",
    ".agents",
    ".codex",
    "workspace",
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in COPY_EXCLUDES
        or name.startswith(".venv-")
        or name.endswith((".pyc", ".pyo", ".whl", ".egg-info"))
    }


def _copy_completed_answers(destination: Path) -> None:
    missing: list[str] = []

    for lab in LAB_NAMES:
        source = ROOT / "answers" / lab / "starter"
        target = destination / "labs" / lab / "starter"
        if not source.is_dir():
            missing.append(str(source.relative_to(ROOT)))
            continue
        shutil.copytree(source, target, dirs_exist_ok=True)

    answer_hazards = ROOT / "answers" / "lab_hazards"
    exercise_hazards = destination / "labs" / "lab_hazards"
    for exercise in sorted(exercise_hazards.glob("hazard_*")):
        source = answer_hazards / exercise.name / "broken.py"
        if not source.is_file():
            missing.append(str(source.relative_to(ROOT)))
            continue
        shutil.copy2(source, exercise / "broken.py")

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"published answer set is incomplete:\n{formatted}")


def _copy_reference_data(destination: Path) -> None:
    """Copy only the verified small corpus needed for offline answer tests."""
    source_dir = ROOT / "data"
    source = source_dir / "tinyshakespeare.txt"
    if not source.is_file():
        return

    from common.data import SHAKESPEARE_SHA256

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SHAKESPEARE_SHA256:
        return

    target_dir = destination / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_dir / source.name)

    token_cache = source_dir / f"tinyshakespeare.{digest[:16]}.uint16.npy"
    if token_cache.is_file():
        shutil.copy2(token_cache, target_dir / token_cache.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab",
        choices=(*LAB_NAMES, "lab_hazards"),
        help="run only this lab directory (default: all labs)",
    )
    args, pytest_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not pytest_args:
        pytest_args = ["-m", "not tier2"]

    with tempfile.TemporaryDirectory(prefix="dtfs-answers-") as temporary:
        checkout = Path(temporary) / "repository"
        shutil.copytree(ROOT, checkout, ignore=_ignore)
        _copy_reference_data(checkout)
        _copy_completed_answers(checkout)

        env = os.environ.copy()
        env["DTFS_REPO_ROOT"] = str(checkout)
        env["PYTHONPATH"] = str(checkout)

        target = str(Path("labs") / args.lab) if args.lab else "labs"
        command = [sys.executable, "-m", "pytest", target, *pytest_args]
        print(f"  reference checkout: {checkout}")
        print(f"  command: {' '.join(command)}")
        completed = subprocess.run(command, cwd=checkout, env=env, check=False)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
