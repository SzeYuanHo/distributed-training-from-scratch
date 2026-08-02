"""Pytest configuration for the whole repository.

Three jobs:

1. Redirect caches and put the repo root on ``sys.path`` before torch loads.
2. Register the tier markers so ``--strict-markers`` stays useful.
3. Skip Tier-2 tests automatically when fewer than two CUDA devices exist,
   with a message that says what to do instead.

The import order at the top matters. ``repo_env.apply()`` has to run before
anything imports torch, or torch will already have decided where its cache
lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import repo_env  # noqa: E402

repo_env.apply(nproc=repo_env.positive_int_env("DTFS_NPROC", 2))

import pytest  # noqa: E402


def pytest_configure(config):
    # Mirrors the markers in pyproject.toml. Duplicated on purpose: a learner
    # who copies conftest.py into a scratch directory should still get them.
    for marker, description in [
        ("tier0", "single process, no torch.distributed at all"),
        ("tier1", "multi-process on gloo/CPU, correctness and semantics"),
        ("tier2", "needs >= 2 CUDA devices, skipped automatically otherwise"),
        ("slow", "takes more than 60 seconds"),
        ("negative", "passes by demonstrating a silent-failure mode"),
    ]:
        config.addinivalue_line("markers", f"{marker}: {description}")


def _cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def pytest_collection_modifyitems(config, items):
    n_devices = _cuda_device_count()
    if n_devices >= 2:
        return

    reason = (
        f"Tier 2 needs 2+ CUDA devices; found {n_devices}. "
        "This curriculum follows PyTorch's one-process-per-GPU model, and "
        "sharing a GPU between NCCL processes can deadlock or report invalid usage. "
        "Use a Linux environment with 2+ visible GPUs; the included Kaggle notebook "
        "describes one hosted option when a multi-GPU accelerator is available: "
        "notebooks/kaggle_2xT4_tier2.ipynb"
    )
    skip_tier2 = pytest.mark.skip(reason=reason)
    for item in items:
        if "tier2" in item.keywords:
            item.add_marker(skip_tier2)


@pytest.fixture
def tmp_runs_dir(tmp_path, monkeypatch):
    """Point a test's artefact output at tmp_path instead of runs/."""
    monkeypatch.setenv("DTFS_RUNS_DIR", str(tmp_path))
    return tmp_path
