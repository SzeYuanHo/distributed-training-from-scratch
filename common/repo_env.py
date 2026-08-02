"""Redirect known third-party caches into the repository.

Several third-party libraries in this stack write caches in your home directory
by default. Hugging Face writes to ``~/.cache/huggingface``, torch to
``~/.cache/torch``, Triton to ``~/.triton``, matplotlib to ``~/.config/matplotlib``,
pip to ``~/.cache/pip``. None of those go away when you delete the repo, and a
year later you have gigabytes of orphaned wheels you cannot attribute to anything.

This module points the caches used by the managed workflow at ``<repo>/.cache``
instead, so ``make nuke`` reclaims those managed artefacts. System tools and a
third-party library that ignores its documented cache variable remain outside
the scope of this module; ``scripts/verify_clean.py`` watches the usual fallback
locations for unexpected growth.

Import it before torch. ``conftest.py`` and ``tasks.py`` both do that, and
``activate.sh`` / ``activate.ps1`` set the same variables for interactive shells.

Nothing here imports torch, numpy, or anything else outside the standard
library, because it has to run before the venv exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "repo_root",
    "cache_dir",
    "cache_env",
    "apply",
    "is_active",
    "parse_positive_int",
    "positive_int_env",
    "recommended_omp_threads",
    "venv_python",
]

_MARKER_FILES = ("pyproject.toml", "tasks.py")
_CACHE_PATH_KEYS = (
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "MPLCONFIGDIR",
    "WANDB_DIR",
)


def parse_positive_int(value: str | int, *, name: str) -> int:
    """Parse a positive integer with a message that names the bad setting."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from None
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return parsed


def positive_int_env(name: str, default: int) -> int:
    """Read a positive-integer environment variable or return ``default``."""
    default = parse_positive_int(default, name=f"default for {name}")
    return parse_positive_int(os.environ.get(name, default), name=name)


def repo_root(start: Path | None = None) -> Path:
    """Walk up from this file until we find the directory holding pyproject.toml."""
    here = (start or Path(__file__).resolve()).parent
    for candidate in (here, *here.parents):
        if all((candidate / marker).exists() for marker in _MARKER_FILES):
            return candidate
    # Fall back to the parent of common/, which is right for a normal checkout.
    return Path(__file__).resolve().parent.parent


def cache_dir() -> Path:
    return repo_root() / ".cache"


def recommended_omp_threads(nproc: int) -> int:
    """Threads per rank so that N ranks do not each grab every core.

    Without this, launching 4 CPU ranks on a 16-core laptop starts 64 OpenMP
    threads that fight over 16 cores, and a two-second step can take far longer.
    """
    cores = os.cpu_count() or 1
    return max(1, cores // max(1, nproc))


def cache_env(nproc: int | None = None) -> dict[str, str]:
    """Return the environment variables that keep every cache inside the repo.

    Pass ``nproc`` when you are about to launch that many ranks and want
    ``OMP_NUM_THREADS`` sized accordingly.
    """
    root = repo_root()
    cache = root / ".cache"

    env = {
        "DTFS_REPO_ROOT": str(root),
        "DTFS_CACHE_ACTIVE": "1",
        # Generic XDG location. Several libraries respect this and nothing else.
        "XDG_CACHE_HOME": str(cache),
        # Hugging Face: hub downloads and datasets.
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        # torch.hub checkpoints and kernel caches.
        "TORCH_HOME": str(cache / "torch"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        # Installers.
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        # matplotlib writes a font cache on first import and warns loudly if it
        # cannot. Point it here so the first plot does not print a wall of text.
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        # Weights & Biases stays offline by default: no account, no network,
        # no surprise uploads from a teaching repo.
        "WANDB_DIR": str(cache / "wandb"),
        "WANDB_MODE": "offline",
        # Silences a warning that fires once per spawned rank.
        "TOKENIZERS_PARALLELISM": "false",
    }

    if nproc is not None:
        nproc = parse_positive_int(nproc, name="nproc")
        env["DTFS_NPROC"] = str(nproc)
        env["OMP_NUM_THREADS"] = str(recommended_omp_threads(nproc))
        env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]

    return env


def apply(
    nproc: int | None = None, *, create: bool = True, override: bool = True
) -> dict[str, str]:
    """Set the cache variables on ``os.environ`` and create the directories.

    Managed commands force these paths into ``<repo>/.cache`` by default. This
    fail-closed behaviour is what makes their footprint removable with the
    repository. ``override=False`` is available for an interactive caller that
    intentionally manages its own environment; :func:`is_active` will then
    report false unless every resulting path still matches the managed layout.
    """
    env = cache_env(nproc)
    for key, value in env.items():
        if override or key not in os.environ:
            os.environ[key] = value

    if create:
        for key in _CACHE_PATH_KEYS:
            try:
                Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"could not create managed cache directory for {key}: {os.environ[key]}"
                ) from exc

    return env


def is_active() -> bool:
    """True when every managed cache points at its expected repo-local path.

    ``scripts/doctor.py`` reports this, because a run with redirection off is
    the one way this repo can leave residue behind.
    """
    expected = cache_env()
    if os.environ.get("DTFS_CACHE_ACTIVE") != "1":
        return False

    for key in _CACHE_PATH_KEYS:
        actual = os.environ.get(key)
        if not actual:
            return False
        try:
            if Path(actual).expanduser().resolve() != Path(expected[key]).resolve():
                return False
        except OSError:
            return False
    return True


def venv_python() -> Path:
    """Path to the interpreter inside ./.venv, whether or not it exists yet."""
    root = repo_root()
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"
