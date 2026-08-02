#!/usr/bin/env python3
"""The repository's task runner. Every ``make`` target lands here.

Why Python instead of a Makefile? Because this repo targets a Windows laptop as
a first-class platform and GNU make is not installed there by default. Putting
the logic in one Python file means the Makefile, ``make.ps1``, and ``make.cmd``
are all three-line shims over the same code, and the behaviour cannot drift
between platforms.

Universal entry point, works anywhere Python does::

    python tasks.py doctor
    python tasks.py test LAB=0
    python tasks.py test-answers

POSIX convenience::

    make doctor

Windows convenience::

    .\\make.ps1 doctor
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import repo_env  # noqa: E402

ROOT = repo_env.repo_root()
VENV = ROOT / ".venv"

CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

LAB_DIRS = {
    "0": "lab0_collectives",
    "1": "lab1_ddp_bucketing",
    "1b": "lab1b_zero_memory",
    "2": "lab2_tensor_parallel",
    "3": "lab3_pipeline_parallel",
    "4": "lab4_capstone_2d",
    "h": "lab_hazards",
}
TIER2_LAB_NAMES = {
    "lab0_collectives",
    "lab1_ddp_bucketing",
    "lab2_tensor_parallel",
    "lab3_pipeline_parallel",
}


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


def _fail(message: str, code: int = 1) -> None:
    print(f"\n  error: {message}\n", file=sys.stderr)
    raise SystemExit(code)


def _run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    cwd: Path | None = None,
) -> int:
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}")
    merged = {**os.environ, **(env or {})}
    result = subprocess.run([str(c) for c in cmd], env=merged, cwd=str(cwd or ROOT))
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def _python() -> str:
    """Prefer the repo venv when it exists, otherwise whatever is running us."""
    candidate = repo_env.venv_python()
    return str(candidate) if candidate.exists() else sys.executable


def _uv() -> str:
    found = shutil.which("uv")
    if not found:
        _fail(
            "uv is not on PATH.\n"
            "    Linux/macOS/WSL:  curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh\n"
            "    Windows:          irm https://astral.sh/uv/0.11.16/install.ps1 | iex\n"
            "  For a manual locked install, follow docs/SETUP.md."
        )
    return found


def _has_cuda() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return out.returncode == 0 and "GPU 0" in out.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def _default_profile() -> str:
    """Pick cpu or gpu, and say why.

    On Windows the answer is CPU even when a GPU is present. Tier 1 is the main
    development loop and uses gloo, while NCCL is unavailable on native Windows.
    The CPU profile is the smaller, most portable default for that curriculum
    path. CUDA builds and gloo support can vary, so an explicit GPU install must
    be followed by ``doctor`` rather than assumed to work.

    `python tasks.py setup-gpu` overrides this if you want that profiling and
    are willing to check `doctor` afterwards.
    """
    cuda = _has_cuda()

    if sys.platform == "win32":
        if cuda:
            print("\n  CUDA detected, but installing the CPU profile on Windows.")
            print("  Tier 1 uses gloo and NCCL is unavailable on native Windows, so the")
            print("  smaller CPU environment is the most portable default. Run")
            print("  `python tasks.py setup-gpu` for local CUDA profiling, then use")
            print("  `python tasks.py doctor` to verify that this build still supports")
            print("  the Tier-1 path.")
        else:
            print("\n  No CUDA detected: installing the CPU profile.")
        return "cpu"

    if cuda:
        print("\n  CUDA detected: installing the GPU profile.")
        return "gpu"

    print("\n  No CUDA detected: installing the CPU profile.")
    print("  Tiers 0 and 1 run fully on CPU. That is most of the curriculum.")
    return "cpu"


def _lab_dir(lab: str | None) -> Path | None:
    if lab is None:
        return None
    key = str(lab).strip().lower()
    if key.startswith("lab"):
        key = key.removeprefix("lab")
    if key not in LAB_DIRS:
        _fail(f"unknown lab {lab!r}. Choose from: {', '.join(LAB_DIRS)}")
    return ROOT / "labs" / LAB_DIRS[key]


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TiB"


def _running_from(path: Path) -> bool:
    """Whether this task runner's interpreter lives under ``path``."""
    try:
        return Path(sys.executable).resolve().is_relative_to(path.resolve())
    except OSError:
        return False


def _relaunch_outside_venv() -> None:
    """Replace this process with its base interpreter before deleting .venv."""
    base = Path(getattr(sys, "_base_executable", ""))
    try:
        usable = (
            base.is_file()
            and not base.resolve().is_relative_to(VENV.resolve())
            and base.resolve() != Path(sys.executable).resolve()
        )
    except OSError:
        usable = False
    if not usable:
        _fail(
            "cannot safely delete .venv while its Python interpreter is running, "
            "and no external base interpreter was found.\n"
            "  Deactivate it, then run this command with a system Python."
        )

    print(f"  relaunching cleanup with base interpreter: {base}")
    try:
        os.execv(str(base), [str(base), str(ROOT / "tasks.py"), *sys.argv[1:]])
    except OSError as exc:
        _fail(f"could not relaunch cleanup with {base}: {exc}")


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #


def setup(opts: dict[str, str]) -> None:
    """Create ./.venv, sync a hashed dependency lock, and fetch the dataset."""
    profile = opts.get("PROFILE")
    if profile is None:
        profile = _default_profile()
    if profile not in {"cpu", "gpu"}:
        _fail(f"unknown setup profile {profile!r}; choose cpu or gpu")

    repo_env.apply()
    uv = _uv()

    if not VENV.exists():
        _run([uv, "venv", str(VENV)])
    else:
        print(f"  reusing existing venv at {VENV}")

    lock_name = "cu126.lock" if profile == "gpu" else "cpu.lock"
    lock = ROOT / "requirements" / lock_name
    if not lock.is_file():
        _fail(f"dependency lock is missing: {lock.relative_to(ROOT)}")

    index = CUDA_INDEX if profile == "gpu" else CPU_INDEX
    size_note = "several GiB; platform-dependent" if profile == "gpu" else "roughly 0.6-1.0 GiB"
    print(f"\n  syncing {lock.relative_to(ROOT)} ({size_note})")
    _run(
        [
            uv,
            "pip",
            "sync",
            "--python",
            str(repo_env.venv_python()),
            "--require-hashes",
            # PyPI supplies general dependencies while the PyTorch index
            # supplies backend-specific wheels. uv's first-index policy stops
            # at duplicate support packages on the PyTorch index, so allow the
            # exact cross-index match. --require-hashes still restricts every
            # accepted artefact to the generated lock.
            "--index-strategy",
            "unsafe-best-match",
            "--extra-index-url",
            index,
            str(lock),
        ]
    )

    print("\n  installing the repository itself")
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(repo_env.venv_python()),
            "--no-deps",
            "--no-build-isolation",
            "-e",
            ".",
        ]
    )

    print("\n  fetching tiny-shakespeare")
    _run([_python(), str(ROOT / "scripts" / "fetch_data.py")])

    print()
    disk({})
    print()
    doctor({})


def setup_cpu(opts: dict[str, str]) -> None:
    """Force the CPU-only torch wheel regardless of what nvidia-smi says."""
    setup({**opts, "PROFILE": "cpu"})


def setup_gpu(opts: dict[str, str]) -> None:
    """Force the locked CUDA 12.6 torch profile; run doctor afterward."""
    setup({**opts, "PROFILE": "gpu"})


def doctor(opts: dict[str, str]) -> None:
    """Environment self-check. Start every troubleshooting session here."""
    repo_env.apply()
    _run([_python(), str(ROOT / "scripts" / "doctor.py")])


def test(opts: dict[str, str]) -> None:
    """Run Tier-0 and Tier-1 tests. LAB=N restricts to one lab, K=expr filters."""
    nproc = repo_env.parse_positive_int(opts.get("NPROC", "2"), name="NPROC")
    repo_env.apply(nproc=nproc)

    target = _lab_dir(opts.get("LAB")) or (ROOT / "labs")
    cmd = [_python(), "-m", "pytest", str(target), "-m", "not tier2"]

    if "K" in opts:
        cmd += ["-k", opts["K"]]
    if opts.get("VERBOSE", "0") == "1":
        cmd += ["-vv"]
    if opts.get("FAST", "0") == "1":
        cmd += ["-m", "not tier2 and not slow", "-x"]

    _run(cmd)


def test_tier2(opts: dict[str, str]) -> None:
    """Run Tier-2 tests from Labs 0, 1, 2 or 3. Needs two visible CUDA devices."""
    repo_env.apply()
    target = _lab_dir(opts.get("LAB")) or (ROOT / "labs")
    if target != ROOT / "labs" and target.name not in TIER2_LAB_NAMES:
        _fail(
            f"{target.name} has no Tier-2 pytest cases. "
            "Its GPU-capable benchmark, if any, runs with "
            f"`python tasks.py bench LAB={opts['LAB']} NPROC=2`."
        )
    _run([_python(), "-m", "pytest", str(target), "-m", "tier2", "-v"])


def test_answers(opts: dict[str, str]) -> None:
    """Run the lab suite against the published answers in an isolated copy."""
    nproc = repo_env.parse_positive_int(opts.get("NPROC", "2"), name="NPROC")
    repo_env.apply(nproc=nproc)
    marker = "not tier2 and not slow" if opts.get("FAST", "0") == "1" else "not tier2"
    cmd = [_python(), str(ROOT / "scripts" / "test_answers.py"), "-m", marker]
    lab_dir = _lab_dir(opts.get("LAB"))
    if lab_dir is not None:
        cmd += ["--lab", lab_dir.name]
    if "K" in opts:
        cmd += ["-k", opts["K"]]
    if opts.get("VERBOSE", "0") == "1":
        cmd.append("-vv")
    if opts.get("FAST", "0") == "1":
        cmd.append("-x")
    _run(cmd)


def bench(opts: dict[str, str]) -> None:
    """Produce a lab's deliverable plots into runs/labN/."""
    lab_dir = _lab_dir(opts.get("LAB"))
    if lab_dir is None:
        _fail("bench needs a lab: python tasks.py bench LAB=1")
    nproc = repo_env.parse_positive_int(opts["NPROC"], name="NPROC") if "NPROC" in opts else None
    repo_env.apply(nproc=nproc)

    bench_dir = lab_dir / "bench"
    scripts = sorted(bench_dir.glob("*.py")) if bench_dir.exists() else []
    scripts = [s for s in scripts if not s.name.startswith("_")]
    if not scripts:
        _fail(f"no bench scripts found in {bench_dir}")

    only = opts.get("SCRIPT")
    selected = [script for script in scripts if not only or script.stem == only]
    if not selected:
        available = ", ".join(script.stem for script in scripts)
        _fail(
            f"no bench script named {only!r} in {bench_dir.relative_to(ROOT)}. "
            f"Choose from: {available}"
        )

    for script in selected:
        print(f"\n  --- {script.relative_to(ROOT)} ---")
        cmd = [_python(), str(script)]
        # Several benchmarks deliberately sweep their own world sizes. Preserve
        # that script-level default unless the caller explicitly selects NPROC.
        if nproc is not None:
            cmd += ["--nproc", str(nproc)]
        _run(cmd)


def reconcile(opts: dict[str, str]) -> None:
    """Print predictions.yaml against whatever bench/ last measured."""
    lab_dir = _lab_dir(opts.get("LAB"))
    if lab_dir is None:
        _fail("reconcile needs a lab: python tasks.py reconcile LAB=1")
    repo_env.apply()
    _run([_python(), "-m", "common.predictions", "--lab", str(lab_dir)])


def gantt(opts: dict[str, str]) -> None:
    """Render the schedule visualisation for a lab that produced an EventLog."""
    lab_dir = _lab_dir(opts.get("LAB"))
    if lab_dir is None:
        _fail("gantt needs a lab: python tasks.py gantt LAB=3")
    script = lab_dir / "bench" / "gantt.py"
    if not script.exists():
        _fail(f"{script.relative_to(ROOT)} does not exist; this lab has no Gantt deliverable")
    nproc = repo_env.parse_positive_int(opts.get("NPROC", "2"), name="NPROC")
    repo_env.apply(nproc=nproc)
    _run([_python(), str(script), "--nproc", str(nproc)])


def kill(opts: dict[str, str]) -> None:
    """Reap orphaned torchrun / spawn children left behind by a hang."""
    args = [_python(), str(ROOT / "scripts" / "kill_stragglers.py")]
    if opts.get("YES", "0") == "1":
        args.append("--yes")
    _run(args)


def disk(opts: dict[str, str]) -> None:
    """Per-directory footprint against the budget in docs/FOOTPRINT.md."""
    args = [_python(), str(ROOT / "scripts" / "disk_report.py")]
    if opts.get("WRITE", "0") == "1":
        args.append("--write")
    _run(args)


def verify_clean(opts: dict[str, str]) -> None:
    """Report watched global caches; snapshot/check must bracket the work."""
    print(
        "  report only: proving a clean run requires "
        "`scripts/verify_clean.py --snapshot`, then the workload, then `--check`."
    )
    _run([_python(), str(ROOT / "scripts" / "verify_clean.py")])


def clean(opts: dict[str, str]) -> None:
    """Remove generated artefacts. Keeps .venv and data/."""
    before = sum(_dir_bytes(ROOT / d) for d in (".cache", "runs", "traces"))
    failures: list[str] = []
    for name in (".cache", "runs", "traces", ".pytest_cache", ".ruff_cache"):
        path = ROOT / name
        if path.exists():
            try:
                shutil.rmtree(path)
            except OSError as exc:
                failures.append(f"{name}/: {exc}")
                print(f"  incomplete {name}/")
            else:
                print(f"  removed {name}/")
    removed = 0
    for pycache in ROOT.rglob("__pycache__"):
        if ".venv" in pycache.parts:
            continue
        try:
            shutil.rmtree(pycache)
        except OSError as exc:
            failures.append(f"{pycache.relative_to(ROOT)}: {exc}")
        else:
            removed += 1
    print(f"  removed {removed} __pycache__ directories")
    print(f"\n  reclaimed roughly {_human(before)}")
    if failures:
        formatted = "\n".join(f"    - {failure}" for failure in failures)
        _fail(f"cleanup was incomplete; close processes using these paths and retry:\n{formatted}")


def nuke(opts: dict[str, str]) -> None:
    """Remove everything this repo generated, including .venv and data/."""
    if _running_from(VENV):
        _relaunch_outside_venv()
    args = [sys.executable, str(ROOT / "scripts" / "nuke.py")]
    if opts.get("YES", "0") == "1":
        args.append("--yes")
    _run(args)


def nuke_venv(opts: dict[str, str]) -> None:
    """Delete just the virtualenv, so you can switch install profiles."""
    if VENV.exists():
        if _running_from(VENV):
            _relaunch_outside_venv()
        size = _dir_bytes(VENV)
        try:
            shutil.rmtree(VENV)
        except OSError as exc:
            _fail(f"could not completely remove .venv: {exc}")
        if VENV.exists():
            _fail(".venv still exists after removal; close processes using it and retry")
        print(f"  removed .venv/ ({_human(size)})")
    else:
        print("  no .venv/ to remove")


def lint(opts: dict[str, str]) -> None:
    """Check Python lint rules and formatting without changing files."""
    _run([_python(), "-m", "ruff", "check", "."])
    _run([_python(), "-m", "ruff", "format", "--check", "."])


def fmt(opts: dict[str, str]) -> None:
    """Format Python files and apply safe Ruff lint fixes."""
    _run([_python(), "-m", "ruff", "format", "."])
    _run([_python(), "-m", "ruff", "check", "--fix", "."])


def freeze_check(opts: dict[str, str]) -> None:
    """Confirm nobody edited common/reference_gpt.py. CI runs this."""
    _run([_python(), str(ROOT / "scripts" / "freeze_check.py")])


def help_(opts: dict[str, str]) -> None:
    """List task targets and supported KEY=VALUE options."""
    print(__doc__)
    print("Targets:\n")
    width = max(len(name) for name in TARGETS)
    for name, fn in TARGETS.items():
        summary = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        print(f"  {name:<{width}}  {summary}")
    print("\nVariables are passed as KEY=VALUE after the target:")
    print("  LAB=0|1|1b|2|3|4|h   NPROC=N   K=<pytest -k expr>")
    print("  PROFILE=cpu|gpu   VERBOSE=1   FAST=1   YES=1   WRITE=1   SCRIPT=<stem>\n")


TARGETS = {
    "setup": setup,
    "setup-cpu": setup_cpu,
    "setup-gpu": setup_gpu,
    "doctor": doctor,
    "test": test,
    "test-tier2": test_tier2,
    "test-answers": test_answers,
    "bench": bench,
    "reconcile": reconcile,
    "gantt": gantt,
    "kill": kill,
    "disk": disk,
    "verify-clean": verify_clean,
    "clean": clean,
    "nuke": nuke,
    "nuke-venv": nuke_venv,
    "lint": lint,
    "format": fmt,
    "freeze-check": freeze_check,
    "help": help_,
}


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help", "help"):
        help_({})
        return

    target = argv[0]
    if target not in TARGETS:
        print(f"  unknown target {target!r}\n")
        help_({})
        raise SystemExit(2)

    opts: dict[str, str] = {}
    for arg in argv[1:]:
        if "=" not in arg:
            _fail(f"arguments are KEY=VALUE pairs; got {arg!r}")
        key, value = arg.split("=", 1)
        opts[key.upper()] = value

    TARGETS[target](opts)


if __name__ == "__main__":
    main(sys.argv[1:])
