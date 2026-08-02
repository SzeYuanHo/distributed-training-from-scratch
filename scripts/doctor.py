#!/usr/bin/env python3
"""Environment self-check. Start every troubleshooting session here.

The most valuable line is the last one: an explicit verdict about which tiers
this machine can run. A learner who starts a two-rank NCCL job with only one
visible GPU may get a deadlock or an invalid-usage error. This tells them up
front.

Exit code 0 when Tier 1 works, 1 otherwise. CI relies on that.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

repo_env.apply()

CHECK = "ok"
CROSS = "no"


def line(label: str, value: object, width: int = 30) -> None:
    print(f"  {label:<{width}}{value}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * max(60, len(title)))


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


def probe_gloo() -> tuple[bool, str]:
    """Actually build a one-rank gloo group instead of trusting the flag.

    ``torch.distributed.is_gloo_available()`` reports whether gloo was compiled
    in, not whether it has a usable transport. Several CUDA Windows wheels
    answer True and then fail at ``init_process_group`` with
    "makeDeviceForHostname(): unsupported gloo device", because the build has
    no TCP or UV transport.

    That distinction matters enough to spend two seconds on. Tier 1 is the main
    development loop, and finding out it does not work three labs in is a bad
    afternoon.

    Runs in a subprocess so a failed probe cannot poison this process's
    distributed state.
    """
    import subprocess
    import textwrap

    # Bind port 0 so the kernel selects an unused local port. The socket must
    # be released before the child can bind it; a tiny race remains, so retry
    # once if another process happens to claim the same port.
    def free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    code = textwrap.dedent(
        """
        import os, torch, torch.distributed as dist
        dist.init_process_group("gloo", rank=0, world_size=1)
        t = torch.ones(2)
        dist.all_reduce(t)
        assert t.tolist() == [1.0, 1.0]
        dist.destroy_process_group()
        print("PROBE_OK")
        """
    )
    result = None
    for _attempt in range(2):
        env = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(free_port()),
            "RANK": "0",
            "WORLD_SIZE": "1",
        }
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=90,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"probe could not run: {exc}"

        if "PROBE_OK" in result.stdout:
            return True, "one-rank group created and all-reduced"
        if "address already in use" not in (result.stderr or "").lower():
            break

    assert result is not None
    tail = (result.stderr or result.stdout).strip().splitlines()
    reason = tail[-1][:110] if tail else "unknown failure"
    return False, reason


def related_pids() -> set[int]:
    """This process, its ancestors, and its descendants.

    Without this the check reports itself as an orphan: ``tasks.py doctor``
    spawns ``doctor.py`` which spawns the gloo probe, and all three have the
    repo path in their command line.
    """
    try:
        import psutil
    except ImportError:
        return {os.getpid()}

    pids = {os.getpid()}
    try:
        me = psutil.Process()
        pids.update(p.pid for p in me.parents())
        pids.update(p.pid for p in me.children(recursive=True))
    except psutil.Error:
        pass
    return pids


def find_orphans() -> list[tuple[int, str]]:
    """Python processes that look like leftovers from a crashed distributed run."""
    try:
        import psutil
    except ImportError:
        return []

    skip = related_pids()
    root = str(repo_env.repo_root()).lower()
    needles = ("torchrun", "spawn_main", "torch.distributed.run", root)
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] in skip:
                continue
            name = (proc.info["name"] or "").lower()
            if "python" not in name:
                continue
            cmd = " ".join(proc.info["cmdline"] or []).lower()
            if any(n in cmd for n in needles):
                found.append((proc.info["pid"], cmd[:90]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def main() -> int:
    root = repo_env.repo_root()

    print("=" * 62)
    print("  dist-train-from-scratch :: environment check")
    print("=" * 62)

    # ---------------------------------------------------------------- system
    section("System")
    line("platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    line("python", f"{sys.version.split()[0]}  ({sys.executable})")
    line("repo root", root)
    line("in venv", "yes" if sys.prefix != sys.base_prefix else "no  <- run tasks.py setup")

    cores = os.cpu_count() or 1
    line("cpu cores", cores)
    line("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "unset"))
    line("recommended max NPROC", f"{cores} at Tier 1 (one rank per physical core)")

    # ----------------------------------------------------------------- torch
    section("PyTorch")
    try:
        import torch
    except ImportError:
        print("  torch is not installed.")
        print("\n  Run: python tasks.py setup-cpu")
        return 1

    line("torch", torch.__version__)
    line("built with CUDA", torch.version.cuda or "no (CPU-only wheel)")

    cuda_ok = torch.cuda.is_available()
    n_devices = torch.cuda.device_count() if cuda_ok else 0
    line("CUDA available", f"{'yes' if cuda_ok else 'no'}   devices: {n_devices}")

    for i in range(n_devices):
        props = torch.cuda.get_device_properties(i)
        bf16 = "yes" if torch.cuda.is_bf16_supported() else "no"
        line(
            f"  device {i}",
            f"{props.name}  {props.total_memory / 1e9:.1f} GB  "
            f"sm_{props.major}{props.minor}  bf16={bf16}",
        )
    if n_devices == 1:
        line("", "(T4s are Turing and have no bf16 tensor cores; gate on")
        line("", " torch.cuda.is_bf16_supported() in any Tier-2 code)")

    # -------------------------------------------------------------- backends
    section("Distributed backends")
    import torch.distributed as dist

    gloo_compiled = dist.is_gloo_available()
    nccl = dist.is_nccl_available()
    line("gloo compiled in", CHECK if gloo_compiled else CROSS)

    gloo = False
    if gloo_compiled:
        print("  probing gloo (a compiled-in backend can still have no transport) ...")
        gloo, detail = probe_gloo()
        line("gloo actually works", f"{CHECK}  {detail}" if gloo else f"{CROSS}  {detail}")
        if not gloo:
            line("", "This wheel reports gloo but cannot create a device.")
            line("", "Known on some CUDA Windows builds. Fixes, in order of ease:")
            line("", "  1. python tasks.py nuke-venv && python tasks.py setup-cpu")
            line("", "  2. run the labs under WSL2 (see docs/SETUP.md)")
            line("", "  3. docker compose -f docker/docker-compose.yml run --rm tests")

    line("nccl", CHECK if nccl else f"{CROSS}  (expected on native Windows and CPU wheels)")
    line("mpi", CHECK if dist.is_mpi_available() else CROSS)

    import multiprocessing as mp

    methods = mp.get_all_start_methods()
    line("start methods", ", ".join(methods))
    line("spawn", CHECK if "spawn" in methods else CROSS)
    line("fork", (CHECK if "fork" in methods else CROSS) + "  (this repo never uses fork)")

    # ------------------------------------------------------------------ disk
    section("Disk and caches")
    usage = shutil.disk_usage(root)
    line("free on repo filesystem", human(usage.free))

    footprint = {
        ".venv": dir_bytes(root / ".venv"),
        ".cache": dir_bytes(root / ".cache"),
        "data": dir_bytes(root / "data"),
        "runs": dir_bytes(root / "runs"),
    }
    for name, size in footprint.items():
        line(f"  {name}/", human(size))
    line(
        "  managed dirs + labs",
        human(sum(footprint.values()) + dir_bytes(root / "labs")),
    )

    active = repo_env.is_active()
    line(
        "cache redirection",
        (CHECK + "  everything lands in ./.cache")
        if active
        else (CROSS + "  caches are going to your home directory"),
    )
    if not active:
        line("", "fix: source activate.sh   (or . .\\activate.ps1)")

    # --------------------------------------------------------------- dataset
    from common import data as data_mod

    data_file = root / "data" / "tinyshakespeare.txt"
    dataset_ok = False
    dataset_detail = f"{CROSS}  run: python scripts/fetch_data.py"
    if data_file.is_file():
        digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
        dataset_ok = (
            data_file.stat().st_size == data_mod.SHAKESPEARE_BYTES
            and digest == data_mod.SHAKESPEARE_SHA256
        )
        dataset_detail = (
            f"{CHECK}  {human(data_file.stat().st_size)}  checksum verified"
            if dataset_ok
            else f"{CROSS}  size or checksum mismatch; run: python scripts/fetch_data.py"
        )
    line(
        "tiny-shakespeare",
        dataset_detail,
    )

    # -------------------------------------------------------------- orphans
    orphans = find_orphans()
    section("Leftover processes")
    if orphans:
        line("orphaned python procs", f"{len(orphans)} found  -> run: python tasks.py kill")
        for pid, cmd in orphans[:5]:
            print(f"    pid {pid}: {cmd}")
    else:
        line("orphaned python procs", f"none {CHECK}")

    # --------------------------------------------------------------- verdict
    tier1_ok = gloo and "spawn" in methods
    tier2_ok = nccl and n_devices >= 2

    print()
    print("=" * 62)
    print(f"VERDICT: {n_devices} CUDA device(s) detected.")
    print(f"  Tier 0 (single process, pure math):     {'AVAILABLE' if True else 'UNAVAILABLE'}")
    print(f"  Tier 1 (gloo/CPU multi-process):        {'AVAILABLE' if tier1_ok else 'UNAVAILABLE'}")
    print(f"  Tier 2 (NCCL multi-GPU):                {'AVAILABLE' if tier2_ok else 'UNAVAILABLE'}")

    if not tier2_ok:
        print()
        print("  This curriculum's NCCL path follows PyTorch's one-process-per-GPU")
        print("  launch model and requires exclusive access to at least two CUDA devices.")
        print("  Sharing a device between NCCL processes can deadlock or report invalid")
        print("  usage. Use a Linux environment with 2+ visible GPUs for Tier 2; the")
        print("  included Kaggle notebook explains how to select a multi-GPU accelerator")
        print("  when that option is available:")
        print("    notebooks/kaggle_2xT4_tier2.ipynb")
        if n_devices == 1:
            print()
            print("  Your single GPU is still useful: single-process profiling, kernel")
            print("  timing, and the memory measurements in Lab 1B all run on it.")

    if not tier1_ok:
        print()
        print("  Tier 1 is the default development loop and it is not working here.")
        if not gloo:
            print("  gloo could not create a device. Tier 0 still runs, so Lab 0's math")
            print("  tests and Lab 2's algebra tests work, but nothing multi-rank will.")
            print("  Pick one:")
            print("    python tasks.py nuke-venv && python tasks.py setup-cpu")
            print("    WSL2:   wsl --install -d Ubuntu, then clone and setup inside it")
            print("    Docker: docker compose -f docker/docker-compose.yml run --rm tests")
        else:
            print("  spawn is unavailable, which is unusual. Check the python build.")

    print("=" * 62)
    print()
    print("Next: python tasks.py test LAB=0")
    print()

    return 0 if tier1_ok and dataset_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
