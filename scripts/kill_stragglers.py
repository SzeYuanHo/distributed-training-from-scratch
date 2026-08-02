#!/usr/bin/env python3
"""Reap orphaned ranks after a hang.

Killing ``torchrun`` with Ctrl-C often leaves its children alive. They keep
holding the master port and one core each, so your next run fails with
"Address already in use" and your laptop fan stays on. This finds them and
offers to kill them.

Matching is deliberately conservative: a process must be Python, be associated
with this repository, carry a distributed-worker marker, and no longer have a
live repository Python parent or child. Active runs and editor language servers
are excluded.

::

    python tasks.py kill          # asks first
    python tasks.py kill YES=1    # does not ask
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import repo_env  # noqa: E402

Candidate = tuple[int, str, float, float]


def find_candidates() -> list[Candidate]:
    try:
        import psutil
    except ImportError:
        print("  psutil is not installed, so this cannot search processes.")
        print("  Install the locked environment, or inspect candidates by hand:")
        print("    POSIX:   pgrep -af 'torchrun|torch.distributed.run|spawn_main'")
        print(
            "    Windows: Get-CimInstance Win32_Process -Filter "
            "\"Name = 'python.exe'\" | Select-Object ProcessId,CommandLine"
        )
        print("  Verify repository ownership, then terminate only the exact PID.")
        return []

    # Skip ourselves, every ancestor, and every child. Otherwise `tasks.py kill`
    # offers to kill `tasks.py kill`, which works exactly once.
    skip = {os.getpid()}
    try:
        me_proc = psutil.Process()
        skip.update(p.pid for p in me_proc.parents())
        skip.update(p.pid for p in me_proc.children(recursive=True))
    except psutil.Error:
        pass

    root_path = repo_env.repo_root().resolve()
    root = str(root_path).lower().replace("\\", "/")
    distributed_markers = ("torchrun", "torch.distributed.run", "spawn_main")

    def repo_associated(proc: psutil.Process, command: str) -> bool:
        if root in command.lower().replace("\\", "/"):
            return True
        try:
            configured = proc.environ().get("DTFS_REPO_ROOT")
            return configured is not None and Path(configured).resolve() == root_path
        except (OSError, psutil.Error):
            return False

    def distributed_worker(proc: psutil.Process, command: str) -> bool:
        low = command.lower()
        if any(marker in low for marker in distributed_markers):
            return True
        try:
            env = proc.environ()
            return int(env.get("WORLD_SIZE", "1")) > 1 and "RANK" in env
        except (OSError, ValueError, psutil.Error):
            return False

    found: list[Candidate] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            pid = proc.info["pid"]
            if pid in skip:
                continue
            if "python" not in (proc.info["name"] or "").lower():
                continue
            cmd = " ".join(proc.info["cmdline"] or [])
            low = cmd.lower()
            if not repo_associated(proc, cmd) or not distributed_worker(proc, cmd):
                continue
            # Skip this script and the task runner invoking it.
            if "kill_stragglers" in low or "tasks.py kill" in low:
                continue
            # A live repository Python parent/controller or child means this is
            # an active process group, not a detached worker left by that group.
            parent = proc.parent()
            if parent is not None:
                parent_cmd = " ".join(parent.cmdline())
                if "python" in parent.name().lower() and repo_associated(parent, parent_cmd):
                    continue
            if any(
                "python" in child.name().lower()
                and repo_associated(child, " ".join(child.cmdline()))
                for child in proc.children()
            ):
                continue
            created_at = proc.info["create_time"]
            if created_at is None:
                continue
            age = time.time() - created_at
            found.append((pid, cmd, age, created_at))
        except Exception:  # noqa: BLE001 - process tables race constantly
            continue
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--dry-run", action="store_true", help="list them and stop")
    args = parser.parse_args(argv)

    candidates = find_candidates()
    if not candidates:
        print("  no orphaned distributed processes found.")
        return 0

    print(f"  found {len(candidates)} candidate process(es):\n")
    for pid, cmd, age, _created_at in candidates:
        print(f"    pid {pid:<8} age {age / 60:5.1f} min   {cmd[:100]}")
    print()

    if args.dry_run:
        return 0

    if not args.yes:
        try:
            answer = input("  terminate these? [y/N] ").strip().lower()
        except EOFError:
            print("  no tty; re-run with --yes to skip the prompt.")
            return 1
        if answer not in ("y", "yes"):
            print("  nothing killed.")
            return 0

    import psutil

    # The prompt can stay open long enough for a PID to exit and be reused.
    # Re-run the full conservative match and require the same creation time and
    # command immediately before sending any signal.
    current = {(pid, created_at): cmd for pid, cmd, _age, created_at in find_candidates()}
    targets = []
    failures = 0
    for pid, cmd, _age, created_at in candidates:
        if current.get((pid, created_at)) != cmd:
            if psutil.pid_exists(pid):
                print(f"    pid {pid}: identity or command changed; skipped safely")
                failures += 1
            continue
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            targets.append(proc)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            print(f"    pid {pid}: {type(exc).__name__}")
            failures += 1

    # Give SIGTERM a moment, then insist.
    _gone, alive = psutil.wait_procs(targets, timeout=5)
    forced = len(alive)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as exc:
            print(f"    pid {proc.pid}: {type(exc).__name__} while forcing termination")
            failures += 1

    _gone, still_alive = psutil.wait_procs(alive, timeout=5)
    for proc in still_alive:
        print(f"    pid {proc.pid}: still alive after forced termination")
    failures += len(still_alive)

    print(f"\n  sent termination to {len(targets)} process(es); {forced} needed SIGKILL.")
    print("  If ports are still held, wait a few seconds for TIME_WAIT to clear.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
