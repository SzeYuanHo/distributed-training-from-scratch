"""Process groups, spawning, seeding, and rank-safe printing.

This is the layer every lab sits on. Four things here exist because getting
them wrong wastes an afternoon:

``pick_backend``
    Refuses the curriculum's NCCL launch path when there are more ranks than
    visible GPUs. These labs follow the normal process-per-GPU contract and
    require exclusive access to one device per rank; oversubscribing a device
    can hang or fail with invalid usage. Multi-process work on one GPU uses gloo.

``free_port``
    A crashed run leaves its ``MASTER_PORT`` in TIME_WAIT and the next run dies
    with "Address already in use". Asking the kernel for a free port each time
    removes the whole class of problem.

``run_distributed``
    Spawns ranks in-process so pytest can drive a multi-rank run. Always
    ``spawn``, never ``fork``: fork plus CUDA is unsafe, and fork does not
    exist on Windows.

``printr``
    Eight ranks printing at once produces interleaved garbage. This serialises
    output through a barrier so rank 0's line always comes first.
"""

from __future__ import annotations

import contextlib
import os
import random
import socket
import sys
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from common.repo_env import parse_positive_int, positive_int_env

__all__ = [
    "init",
    "cleanup",
    "is_initialized",
    "rank",
    "world_size",
    "local_rank",
    "is_main",
    "device_for_rank",
    "pick_backend",
    "seed_everything",
    "sync_module_params",
    "printr",
    "free_port",
    "run_distributed",
    "DistributedTimeout",
]

DEFAULT_TIMEOUT_S = positive_int_env("DTFS_TIMEOUT_S", 300)


class DistributedTimeout(RuntimeError):
    """A spawned run did not finish in time. Almost always a deadlock."""


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #


def pick_backend(world_size_: int, requested: str | None = None) -> str:
    """Choose a backend, or explain why the requested one cannot work.

    The rule that matters here: the launcher assigns one exclusive device per
    NCCL rank. Oversubscribing a GPU can deadlock or fail with invalid usage
    rather than producing a useful launch-time error. This function makes it a
    readable exception first.
    """
    world_size_ = parse_positive_int(world_size_, name="world_size")
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if requested is not None:
        requested = requested.lower()
        if requested == "nccl":
            if not dist.is_nccl_available():
                raise RuntimeError(
                    "NCCL was requested but this torch build has no NCCL. "
                    "That is normal on native Windows and on CPU-only wheels. "
                    "Use backend='gloo', or run Tier 2 in a Linux environment with "
                    "enough visible CUDA devices."
                )
            if world_size_ > n_cuda:
                raise RuntimeError(
                    f"NCCL was requested for world_size={world_size_} but only {n_cuda} "
                    f"CUDA device(s) are visible.\n"
                    "  These labs require one exclusive CUDA device per NCCL rank. "
                    "More ranks than devices can hang or fail with invalid usage.\n"
                    "  Use backend='gloo' for multi-process work on one GPU, or run this "
                    "on a machine with 2+ GPUs (see notebooks/kaggle_2xT4_tier2.ipynb)."
                )
        if requested == "gloo" and not dist.is_gloo_available():
            raise RuntimeError("gloo is not available in this torch build, which is unusual.")
        return requested

    # Auto: NCCL only when it can genuinely be one rank per device.
    if n_cuda >= 2 and world_size_ <= n_cuda and dist.is_nccl_available():
        return "nccl"
    return "gloo"


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def init(backend: str | None = None, timeout_s: int = DEFAULT_TIMEOUT_S) -> str:
    """Initialise the default process group from the standard env variables.

    Expects ``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``, which
    ``run_distributed`` and ``torchrun`` both set. Returns the backend chosen.

    The explicit timeout matters more than it looks. The default is long
    enough that a deadlocked test looks like a slow test, and you sit watching
    a blank terminal wondering whether to wait. Five minutes is long enough for
    any legitimate Tier-1 collective and short enough to notice.
    """
    if dist.is_initialized():
        return dist.get_backend()

    timeout_s = parse_positive_int(timeout_s, name="timeout_s")
    ws = parse_positive_int(os.environ.get("WORLD_SIZE", "1"), name="WORLD_SIZE")
    rk = int(os.environ.get("RANK", "0"))
    if not 0 <= rk < ws:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE); got RANK={rk}, WORLD_SIZE={ws}")
    chosen = pick_backend(ws, backend)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(free_port()))

    dist.init_process_group(
        backend=chosen,
        rank=rk,
        world_size=ws,
        timeout=timedelta(seconds=timeout_s),
    )

    if chosen == "nccl":
        torch.cuda.set_device(local_rank())

    return chosen


def cleanup() -> None:
    """Tear the process group down. Safe to call when nothing is initialised."""
    if dist.is_initialized():
        with contextlib.suppress(Exception):
            dist.barrier()
        dist.destroy_process_group()


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_initialized() else int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return dist.get_world_size() if is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))


def local_rank() -> int:
    """Rank within this node. Identical to ``rank()`` for single-node runs."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def is_main() -> bool:
    return rank() == 0


def device_for_rank() -> torch.device:
    """The device this rank should put tensors on.

    An initialized Gloo process group always uses CPU tensors, even on a host
    where CUDA devices happen to be visible. An initialized NCCL group uses
    the rank-local CUDA device. Before process-group initialization, the helper
    retains the convenient one-visible-device-per-rank auto-selection.
    """
    if is_initialized():
        backend = str(dist.get_backend()).lower()
        if backend != "nccl":
            return torch.device("cpu")
        return torch.device(f"cuda:{local_rank()}")
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size():
        return torch.device(f"cuda:{local_rank()}")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def seed_everything(seed: int = 1337, rank_offset: bool = False) -> int:
    """Seed python, numpy, and torch. Returns the seed actually used.

    ``rank_offset`` is the whole subtlety, and getting it backwards is one of
    the silent bugs in Lab H:

    * Model initialisation wants ``rank_offset=False``. Every rank must build
      the *same* weights, or you are training N unrelated models and averaging
      their gradients, which produces a falling loss and a useless checkpoint.
    * Data sampling and dropout on sharded activations want
      ``rank_offset=True``, or every rank draws the same batch and your
      effective dataset is 1/N the size you think it is.
    """
    effective = seed + (rank() if rank_offset else 0)
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective


def sync_module_params(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast rank ``src``'s parameters and buffers to every other rank.

    Call this in your DDP wrapper's constructor. Skipping it is the classic
    desync bug: the loss falls, the gradients average cleanly, and the model
    is garbage because each rank started somewhere different.
    """
    if not is_initialized() or world_size() == 1:
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #


def printr(*args: Any, only: int | None = None, flush: bool = True, **kwargs: Any) -> None:
    """Rank-prefixed print, ordered by rank via barriers.

    Pass ``only=0`` to print from one rank. Without ordering, eight ranks
    writing to the same terminal produce text that is technically all there and
    practically unreadable.

    This costs ``world_size`` barriers per call. Do not put it in a hot loop;
    write per-rank log files instead, as ``docs/DEBUGGING.md`` describes.
    """
    if not is_initialized():
        print(*args, flush=flush, **kwargs)
        return

    ws, rk = world_size(), rank()
    if only is not None:
        if rk == only:
            print(f"[rank {rk}]", *args, flush=flush, **kwargs)
        dist.barrier()
        return

    for r in range(ws):
        if rk == r:
            print(f"[rank {rk}]", *args, flush=flush, **kwargs)
            sys.stdout.flush()
        dist.barrier()


# --------------------------------------------------------------------------- #
# launching
# --------------------------------------------------------------------------- #


def free_port() -> int:
    """Ask the kernel for an unused TCP port and immediately release it.

    There is a race here: something else can grab the port between the release
    and the bind inside ``init_process_group``. In practice that never happens
    on a laptop, and the alternative, hardcoding 29500, collides with itself the
    moment you run two labs at once.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _worker_entry(
    rank_: int,
    fn: Callable[..., Any],
    world_size_: int,
    backend: str,
    master_addr: str,
    master_port: int,
    timeout_s: int,
    env_overrides: dict[str, str],
    fn_args: Sequence[Any],
    fn_kwargs: dict[str, Any],
    results: Any,
) -> None:
    """Runs inside each spawned child. Module-level so it pickles."""
    os.environ.update(env_overrides)
    os.environ["RANK"] = str(rank_)
    os.environ["LOCAL_RANK"] = str(rank_)
    os.environ["WORLD_SIZE"] = str(world_size_)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    init(backend=backend, timeout_s=timeout_s)
    try:
        value = fn(*fn_args, **fn_kwargs)
        if results is not None:
            results[rank_] = value
    finally:
        cleanup()


def run_distributed(
    fn: Callable[..., Any],
    world_size: int = 2,
    backend: str | None = None,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    join_timeout_s: float | None = None,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    collect_results: bool = True,
) -> dict[int, Any]:
    """Spawn ``world_size`` ranks running ``fn`` and return ``{rank: result}``.

    ``fn`` takes no rank argument. Call ``dist_utils.rank()`` inside it. That
    keeps worker functions usable unchanged under ``torchrun``, where nobody
    passes you a rank either.

    Two constraints inherited from ``spawn``, and both bite the first time:

    1. ``fn`` must be defined at module level. A closure, a lambda, or a
       function defined inside a test body will not pickle, and the error
       message ("Can't pickle local object") does not obviously point here.
    2. Any caller script needs ``if __name__ == "__main__":``. Without it,
       each child re-imports the module, spawns its own children, and you get
       a fork bomb that is slow enough to look like a hang.

    ``join_timeout_s`` defaults to ``timeout_s + 60`` and is the outer safety
    net: if the process group's own timeout does not fire, this kills the
    children and raises ``DistributedTimeout`` rather than letting pytest sit
    there forever.
    """
    world_size = parse_positive_int(world_size, name="world_size")
    timeout_s = parse_positive_int(timeout_s, name="timeout_s")
    kwargs = dict(kwargs or {})
    join_timeout_s = join_timeout_s if join_timeout_s is not None else timeout_s + 60
    if join_timeout_s <= 0:
        raise ValueError(f"join_timeout_s must be positive; got {join_timeout_s!r}")
    chosen = pick_backend(world_size, backend)

    from common import repo_env

    env_overrides = repo_env.cache_env(nproc=world_size)
    env_overrides["DTFS_SPAWNED"] = "1"

    ctx = mp.get_context("spawn")
    manager = ctx.Manager() if collect_results else None
    results = manager.dict() if manager is not None else None

    master_addr = "127.0.0.1"
    master_port = free_port()

    proc_ctx = mp.start_processes(
        _worker_entry,
        args=(
            fn,
            world_size,
            chosen,
            master_addr,
            master_port,
            timeout_s,
            env_overrides,
            tuple(args),
            kwargs,
            results,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
    )

    try:
        # ProcessContext.join returns True only once every child has exited, and
        # returns False whenever it comes back early. Poll in short slices
        # against a wall-clock deadline instead of trusting a single call.
        deadline = time.monotonic() + join_timeout_s
        finished = False
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                if proc_ctx.join(timeout=min(remaining, 5.0)):
                    finished = True
                    break
            except mp.ProcessExitedException as exc:
                # A child died without raising a python exception, which means
                # something below python killed it. During a distributed run that
                # is almost always the backend aborting a collective that timed
                # out, and the cause is a deadlock rather than a crash.
                #
                # Worth translating, because the raw message is an exit code
                # (0xC0000409 on Windows, SIGABRT on POSIX) that tells the reader
                # nothing about what actually went wrong.
                raise DistributedTimeout(
                    f"rank {exc.error_index} of {world_size} was killed by the backend "
                    f"(exit code {exc.exit_code}) on {chosen}.\n"
                    "  A child that dies without a python traceback during a collective "
                    "has almost always hit the process-group timeout, which means the "
                    "peers it was waiting for never arrived.\n"
                    "  The usual cause is one rank calling a collective the others do "
                    "not, often inside `if rank == 0:`.\n"
                    "  Try: TORCH_DISTRIBUTED_DEBUG=DETAIL, then `python tasks.py kill`."
                ) from exc

        if not finished:
            raise DistributedTimeout(
                f"{world_size} ranks did not finish within {join_timeout_s:.0f}s on "
                f"backend {chosen}.\n"
                "  This is a deadlock until proven otherwise. The usual cause is one rank "
                "calling a collective the others do not, often inside `if rank == 0:`.\n"
                "  Try: TORCH_DISTRIBUTED_DEBUG=DETAIL, then `python tasks.py kill`."
            )
    finally:
        for proc in proc_ctx.processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

    out = dict(results) if results is not None else {}
    if manager is not None:
        manager.shutdown()
    return out
