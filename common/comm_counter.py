"""Counts and sizes every collective a run issues.

This is the most useful instrument in the repository, and the reason is worth
stating plainly: **a numerically correct implementation can still be wrong.**
A tensor-parallel block that produces the right answer with six all-reduces per
layer instead of four is a real bug with a real cost, and no comparison against
a reference tensor will ever catch it. Counting collectives does.

Two capture paths, because neither alone is enough:

1. **The wrapper path.** ``common.collectives`` records into whatever counter is
   active before delegating to ``torch.distributed``. Lab code is required to
   call those wrappers. This path is guaranteed and it carries tags, so you can
   attribute traffic to "bucket flush" or "row-parallel forward".

2. **The dispatch path.** A ``TorchDispatchMode`` that intercepts ops in the
   ``c10d`` and ``_c10d_functional`` namespaces. This catches collectives issued
   by machinery you did not write: real ``DistributedDataParallel``, DTensor,
   ``parallelize_module``. Turn it on when you want to diff your implementation
   against PyTorch's own.

Usage::

    with CommCounter() as counter:
        model_step()

    print(counter.summary())
    counter.assert_equals({"all_reduce": 4})          # count per op
    counter.assert_equals({"all_reduce": (4, None)})  # count, bytes

Counters nest. An inner counter records into itself and into every enclosing
counter, so you can wrap a whole step and a single layer at the same time.
"""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from torch.utils._python_dispatch import TorchDispatchMode

    _DISPATCH_AVAILABLE = True
except ImportError:  # pragma: no cover - only on a torch build without it
    TorchDispatchMode = object  # type: ignore[assignment,misc]
    _DISPATCH_AVAILABLE = False

__all__ = ["CommCounter", "CommRecord", "tag", "current_counters", "record"]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommRecord:
    """One collective call."""

    op: str
    n_bytes: int
    group_size: int
    tag: str = ""
    rank: int = -1
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        where = f" @{self.tag}" if self.tag else ""
        return f"{self.op}({_human_bytes(self.n_bytes)}, group={self.group_size}){where}"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} GiB"


def tensor_bytes(t: Any) -> int:
    """Bytes in a tensor, or the sum over a list of them. Zero for anything else."""
    if isinstance(t, torch.Tensor):
        return t.numel() * t.element_size()
    if isinstance(t, (list, tuple)):
        return sum(tensor_bytes(x) for x in t)
    return 0


# --------------------------------------------------------------------------- #
# the active-counter stack
# --------------------------------------------------------------------------- #

_state = threading.local()


def _stack() -> list[CommCounter]:
    if not hasattr(_state, "counters"):
        _state.counters = []
    return _state.counters


def _tags() -> list[str]:
    if not hasattr(_state, "tags"):
        _state.tags = []
    return _state.tags


def current_counters() -> list[CommCounter]:
    return list(_stack())


def current_tag() -> str:
    parts = _tags()
    return "/".join(parts) if parts else ""


@contextlib.contextmanager
def tag(name: str) -> Iterator[None]:
    """Label every collective issued inside this block.

    Tags nest with a slash, so ``tag("step") / tag("mlp")`` shows up as
    ``step/mlp`` in the summary. Use it to answer "which part of my model is
    generating all this traffic".
    """
    _tags().append(name)
    try:
        yield
    finally:
        _tags().pop()


def record(op: str, n_bytes: int, group_size: int = 1, **extra: Any) -> None:
    """Called by ``common.collectives``. Records into every active counter."""
    stack = _stack()
    if not stack:
        return
    from common import dist_utils

    rec = CommRecord(
        op=op,
        n_bytes=int(n_bytes),
        group_size=group_size,
        tag=current_tag(),
        rank=dist_utils.rank(),
        extra=extra,
    )
    for counter in stack:
        counter._add(rec)


# --------------------------------------------------------------------------- #
# the dispatch-mode capture path
# --------------------------------------------------------------------------- #

# Op names as they appear under torch.ops, mapped to the friendly names the
# wrapper layer uses, so both capture paths land in the same bucket.
_DISPATCH_NAME_MAP = {
    "allreduce_": "all_reduce",
    "all_reduce": "all_reduce",
    "all_reduce_": "all_reduce",
    "allgather_": "all_gather",
    "all_gather_into_tensor": "all_gather",
    "all_gather_single": "all_gather",
    "_allgather_base_": "all_gather",
    "reduce_scatter_": "reduce_scatter",
    "reduce_scatter_tensor": "reduce_scatter",
    "reduce_scatter_single": "reduce_scatter",
    "_reduce_scatter_base_": "reduce_scatter",
    "broadcast_": "broadcast",
    "reduce_": "reduce",
    "alltoall_": "all_to_all",
    "all_to_all_single": "all_to_all",
    "send": "send",
    "recv_": "recv",
    "barrier": "barrier",
}


class _CollectiveDispatchMode(TorchDispatchMode):
    """Intercepts c10d ops so PyTorch's own collectives show up too."""

    def __init__(self, counter: CommCounter):
        super().__init__()
        self.counter = counter

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "_schema", None)
        qualified = str(name.name) if name is not None else str(func)

        if "c10d" in qualified:
            short = qualified.split("::")[-1]
            friendly = _DISPATCH_NAME_MAP.get(short, short)
            n_bytes = tensor_bytes(args[0]) if args else 0
            from common import dist_utils

            self.counter._add(
                CommRecord(
                    op=friendly,
                    n_bytes=n_bytes,
                    group_size=dist_utils.world_size(),
                    tag=current_tag(),
                    rank=dist_utils.rank(),
                    extra={"source": "dispatch", "schema": qualified},
                )
            )

        return func(*args, **kwargs)


# --------------------------------------------------------------------------- #
# the counter
# --------------------------------------------------------------------------- #


class CommCounter:
    """Context manager that records every collective issued inside it.

    Parameters
    ----------
    dispatch:
        Also install a ``TorchDispatchMode`` to catch collectives from code you
        did not write. Off by default because it adds a per-op cost to every
        tensor operation, not just collectives.
    name:
        Shown in ``summary()``. Handy when you print several counters.
    """

    def __init__(self, *, dispatch: bool = False, name: str = ""):
        self.name = name
        self.records: list[CommRecord] = []
        self._dispatch = dispatch
        self._mode: _CollectiveDispatchMode | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def __enter__(self) -> CommCounter:
        _stack().append(self)
        if self._dispatch:
            if not _DISPATCH_AVAILABLE:
                raise RuntimeError(
                    "dispatch=True needs torch.utils._python_dispatch.TorchDispatchMode, "
                    "which this torch build does not expose. The wrapper path in "
                    "common.collectives still works; you just will not see collectives "
                    "issued by PyTorch's own DDP or DTensor."
                )
            self._mode = _CollectiveDispatchMode(self)
            self._mode.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._mode is not None:
            self._mode.__exit__(*exc_info)
            self._mode = None
        stack = _stack()
        if stack and stack[-1] is self:
            stack.pop()
        elif self in stack:
            stack.remove(self)

    def _add(self, rec: CommRecord) -> None:
        self.records.append(rec)

    def reset(self) -> None:
        self.records.clear()

    # -- queries ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.records)

    @property
    def total_bytes(self) -> int:
        return sum(r.n_bytes for r in self.records)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for rec in self.records:
            out[rec.op] += 1
        return dict(out)

    def bytes_by_op(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for rec in self.records:
            out[rec.op] += rec.n_bytes
        return dict(out)

    def by_tag(self) -> dict[str, tuple[int, int]]:
        """``{tag: (call_count, total_bytes)}``."""
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for rec in self.records:
            slot = out[rec.tag or "(untagged)"]
            slot[0] += 1
            slot[1] += rec.n_bytes
        return {k: (v[0], v[1]) for k, v in out.items()}

    def filter(self, op: str | None = None, tag_prefix: str | None = None) -> list[CommRecord]:
        return [
            r
            for r in self.records
            if (op is None or r.op == op) and (tag_prefix is None or r.tag.startswith(tag_prefix))
        ]

    # -- reporting ---------------------------------------------------------- #

    def summary(self) -> str:
        if not self.records:
            title = f"CommCounter{f' {self.name}' if self.name else ''}"
            return f"{title}: no collectives recorded."

        counts, sizes = self.counts(), self.bytes_by_op()
        lines = []
        header = f"CommCounter{f' {self.name}' if self.name else ''}"
        lines.append(header)
        lines.append("-" * max(58, len(header)))
        lines.append(f"{'op':<20}{'calls':>8}{'bytes':>16}{'avg':>14}")
        for op in sorted(counts, key=lambda o: -sizes[o]):
            n, b = counts[op], sizes[op]
            lines.append(f"{op:<20}{n:>8}{_human_bytes(b):>16}{_human_bytes(b / n):>14}")
        lines.append("-" * 58)
        lines.append(f"{'total':<20}{len(self.records):>8}{_human_bytes(self.total_bytes):>16}")

        tags = self.by_tag()
        if len(tags) > 1 or "(untagged)" not in tags:
            lines.append("")
            lines.append(f"{'by tag':<32}{'calls':>8}{'bytes':>16}")
            for t, (n, b) in sorted(tags.items(), key=lambda kv: -kv[1][1]):
                lines.append(f"{t:<32}{n:>8}{_human_bytes(b):>16}")
        return "\n".join(lines)

    def assert_equals(
        self,
        expected: dict[str, int | tuple[int, int | None]],
        *,
        exact: bool = True,
        byte_tol: float = 0.0,
    ) -> None:
        """Assert the collective counts, and optionally the byte totals.

        ``expected`` maps an op name to either a call count, or a
        ``(count, bytes)`` pair where ``bytes`` may be ``None`` to skip the
        byte check.

        With ``exact=True`` (the default), an op that fired but was not listed
        is a failure. That is usually what you want: "four all-reduces per
        layer" means four all-reduces and nothing else.

        The failure message prints the whole table rather than a dict diff,
        because when this fails you want to see what actually happened.
        """
        counts, sizes = self.counts(), self.bytes_by_op()
        problems: list[str] = []

        for op, spec in expected.items():
            want_n, want_b = spec if isinstance(spec, tuple) else (spec, None)
            got_n = counts.get(op, 0)
            if got_n != want_n:
                problems.append(f"  {op}: expected {want_n} call(s), saw {got_n}")
            if want_b is not None:
                got_b = sizes.get(op, 0)
                tol = max(byte_tol * want_b, 0)
                if abs(got_b - want_b) > tol:
                    problems.append(
                        f"  {op}: expected {_human_bytes(want_b)}, saw {_human_bytes(got_b)}"
                    )

        if exact:
            for op, got_n in counts.items():
                if op not in expected:
                    problems.append(f"  {op}: {got_n} unexpected call(s)")

        if problems:
            raise AssertionError(
                "Collective counts do not match.\n"
                + "\n".join(problems)
                + "\n\nWhat actually happened:\n"
                + self.summary()
            )

    def __repr__(self) -> str:
        return f"<CommCounter {len(self.records)} calls, {_human_bytes(self.total_bytes)}>"
