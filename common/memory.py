"""The memory ledger: what you predicted, what actually got allocated, and why they differ.

Lab 1B is built on this file. Before you read ``analytic_ledger``, write it
yourself in ``labs/lab1b_zero_memory/starter/ledger.py`` from the ZeRO paper.
Nothing stops you looking first. The lab is worth much less if you do.

The headline arithmetic, per parameter, for mixed-precision AdamW::

    fp16/bf16 parameters      2 bytes
    fp16/bf16 gradients       2 bytes
    fp32 master parameters    4 bytes
    fp32 Adam m               4 bytes
    fp32 Adam v               4 bytes
                             --------
                             16 bytes per parameter

Conventionally written ``2 + 2 + K`` with ``K = 12`` for the optimizer group,
which is the form the ZeRO paper uses when it shards things::

    ZeRO-0 (plain DDP)   2 + 2 + 12
    ZeRO-1               2 + 2 + 12/N       optimizer states sharded
    ZeRO-2               2 + (2 + 12)/N     gradients sharded too
    ZeRO-3               (2 + 2 + 12)/N     parameters sharded too

Activations are not in that table, and at large batch times sequence length they
dominate everything in it. ``activation_estimate`` covers them separately.

One communication fact that surprises people, and which Lab 1B asserts as a
test: ZeRO-1 and ZeRO-2 move exactly as many bytes as plain DDP. An all-reduce
is a reduce-scatter followed by an all-gather, so replacing one with the other
two changes where the bytes go, not how many there are. ZeRO-3 is different: it
adds parameter all-gathers in both forward and backward.
"""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "BYTES_PER_DTYPE",
    "MemoryLedger",
    "analytic_ledger",
    "activation_estimate",
    "measured_peak",
    "reset_peak",
    "reconcile",
    "format_bytes",
    "count_parameters",
    "fits_in",
]

BYTES_PER_DTYPE = {
    "fp32": 4,
    "float32": 4,
    "f32": 4,
    "fp16": 2,
    "float16": 2,
    "f16": 2,
    "half": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "float8": 1,
    "int8": 1,
}

# Per-parameter fp32 state each optimizer keeps, on top of master weights.
OPTIMIZER_STATE_SLOTS = {
    "sgd": 0,  # plain SGD keeps nothing
    "sgd_momentum": 1,  # one momentum buffer
    "adam": 2,  # m and v
    "adamw": 2,
    "adafactor": 0,  # factored, so roughly O(sqrt) not O(n); treated as 0 here
    "lion": 1,
}


def format_bytes(n: float) -> str:
    sign = "-" if n < 0 else ""
    n = abs(float(n))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{sign}{n:.0f} {unit}" if unit == "B" else f"{sign}{n:.2f} {unit}"
        n /= 1024.0
    return f"{sign}{n:.2f} TiB"


def _dtype_bytes(dtype: str | torch.dtype) -> int:
    if isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    key = str(dtype).lower().replace("torch.", "")
    if key not in BYTES_PER_DTYPE:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(set(BYTES_PER_DTYPE))}")
    return BYTES_PER_DTYPE[key]


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------- #
# the analytic side
# --------------------------------------------------------------------------- #


@dataclass
class MemoryLedger:
    """Per-rank byte counts for one training configuration."""

    n_params: int
    world_size: int
    zero_stage: int
    compute_dtype: str
    optimizer: str

    params: int = 0
    grads: int = 0
    master_params: int = 0
    optimizer_states: int = 0
    activations: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    @property
    def states_total(self) -> int:
        """Everything that persists across a step. Excludes activations."""
        return self.params + self.grads + self.master_params + self.optimizer_states

    @property
    def total(self) -> int:
        return self.states_total + self.activations + sum(self.extra.values())

    @property
    def bytes_per_param(self) -> float:
        return self.states_total / self.n_params if self.n_params else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "grads": self.grads,
            "master_params": self.master_params,
            "optimizer_states": self.optimizer_states,
            "activations": self.activations,
            **self.extra,
            "states_total": self.states_total,
            "total": self.total,
        }

    def __str__(self) -> str:
        rows = [
            ("parameters", self.params),
            ("gradients", self.grads),
            ("fp32 master params", self.master_params),
            ("optimizer states", self.optimizer_states),
            *[(k, v) for k, v in self.extra.items()],
        ]
        if self.activations:
            rows.append(("activations (est.)", self.activations))

        width = max(len(r[0]) for r in rows) + 2
        head = (
            f"Memory ledger: {self.n_params / 1e6:.2f} M params, "
            f"{self.compute_dtype}, {self.optimizer}, "
            f"ZeRO-{self.zero_stage}, N={self.world_size}"
        )
        lines = [head, "-" * max(len(head), width + 26)]
        for label, value in rows:
            share = value / self.total * 100 if self.total else 0
            lines.append(f"{label:<{width}}{format_bytes(value):>12}   {share:5.1f}%")
        lines.append("-" * max(len(head), width + 26))
        lines.append(f"{'states subtotal':<{width}}{format_bytes(self.states_total):>12}")
        lines.append(f"{'total':<{width}}{format_bytes(self.total):>12}")
        lines.append(f"{'bytes/param (states)':<{width}}{self.bytes_per_param:>12.2f}")
        return "\n".join(lines)


def analytic_ledger(
    n_params: int,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    zero_stage: int = 0,
    world_size: int = 1,
    *,
    master_weights: bool | None = None,
    activations: int = 0,
) -> MemoryLedger:
    """Predicted per-rank memory for a training configuration.

    Parameters
    ----------
    n_params:
        Trainable parameter count, usually ``count_parameters(model)``.
    dtype:
        The compute dtype. ``bf16`` or ``fp16`` implies mixed precision, which
        means an fp32 master copy unless you override ``master_weights``.
        ``fp32`` means no separate master copy, because the parameters already
        are the master copy.
    optimizer:
        Determines how many fp32 state tensors per parameter. See
        ``OPTIMIZER_STATE_SLOTS``.
    zero_stage:
        0 replicates everything (plain DDP). 1 shards optimizer states.
        2 also shards gradients. 3 also shards parameters.
    world_size:
        Number of ranks to divide sharded components by.

    Returns a :class:`MemoryLedger`. Print it.

    Worked example, 1 B parameters, bf16 AdamW, 8 ranks::

        stage 0:  2 + 2 + 12       = 16 GB per rank
        stage 1:  2 + 2 + 12/8     =  5.5 GB
        stage 2:  2 + (2 + 12)/8   =  3.75 GB
        stage 3:  (2 + 2 + 12)/8   =  2 GB

    These values use decimal GB because parameter counts and bytes per
    parameter are decimal quantities; :func:`format_bytes` reports measured
    byte totals in binary KiB/MiB/GiB.
    """
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2 or 3; got {zero_stage}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1; got {world_size}")

    compute_bytes = _dtype_bytes(dtype)
    mixed = compute_bytes < 4
    if master_weights is None:
        master_weights = mixed

    opt_key = optimizer.lower()
    if opt_key not in OPTIMIZER_STATE_SLOTS:
        raise ValueError(f"unknown optimizer {optimizer!r}; known: {sorted(OPTIMIZER_STATE_SLOTS)}")
    slots = OPTIMIZER_STATE_SLOTS[opt_key]

    # Unsharded byte counts.
    p_bytes = n_params * compute_bytes
    g_bytes = n_params * compute_bytes
    master_bytes = n_params * 4 if master_weights else 0
    opt_bytes = n_params * 4 * slots

    # ZeRO shards progressively. Note master weights travel with the optimizer
    # group: they are part of the 12 in "2 + 2 + 12", not part of the 2 + 2.
    n = world_size
    if zero_stage >= 1:
        master_bytes //= n
        opt_bytes //= n
    if zero_stage >= 2:
        g_bytes //= n
    if zero_stage >= 3:
        p_bytes //= n

    return MemoryLedger(
        n_params=n_params,
        world_size=world_size,
        zero_stage=zero_stage,
        compute_dtype=str(dtype),
        optimizer=optimizer,
        params=p_bytes,
        grads=g_bytes,
        master_params=master_bytes,
        optimizer_states=opt_bytes,
        activations=activations,
    )


def activation_estimate(
    batch_size: int,
    seq_len: int,
    hidden: int,
    n_layers: int,
    n_heads: int,
    dtype: str = "bf16",
    *,
    recompute: str = "none",
    tensor_parallel: int = 1,
) -> int:
    """Rough activation memory per rank, following Korthikanti et al. section 4.

    Per transformer layer with no recomputation and no parallelism, stored
    activations run to roughly::

        s*b*h * (34 + 5 * a*s/h)

    bytes in 16-bit, where ``a`` is the head count. The ``a*s/h`` term is the
    attention score matrix, and it is quadratic in sequence length, which is
    why long-context training is an activation-memory problem before it is
    anything else.

    ``recompute="full"`` keeps only the layer inputs, dropping the estimate to
    ``2*s*b*h`` per layer at the cost of a second forward pass. ``"selective"``
    drops just the quadratic attention term, which buys most of the saving for
    a small fraction of the recompute.

    This is an estimate. It ignores the allocator, workspace buffers, and
    anything your framework caches. Lab 1B is about the size of the gap.
    """
    elem = _dtype_bytes(dtype)
    s, b, h, a, t = seq_len, batch_size, hidden, n_heads, max(1, tensor_parallel)

    if recompute == "full":
        per_layer = elem * s * b * h
    elif recompute == "selective":
        per_layer = (elem / 2) * s * b * h * 34 / t
    else:
        per_layer = (elem / 2) * s * b * h * (34 / t + 5 * a * s / (h * t))

    return int(per_layer * n_layers)


def fits_in(ledger: MemoryLedger, vram_bytes: int, *, headroom: float = 0.15) -> tuple[bool, str]:
    """Will this configuration fit, with a margin for the allocator?

    Fifteen percent headroom is not superstition. The caching allocator
    fragments, cuBLAS grabs a workspace, and the last block you need is always
    the one that does not fit. Lab 1B has you calibrate this number against
    your own device.
    """
    budget = vram_bytes * (1 - headroom)
    ok = ledger.total <= budget
    verdict = (
        f"{format_bytes(ledger.total)} needed, {format_bytes(budget)} usable "
        f"of {format_bytes(vram_bytes)} ({headroom:.0%} headroom). "
        + ("Fits." if ok else f"Short by {format_bytes(ledger.total - budget)}.")
    )
    return ok, verdict


# --------------------------------------------------------------------------- #
# the measured side
# --------------------------------------------------------------------------- #


def reset_peak(device: torch.device | str | None = None) -> None:
    """Zero the peak counter before a region you want to measure.

    Pass an explicit CPU device to select RSS measurement on a CUDA-capable
    host. With no device, CUDA remains the default when it is available.
    """
    selected = torch.device(device) if device is not None else None
    use_cuda = selected.type == "cuda" if selected is not None else torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(selected)
        torch.cuda.synchronize(selected)
    gc.collect()


def measured_peak(device: torch.device | str | None = None) -> dict[str, int]:
    """Peak memory since the last ``reset_peak``.

    On CUDA this reports ``max_memory_allocated`` (tensors) and
    ``max_memory_reserved`` (what the caching allocator took from the driver).
    The gap between them is fragmentation, and on a small model it can be large
    in relative terms.

    On CPU there is no equivalent counter, so this falls back to peak RSS from
    ``resource.getrusage`` on POSIX and the process working set on Windows.
    That measures the whole process including the interpreter, so treat it as a
    trend rather than an absolute. Pass an explicit CPU device when the measured
    model is on CPU but the host also has CUDA.
    """
    selected = torch.device(device) if device is not None else None
    use_cuda = selected.type == "cuda" if selected is not None else torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(selected)
        return {
            "allocated": torch.cuda.max_memory_allocated(selected),
            "reserved": torch.cuda.max_memory_reserved(selected),
            "current": torch.cuda.memory_allocated(selected),
            "source": 0,  # 0 = cuda
        }

    peak_rss = 0
    if sys.platform != "win32":
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports kilobytes, macOS reports bytes.
            peak_rss = usage * 1024 if sys.platform.startswith("linux") else usage
        except (ImportError, OSError):
            peak_rss = 0
    else:
        try:
            import psutil

            info = psutil.Process(os.getpid()).memory_info()
            peak_rss = getattr(info, "peak_wset", info.rss)
        except Exception:
            peak_rss = 0

    return {"allocated": peak_rss, "reserved": peak_rss, "current": peak_rss, "source": 1}


def reconcile(
    ledger: MemoryLedger, measured: dict[str, int] | None = None, *, label: str = ""
) -> str:
    """Print the prediction and the measurement side by side, and name the gap.

    Reading the output: a measured figure *above* the analytic states total is
    normal and expected. The difference is activations, the allocator's
    rounding, cuBLAS workspaces, and on CPU the interpreter itself. A measured
    figure *below* the prediction means either the model is smaller than you
    think or the optimizer has not stepped yet, because Adam allocates ``m``
    and ``v`` lazily on the first ``step()``.
    """
    measured = measured or measured_peak()
    on_cuda = measured.get("source", 0) == 0

    predicted = ledger.total
    actual = measured["allocated"]
    gap = actual - predicted
    ratio = actual / predicted if predicted else float("nan")

    lines = [
        f"Memory reconciliation{f' [{label}]' if label else ''}",
        "=" * 62,
        str(ledger),
        "",
        f"{'measured peak allocated':<30}{format_bytes(actual):>14}",
    ]
    if on_cuda:
        lines.append(f"{'measured peak reserved':<30}{format_bytes(measured['reserved']):>14}")
        frag = measured["reserved"] - actual
        lines.append(f"{'  allocator overhead':<30}{format_bytes(frag):>14}")
    else:
        lines.append("  (CPU: this is process RSS, which includes the interpreter)")

    lines += [
        "-" * 62,
        f"{'predicted total':<30}{format_bytes(predicted):>14}",
        f"{'unexplained gap':<30}{format_bytes(gap):>14}   ({ratio:.2f}x)",
        "",
        "Where a positive gap usually comes from:",
        "  activations       not in the ledger unless you passed activations=",
        "  workspaces        cuBLAS and cuDNN scratch buffers",
        "  fragmentation     the caching allocator rounds up block sizes",
        "  autograd graph    saved tensors held until backward completes",
    ]
    if gap < 0:
        lines += [
            "",
            "A negative gap means less was allocated than predicted. Check:",
            "  did optimizer.step() run? Adam allocates m and v lazily.",
            "  is the model actually the size you passed to analytic_ledger?",
        ]
    return "\n".join(lines)
