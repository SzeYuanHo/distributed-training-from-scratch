from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Ledger", "ledger_by_hand", "activation_bytes", "predict_fits", "min_ranks_for", "BYTES"]

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}
_OPTIMIZER_SLOTS = {"adamw": 2, "adam": 2, "sgd_momentum": 1, "sgd": 0}


@dataclass
class Ledger:
    params: int = 0
    grads: int = 0
    master_params: int = 0
    optimizer_states: int = 0

    @property
    def total(self) -> int:
        return self.params + self.grads + self.master_params + self.optimizer_states


def ledger_by_hand(
    n_params: int,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    zero_stage: int = 0,
    world_size: int = 1,
) -> Ledger:
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, got {n_params}")
    if dtype not in BYTES:
        raise ValueError(f"unknown dtype {dtype!r}")
    if optimizer not in _OPTIMIZER_SLOTS:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2 or 3; got {zero_stage}")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")

    params = n_params * BYTES[dtype]
    grads = n_params * BYTES[dtype]
    master = n_params * 4 if dtype != "fp32" else 0
    states = n_params * 4 * _OPTIMIZER_SLOTS[optimizer]

    # the fp32 master copy belongs to the optimizer shard
    if zero_stage >= 1:
        master //= world_size
        states //= world_size
    if zero_stage >= 2:
        grads //= world_size
    if zero_stage >= 3:
        params //= world_size
    return Ledger(params, grads, master, states)


def activation_bytes(
    batch_size: int,
    seq_len: int,
    hidden: int,
    n_layers: int,
    n_heads: int,
    dtype: str = "bf16",
    recompute: str = "none",
) -> int:
    if dtype not in BYTES:
        raise ValueError(f"unknown dtype {dtype!r}")
    if recompute not in ("none", "selective", "full"):
        raise ValueError(f"unknown recompute mode {recompute!r}")
    if min(batch_size, seq_len, hidden, n_layers, n_heads) < 0 or hidden == 0:
        raise ValueError("activation dimensions must be non-negative and hidden must be positive")

    elem = BYTES[dtype]
    core = seq_len * batch_size * hidden
    if recompute == "full":
        per_layer = elem * core
    elif recompute == "selective":
        per_layer = elem / 2 * core * 34
    else:
        per_layer = elem / 2 * core * (34 + 5 * n_heads * seq_len / hidden)
    return int(per_layer * n_layers)


def predict_fits(
    n_params: int,
    vram_bytes: int,
    *,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    zero_stage: int = 0,
    world_size: int = 1,
    activations: int = 0,
    headroom: float = 0.15,
) -> tuple[bool, str]:
    if vram_bytes < 0 or activations < 0:
        raise ValueError("memory byte counts must be non-negative")
    if not 0 <= headroom < 1:
        raise ValueError(f"headroom must be in [0, 1), got {headroom}")
    ledger = ledger_by_hand(n_params, dtype, optimizer, zero_stage, world_size)
    needed = ledger.total + activations
    usable = int(vram_bytes * (1 - headroom))
    fits = needed <= usable
    delta = usable - needed
    if fits:
        verdict = f"fits with {delta / 1024**3:.2f} GiB spare"
    else:
        verdict = f"short by {-delta / 1024**3:.2f} GiB"
    return (
        fits,
        f"{needed / 1024**3:.2f} GiB needed, {usable / 1024**3:.2f} GiB usable; {verdict}",
    )


def min_ranks_for(
    n_params: int,
    vram_bytes: int,
    *,
    zero_stage: int = 1,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    activations_per_rank: int = 0,
    headroom: float = 0.15,
    max_ranks: int = 1024,
) -> int | None:
    if max_ranks < 1:
        return None
    for world in range(1, max_ranks + 1):
        fits, _ = predict_fits(
            n_params,
            vram_bytes,
            dtype=dtype,
            optimizer=optimizer,
            zero_stage=zero_stage,
            world_size=world,
            activations=activations_per_rank,
            headroom=headroom,
        )
        if fits:
            return world
    return None
