from __future__ import annotations

import math
from dataclasses import dataclass, field

from common import cluster_specs, memory

__all__ = [
    "ModelSpec",
    "Config",
    "ScoredConfig",
    "enumerate_configs",
    "score",
    "rank_configs",
    "report_table",
]


@dataclass(frozen=True)
class ModelSpec:
    n_params: int
    n_layers: int
    hidden: int
    n_heads: int
    seq_len: int
    global_batch_tokens: int
    dtype: str = "bf16"
    optimizer: str = "adamw"
    recompute: str = "none"

    def __post_init__(self) -> None:
        positive = {
            "n_params": self.n_params,
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "seq_len": self.seq_len,
            "global_batch_tokens": self.global_batch_tokens,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"model values must be positive: {', '.join(invalid)}")
        if self.global_batch_tokens % self.seq_len:
            raise ValueError(
                f"global_batch_tokens={self.global_batch_tokens:,} is not a whole "
                f"number of seq_len={self.seq_len:,} sequences"
            )

    def micro_batch_shape(self, config: Config) -> tuple[int, int]:
        """Return the complete sequences and tokens in one local micro-batch."""
        slots = config.dp * config.micro_batches
        global_sequences = self.global_batch_tokens // self.seq_len
        sequences, remainder = divmod(global_sequences, slots)
        if remainder:
            raise ValueError(
                f"{global_sequences} global sequences do not divide evenly across "
                f"dp={config.dp} and m={config.micro_batches} ({slots} micro-batch slots)"
            )
        return sequences, sequences * self.seq_len


@dataclass(frozen=True)
class Config:
    dp: int
    tp: int
    pp: int
    micro_batches: int
    zero_stage: int
    schedule: str = "1f1b"

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp

    def __str__(self) -> str:
        return (
            f"dp={self.dp} tp={self.tp} pp={self.pp} m={self.micro_batches} zero={self.zero_stage}"
        )


@dataclass
class ScoredConfig:
    config: Config
    micro_batch_sequences: int = 0
    micro_batch_tokens: int = 0
    memory_per_gpu: int = 0
    param_memory: int = 0
    activation_memory: int = 0
    comm_bytes_per_step: int = 0
    comm_dp: int = 0
    comm_tp: int = 0
    comm_pp: int = 0
    bubble_overhead: float = 0.0
    est_step_seconds: float = 0.0
    feasible: bool = False
    reasons: list[str] = field(default_factory=list)


def enumerate_configs(
    world_size: int,
    *,
    max_tp: int = 8,
    micro_batch_options: tuple[int, ...] = (1, 4, 8, 16, 32),
    zero_stages: tuple[int, ...] = (0, 1, 2, 3),
) -> list[Config]:
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")

    # Filtering split two ways on purpose. The tp cap and the m >= pp rule drop
    # candidates here, because a table with tp=64 rows in it is just longer. Head
    # divisibility goes to score() as a reason instead, so the table still shows
    # that tp=8 on a 12-head model was considered and why it lost.
    configs = []
    for dp in range(1, world_size + 1):
        for tp in range(1, min(max_tp, world_size) + 1):
            used = dp * tp
            if world_size % used:
                continue
            pp = world_size // used
            for micro_batches in micro_batch_options:
                if micro_batches < pp:
                    continue
                for zero_stage in zero_stages:
                    configs.append(Config(dp, tp, pp, micro_batches, zero_stage))
    return configs


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype.lower() in {"fp32", "float32", "f32"} else 2


def score(model: ModelSpec, config: Config, cluster: dict) -> ScoredConfig:
    result = ScoredConfig(config=config)
    if min(config.dp, config.tp, config.pp, config.micro_batches) < 1:
        raise ValueError(f"configuration values must be positive; got {config}")
    if model.recompute not in {"none", "selective", "full"}:
        result.reasons.append(f"unknown recompute mode {model.recompute!r}")
    if config.schedule not in {"1f1b", "afab"}:
        result.reasons.append(f"unknown schedule {config.schedule!r}")
    if config.pp > model.n_layers:
        result.reasons.append(f"{config.pp} pipeline stages exceed {model.n_layers} model layers")
    if model.n_heads % config.tp:
        result.reasons.append(f"{model.n_heads} heads do not divide across tp={config.tp}")
    if config.world_size > cluster.get("n_devices", config.world_size):
        result.reasons.append(
            f"needs {config.world_size} devices, cluster has {cluster.get('n_devices', 0)}"
        )
    if model.dtype.lower() in {"bf16", "bfloat16"} and not cluster.get("bf16", False):
        result.reasons.append("bf16 is not supported by this cluster profile")

    local_params = math.ceil(model.n_params / (config.tp * config.pp))
    ledger = memory.analytic_ledger(
        local_params,
        model.dtype,
        model.optimizer,
        config.zero_stage,
        config.dp,
    )

    layers = max(1, math.ceil(model.n_layers / config.pp))
    try:
        result.micro_batch_sequences, result.micro_batch_tokens = model.micro_batch_shape(config)
    except ValueError as exc:
        result.reasons.append(str(exc))

    inflight = (
        config.micro_batches if config.schedule == "afab" else min(config.pp, config.micro_batches)
    )
    if result.micro_batch_sequences:
        one_micro = memory.activation_estimate(
            result.micro_batch_sequences,
            model.seq_len,
            model.hidden,
            layers,
            model.n_heads,
            model.dtype,
            recompute=model.recompute,
            tensor_parallel=config.tp,
        )
        result.activation_memory = one_micro * inflight
    result.param_memory = ledger.states_total
    result.memory_per_gpu = result.param_memory + result.activation_memory

    elem = _dtype_bytes(model.dtype)
    grad_bytes = local_params * elem
    if config.dp > 1:
        result.comm_dp = int(2 * (config.dp - 1) / config.dp * grad_bytes)

    local_tokens = (
        result.micro_batch_tokens * config.micro_batches
        if result.micro_batch_tokens
        else math.ceil(model.global_batch_tokens / config.dp)
    )
    activation_bytes = local_tokens * model.hidden * elem
    if config.tp > 1:
        result.comm_tp = int(4 * layers * 2 * (config.tp - 1) / config.tp * activation_bytes)
    if config.pp > 1:
        neighbouring_boundaries = min(2, config.pp - 1)
        result.comm_pp = int(2 * neighbouring_boundaries * activation_bytes)
    result.comm_bytes_per_step = result.comm_dp + result.comm_tp + result.comm_pp
    result.bubble_overhead = (config.pp - 1) / config.micro_batches if config.pp > 1 else 0.0

    budget = cluster.get("vram_bytes", 0) * 0.85
    if result.memory_per_gpu > budget:
        result.reasons.append(
            f"needs {memory.format_bytes(result.memory_per_gpu)}, "
            f"{memory.format_bytes(budget)} available, "
            f"short by {memory.format_bytes(result.memory_per_gpu - budget)}"
        )

    low_precision = model.dtype.lower() in {"bf16", "bfloat16", "fp16", "float16", "f16"}
    peak_key = "peak_flops_fp16" if low_precision else "peak_flops_fp32"
    peak = cluster.get(peak_key, 0.0)
    compute_seconds = 0.0
    if peak > 0:
        useful_flops = 6 * model.n_params * model.global_batch_tokens
        if model.recompute == "full":
            useful_flops *= 4 / 3
        elif model.recompute == "selective":
            useful_flops *= 13 / 12
        compute_seconds = useful_flops / (
            peak * cluster_specs.DEFAULT_ACHIEVABLE_FRACTION * config.world_size
        )
        compute_seconds *= 1 + result.bubble_overhead

    bandwidth = cluster.get("interconnect_bytes_per_s", 0.0)
    alpha = cluster.get("alpha_s", 0.0)
    dp_seconds = tp_seconds = pp_seconds = 0.0
    if bandwidth > 0:
        dp_seconds = result.comm_dp / bandwidth
        tp_seconds = result.comm_tp / bandwidth
        pp_seconds = result.comm_pp / bandwidth
    if config.dp > 1:
        dp_seconds += 2 * (config.dp - 1) * alpha
    if config.tp > 1:
        tp_calls = 4 * layers * config.micro_batches
        tp_seconds += tp_calls * 2 * (config.tp - 1) * alpha
    if config.pp > 1:
        neighbouring_boundaries = min(2, config.pp - 1)
        pp_seconds += 2 * neighbouring_boundaries * config.micro_batches * alpha

    comm_seconds = 0.3 * dp_seconds + tp_seconds + 0.5 * pp_seconds
    result.est_step_seconds = compute_seconds + comm_seconds
    result.feasible = not result.reasons
    return result


def rank_configs(
    model: ModelSpec,
    configs: list[Config],
    cluster: dict,
    *,
    objective: str = "throughput",
) -> list[ScoredConfig]:
    if objective not in {"throughput", "memory", "comm"}:
        raise ValueError(f"unknown objective {objective!r}")
    scored = [score(model, config, cluster) for config in configs]
    metric = {
        "throughput": lambda row: row.est_step_seconds,
        "memory": lambda row: row.memory_per_gpu,
        "comm": lambda row: row.comm_bytes_per_step,
    }[objective]
    return sorted(scored, key=lambda row: (not row.feasible, metric(row)))


def report_table(scored: list[ScoredConfig], limit: int = 20) -> str:
    rows = [
        "   dp tp pp   m z seq/mb  tok/mb   memory/GPU     DP/step     TP/step     PP/step   total/step overhead  step(s)  status",
        "   -- -- -- --- - ------ ------- ------------ ----------- ----------- ----------- ------------ -------- ---------  ------",
    ]
    chosen = next((index for index, row in enumerate(scored) if row.feasible), None)
    for index, row in enumerate(scored[:limit]):
        config = row.config
        status = "yes" if row.feasible else "; ".join(row.reasons)
        rows.append(
            f"{'*' if index == chosen else ' '}  {config.dp:2d} {config.tp:2d} {config.pp:2d} "
            f"{config.micro_batches:3d} {config.zero_stage:d} "
            f"{row.micro_batch_sequences:6d} {row.micro_batch_tokens:7d} "
            f"{memory.format_bytes(row.memory_per_gpu):>12} "
            f"{memory.format_bytes(row.comm_dp):>11} "
            f"{memory.format_bytes(row.comm_tp):>11} "
            f"{memory.format_bytes(row.comm_pp):>11} "
            f"{memory.format_bytes(row.comm_bytes_per_step):>11} "
            f"{row.bubble_overhead:8.3f} {row.est_step_seconds:9.3f}  {status}"
        )
    return "\n".join(rows)


DEFAULT_PROBLEM = ModelSpec(
    n_params=7_000_000_000,
    n_layers=32,
    hidden=4096,
    n_heads=32,
    seq_len=4096,
    global_batch_tokens=1_048_576,
    dtype="bf16",
    optimizer="adamw",
    recompute="full",
)
