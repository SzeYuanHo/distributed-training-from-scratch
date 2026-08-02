"""Step 3: the planner. This is the lab.

Given a model and a cluster, enumerate the parallelism configurations, score each
on memory, communication and overhead, and rank them. Everything else in Lab 4 is
plumbing around this file.

Why it matters more than the composition code: anyone can wire two process groups
together. Producing a defensible answer to "we have 64 A100s and a 30B model, how
should we configure this" *before* burning a week of cluster time is the thing
people actually get asked, and the thing this whole repository has been building
toward.

Pure arithmetic. No torch, no distribution, so the tests run in milliseconds and
you can iterate on the model quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """What you are training.

    ``global_batch_tokens`` is the optimizer-step batch across all data-parallel
    replicas. Because this planner schedules complete, fixed-length sequences,
    it must be an integer multiple of ``seq_len``.
    """

    n_params: int
    n_layers: int
    hidden: int
    n_heads: int
    seq_len: int
    global_batch_tokens: int
    dtype: str = "bf16"
    optimizer: str = "adamw"
    recompute: str = "none"  # none | selective | full

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
        """Return ``(sequences, tokens)`` in one scheduled micro-batch.

        A global optimizer-step batch is split first across ``dp`` replicas and
        then across ``m`` gradient-accumulation micro-batches. TP and PP ranks
        cooperate on the same micro-batch, so neither divides it further.

        The sequence count must divide exactly. Padding would change the stated
        global batch and silently corrupt every memory and communication estimate.
        """
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
    """One candidate parallelism configuration."""

    dp: int
    tp: int
    pp: int
    micro_batches: int
    zero_stage: int
    schedule: str = "1f1b"  # 1f1b | afab, only relevant when pp > 1

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp

    def __str__(self) -> str:
        return (
            f"dp={self.dp} tp={self.tp} pp={self.pp} m={self.micro_batches} zero={self.zero_stage}"
        )


@dataclass
class ScoredConfig:
    """A candidate with its predicted costs."""

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
    """Every valid ``(dp, tp, pp, m, z)`` for this world size.

    Constraints worth encoding rather than discovering later:

    * ``dp * tp * pp == world_size``, exactly.
    * ``tp <= max_tp``, because tensor parallelism past a fast-interconnect domain
      is dominated by communication. Lab 2 derived where that limit falls; 8 is the
      usual practical answer on an NVLink node.
    * ``tp`` should divide the head count. The caller passes ``max_tp``, so you can
      either filter here or leave it to ``score`` to mark infeasible. Filtering
      here keeps the table shorter; marking infeasible keeps it honest about what
      was considered. Pick one and say which.
    * ``micro_batches >= pp``, or the pipeline never reaches a steady state and
      1F1B degenerates to AFAB.

    Returns:
        Every valid combination. At world_size 8 this is a few dozen; at 512 it is
        thousands, which is fine because scoring is arithmetic.
    """
    raise NotImplementedError("Step 3a: nested loops over the divisors, with the filters.")


def score(model: ModelSpec, config: Config, cluster: dict) -> ScoredConfig:
    """Predict memory, communication and overhead for one configuration.

    Args:
        model: what you are training.
        config: how you propose to split it.
        cluster: a profile from ``common.cluster_specs.PROFILES``.

    Returns:
        A :class:`ScoredConfig` with every field filled and ``reasons`` explaining
        any infeasibility.

    Build it from the four models you already have. THEORY section 2 has the
    composition; the pieces are:

    **Memory.** ``common.memory.analytic_ledger`` for the states, with
    ``n_params`` divided by ``tp * pp`` and ``world_size`` set to ``dp`` (ZeRO
    shards across the data-parallel dimension only). Then
    ``common.memory.activation_estimate`` for the activations. Its batch size is
    exactly ``global_batch_tokens / (dp * micro_batches * seq_len)`` sequences;
    reject a configuration when that is not an integer. Divide the layer count
    by ``pp`` and multiply by the in-flight micro-batch count, which is
    ``micro_batches`` under AFAB and about ``pp`` under 1F1B.

    **Communication.** Three terms on three groups:

    * DP: one all-reduce of the gradients over ``dp`` ranks, so
      ``2(dp-1)/dp * grad_bytes``. Under ZeRO it is a reduce-scatter plus an
      all-gather, which moves the same bytes.
    * TP: four all-reduces per layer of a ``(b, s, h)`` tensor over ``tp`` ranks.
      Independent of ``tp`` in volume, which is Lab 2's whole point.
    * PP: two point-to-point transfers per micro-batch per stage boundary.

    **Bubble overhead.** ``(pp - 1) / micro_batches``, zero when ``pp == 1``.
    This is overhead relative to ideal pipeline time, not the fraction of elapsed
    time spent idle. The elapsed-time idle fraction is
    ``(pp - 1) / (micro_batches + pp - 1)``.

    **Feasibility.** The micro-batch must divide into complete sequences, the
    requested dtype must be supported by the cluster profile, and memory must fit
    in VRAM with headroom. Record *why* a candidate fails in ``reasons``:
    "needs 94 GiB, 80 GiB available" is actionable and "infeasible" is not.

    **Estimated step time.** Compute divided by the world size, inflated by the
    bubble, plus the exposed communication. Rough, and rough is fine: you are
    ranking candidates, not promising a schedule. State the assumption about which
    communication overlaps.
    """
    raise NotImplementedError("Step 3b: assemble the four models from Labs 0-3.")


def rank_configs(
    model: ModelSpec, configs: list[Config], cluster: dict, *, objective: str = "throughput"
) -> list[ScoredConfig]:
    """Score every candidate and sort best first.

    Args:
        objective: ``"throughput"`` sorts by estimated step time.
            ``"memory"`` sorts by memory per GPU, for when you are trying to fit at
            all rather than to go fast. ``"comm"`` sorts by bytes moved, which is
            what you want when the interconnect is shared with other jobs.

    Infeasible configurations sort last, but they stay in the list. A table that
    silently drops what did not fit hides the most useful information in it, which
    is *how close* the near misses came.
    """
    raise NotImplementedError("Step 3c.")


def report_table(scored: list[ScoredConfig], limit: int = 20) -> str:
    """The configuration table for the scaling report.

    One row per candidate: the configuration, memory per GPU, communication per
    step, bubble, estimated step time, and feasibility. Sorted, with the chosen
    one marked.

    This table is the deliverable. Make it readable by someone who did not write
    it: units in the header, no bare byte counts, and a column that says why an
    infeasible row is infeasible.
    """
    raise NotImplementedError("Step 3d: format it for a human, not for a parser.")


# --------------------------------------------------------------------------- #
# the default problem
# --------------------------------------------------------------------------- #

DEFAULT_PROBLEM = ModelSpec(
    n_params=7_000_000_000,
    n_layers=32,
    hidden=4096,
    n_heads=32,
    seq_len=4096,
    global_batch_tokens=1_048_576,  # 256 complete sequences
    dtype="bf16",
    optimizer="adamw",
    recompute="full",
)
"""A 7B model, the size where these decisions start to matter and still fits on a
node you might plausibly get access to. The 1,048,576-token batch is exactly 256
sequences, so the default DP and micro-batch options divide it without padding.
Full recomputation is explicit because the unrecomputed 4096-token activations do
not fit on this eight-GPU problem. Swap it for your own problem; the planner does
not care.
"""
