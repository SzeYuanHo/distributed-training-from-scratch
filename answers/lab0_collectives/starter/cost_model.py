from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = [
    "fit_alpha_beta",
    "predict_allreduce",
    "algorithm_bandwidth",
    "bus_bandwidth",
    "crossover_bytes",
]


def fit_alpha_beta(sizes_bytes: np.ndarray, times_s: np.ndarray) -> tuple[float, float]:
    sizes = np.asarray(sizes_bytes, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if sizes.ndim != 1 or times.ndim != 1 or len(sizes) != len(times):
        raise ValueError(
            f"sizes and times must be one-dimensional with equal lengths; got {len(sizes)} and {len(times)}"
        )
    if len(sizes) < 2:
        raise ValueError("at least two measurements are required")
    if not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(times)):
        raise ValueError("measurements must be finite")
    if np.any(sizes < 0) or np.any(times <= 0):
        raise ValueError("sizes must be non-negative and times must be positive")

    # relative weighting keeps the small-message intercept visible
    beta, alpha = np.polyfit(sizes, times, 1, w=1.0 / times)
    if alpha < 0:
        warnings.warn(
            "fitted alpha was negative; clamping it to zero",
            RuntimeWarning,
            stacklevel=2,
        )
        alpha = 0.0
    return float(alpha), float(beta)


def predict_allreduce(
    algorithm: str, n_ranks: int, n_bytes: int, alpha: float, beta: float
) -> float:
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be positive, got {n_ranks}")
    if n_bytes < 0:
        raise ValueError(f"n_bytes must be non-negative, got {n_bytes}")

    if algorithm == "naive":
        return 2 * (n_ranks - 1) * (alpha + beta * n_bytes)
    if algorithm == "ring":
        return 2 * (n_ranks - 1) * alpha + 2 * (n_ranks - 1) / n_ranks * beta * n_bytes
    if algorithm == "recursive_doubling":
        if n_ranks & (n_ranks - 1):
            raise ValueError(
                f"recursive doubling requires a power-of-two world size; got {n_ranks}"
            )
        return 2 * math.log2(n_ranks) * alpha + 2 * (n_ranks - 1) / n_ranks * beta * n_bytes
    raise ValueError(f"unknown all-reduce algorithm {algorithm!r}")


def algorithm_bandwidth(n_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    return n_bytes / seconds


def bus_bandwidth(n_bytes: int, seconds: float, n_ranks: int) -> float:
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be positive, got {n_ranks}")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    return 2 * (n_ranks - 1) / n_ranks * n_bytes / seconds


def crossover_bytes(
    n_ranks: int, alpha: float, beta_ring: float, beta_rd: float | None = None
) -> float:
    if n_ranks < 1 or n_ranks & (n_ranks - 1):
        raise ValueError(f"n_ranks must be a positive power of two; got {n_ranks}")
    if beta_rd is None:
        beta_rd = beta_ring

    latency_gap = 2 * ((n_ranks - 1) - math.log2(n_ranks)) * alpha
    bandwidth_advantage = 2 * (n_ranks - 1) / n_ranks * (beta_rd - beta_ring)
    if latency_gap <= 0:
        return 0.0 if bandwidth_advantage >= 0 else float("inf")
    if bandwidth_advantage <= 0:
        return float("inf")
    return latency_gap / bandwidth_advantage
