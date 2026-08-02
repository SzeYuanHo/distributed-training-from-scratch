"""Named hardware profiles, with sources, for the planner and the scaling curves.

These are vendor-published peak numbers. Real achieved figures are lower, often by
30 to 50 percent, and the gap is what MFU measures. Using peaks makes the
comparisons between configurations honest even though the absolute predictions are
optimistic, and Lab 4 asks you to apply an achievable-fraction correction and say
what you chose.

Every number here has a source. If you change one, change the source too.
"""

from __future__ import annotations

__all__ = ["PROFILES", "describe", "achievable"]

# A planning assumption, not a promised or measured utilisation. Lab 4 varies
# this number and asks the reader to replace it with a workload-specific value.
DEFAULT_ACHIEVABLE_FRACTION = 0.45

PROFILES: dict[str, dict] = {
    "2xT4 (PCIe 3.0)": {
        "n_devices": 2,
        "vram_bytes": 16 * 1024**3,
        # NVIDIA T4 datasheet: 65 TFLOPS fp16 tensor, 8.1 TFLOPS fp32.
        "peak_flops_fp16": 65e12,
        "peak_flops_fp32": 8.1e12,
        # PCIe 3.0 x16 is 15.75 GB/s theoretical; ~12 GB/s achieved is typical.
        "interconnect_bytes_per_s": 12e9,
        "alpha_s": 10e-6,
        "bf16": False,  # Turing. No bf16 tensor cores.
        "note": (
            "An illustrative two-T4 profile. Hosted availability and topology vary; "
            "Turing supports fp16 tensor cores but not bf16 tensor cores."
        ),
        "source": (
            "NVIDIA T4 datasheet for compute/VRAM; 12 GB/s and 10 us are explicit "
            "PCIe planning assumptions to replace with measured values"
        ),
    },
    "8xA100 (NVLink 3.0)": {
        "n_devices": 8,
        "vram_bytes": 80 * 1024**3,
        # A100 80GB SXM: 312 TFLOPS bf16/fp16 tensor with sparsity off.
        "peak_flops_fp16": 312e12,
        "peak_flops_fp32": 19.5e12,
        # NVLink 3.0: 600 GB/s bidirectional per GPU, so ~300 GB/s each way.
        "interconnect_bytes_per_s": 300e9,
        "alpha_s": 5e-6,
        "bf16": True,
        "note": "The canonical single-node training box. TP up to 8 stays on NVLink.",
        "source": "NVIDIA A100 datasheet; NVLink 3.0 specification",
    },
    "2 nodes x 8xA100 (InfiniBand)": {
        "n_devices": 16,
        "vram_bytes": 80 * 1024**3,
        "peak_flops_fp16": 312e12,
        "peak_flops_fp32": 19.5e12,
        # Cross-node HDR InfiniBand: 200 Gb/s = 25 GB/s. An order of magnitude
        # below NVLink, which is the entire reason TP does not cross node boundaries.
        "interconnect_bytes_per_s": 25e9,
        "alpha_s": 15e-6,
        "bf16": True,
        "note": (
            "Intra-node is NVLink at 300 GB/s, inter-node is InfiniBand at 25 GB/s. "
            "The 12x gap is why tensor parallelism stays inside a node and pipeline "
            "or data parallelism crosses between them."
        ),
        "source": "NVIDIA A100 datasheet; HDR InfiniBand 200 Gb/s",
    },
    "gloo on localhost": {
        "n_devices": 0,
        "vram_bytes": 0,
        "peak_flops_fp16": 0.0,
        "peak_flops_fp32": 0.0,
        # Placeholders. Replace with the alpha and beta you fitted in Lab 0; that
        # is the point of having fitted them.
        "interconnect_bytes_per_s": 2e9,
        "alpha_s": 30e-6,
        "bf16": True,
        "note": (
            "Tier 1. The numbers here are placeholders: substitute your own Lab 0 "
            "fit. Loopback bandwidth is a memcpy and alpha is dominated by python, "
            "not by any network."
        ),
        "source": "measure it yourself with labs/lab0_collectives/bench/sweep.py",
    },
}


def achievable(
    profile: str, fraction: float = DEFAULT_ACHIEVABLE_FRACTION, precision: str = "fp16"
) -> float:
    """Peak FLOP/s scaled by an achievable fraction.

    Peak is what the datasheet promises with every tensor core saturated and no
    memory traffic. Real training reaches a fraction of it, and quoting peak in a
    capacity plan is how you end up promising a schedule you cannot hit.
    """
    spec = PROFILES[profile]
    key = "peak_flops_fp16" if precision in ("fp16", "bf16") else "peak_flops_fp32"
    return spec[key] * fraction


def describe(profile: str) -> str:
    spec = PROFILES[profile]
    return (
        f"{profile}\n"
        f"  devices              {spec['n_devices']}\n"
        f"  VRAM per device      {spec['vram_bytes'] / 1024**3:.0f} GiB\n"
        f"  peak fp16            {spec['peak_flops_fp16'] / 1e12:.0f} TFLOP/s\n"
        f"  interconnect         {spec['interconnect_bytes_per_s'] / 1e9:.0f} GB/s\n"
        f"  latency              {spec['alpha_s'] * 1e6:.0f} us\n"
        f"  bf16 tensor cores    {'yes' if spec['bf16'] else 'no'}\n"
        f"  {spec['note']}\n"
        f"  source: {spec['source']}"
    )


if __name__ == "__main__":
    for name in PROFILES:
        print(describe(name))
        print()
