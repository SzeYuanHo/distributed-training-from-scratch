"""Shared library for the distributed-training labs.

Everything here ships finished. It is infrastructure, not curriculum: if you
have to debug ``CommCounter`` before you can debug your own all-reduce, the lab
has failed you. Read these files freely, and treat them the way you would treat
a standard library.

The one exception is ``memory.py``. Lab 1B asks you to reconstruct the ZeRO
memory ledger from the paper before you look at ``analytic_ledger``. Nothing
enforces that, but the lab is worth much less if you skip it.

Module map:

===================== =======================================================
``repo_env``          Cache redirection. Import before torch.
``dist_utils``        Process groups, spawning, seeding, rank-safe printing.
``collectives``       Instrumented wrappers. Lab code calls these, not ``dist``.
``comm_counter``      Counts and sizes every collective. The main instrument.
``timers``            Region timers, profiler traces, Gantt charts.
``memory``            Analytic memory ledger and measured peak, side by side.
``reference_gpt``     FROZEN nanoGPT-derived model. Never edit.
``data``              tiny-shakespeare, char tokenizer, rank-disjoint batches.
``predictions``       Loads predictions.yaml and prints the reconciliation.
``testing``           Comparison helpers with failure messages worth reading.
===================== =======================================================

Imports stay lazy. ``mp.spawn`` re-imports this package inside every rank, so a
heavy top-level import here costs you once per process.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "repo_env",
    "dist_utils",
    "collectives",
    "comm_counter",
    "timers",
    "memory",
    "reference_gpt",
    "data",
    "predictions",
    "testing",
]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
