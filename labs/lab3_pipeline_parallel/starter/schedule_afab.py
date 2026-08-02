"""Step 3: all-forward-all-backward, the GPipe schedule.

Run every micro-batch's forward pass, then every micro-batch's backward pass.
Simple to write, simple to reason about, and it holds every micro-batch's
activations at once. That memory cost is what 1F1B fixes.

Two things to write: a schedule *simulation* that produces the ordering as a list
of operations with no tensors involved, and the executor that runs it. Separating
them means the ordering can be tested in microseconds without spawning anything,
and an off-by-one in the warmup count fails before any message is sent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Op", "afab_schedule", "run_afab", "max_inflight"]


@dataclass(frozen=True)
class Op:
    """One scheduled operation.

    ``kind`` is ``"F"`` or ``"B"``. ``micro`` is the micro-batch index.

    Making the schedule a data structure rather than control flow is what lets
    ``test_00_math.py`` check the ordering exactly, and it is also how
    ``torch.distributed.pipelining`` is built.
    """

    kind: str
    micro: int

    def __str__(self) -> str:
        return f"{self.kind}{self.micro}"


def afab_schedule(stage_index: int, n_stages: int, n_micro: int) -> list[Op]:
    """The operation sequence for one stage under AFAB.

    Every stage runs `n_micro` forwards then `n_micro` backwards. The stages differ
    only in *when* they can start, and that is handled by the communication
    dependencies rather than by the schedule.

    Backward order matters: micro-batch `m-1`'s backward runs first, because it is
    the one whose forward finished last and whose activations are on top of the
    stack. Running them in forward order also works numerically and holds
    activations longer.

    Returns:
        Ops in execution order. Length ``2 * n_micro``.
    """
    raise NotImplementedError("Step 3a: n_micro forwards, then n_micro backwards.")


def max_inflight(schedule: list[Op]) -> int:
    """Peak number of micro-batches whose activations are held simultaneously.

    Walk the schedule: a forward adds one, a backward releases one. Return the
    maximum.

    For AFAB this comes out as `n_micro`. For 1F1B it comes out around `p`. That
    single number is the difference between the two schedules, and
    ``test_00_math.py`` asserts it without allocating a single tensor.
    """
    raise NotImplementedError("Step 3b.")


def run_afab(
    stage, n_micro: int, batches: list, targets: list | None = None, event_log=None
) -> float:
    """Execute the AFAB schedule on this rank's stage.

    Args:
        stage: this rank's ``PipelineStage``.
        n_micro: number of micro-batches.
        batches: this rank's inputs. Only the first stage uses them; other stages
            receive their input over the wire.
        targets: only the last stage uses them.
        event_log: optional ``common.timers.EventLog`` for the Gantt chart. Wrap
            each forward and backward in ``event_log.span(...)`` so the deliverable
            has something to draw.

    Returns:
        The summed loss across micro-batches, on the last stage. Zero elsewhere.

    **Loss scaling.** Each micro-batch's loss is scaled by `1/n_micro` before
    backward, exactly as in Lab 1's gradient accumulation, so that the summed
    gradient equals the gradient of one large batch. Skip the scaling and your
    gradients are `n_micro` times too large, the loss falls, and the run diverges
    later.

    **Holding the graph.** A forward pass whose output tensor goes out of scope
    has its graph freed, and the backward for that micro-batch then fails with
    "trying to backward through the graph a second time" or produces nothing.
    Keep the outputs in a list across the whole forward phase. That list *is* the
    activation memory this schedule is criticised for.

    **Crossing the boundary in the backward pass.** Stage `i` receives a gradient
    with respect to its output, calls ``backward(gradient=received)`` on the
    output tensor it saved, and then sends ``input_tensor.grad`` to stage `i-1`.
    The input tensor has a gradient only because ``recv_activation`` marked it as
    requiring grad, which is the point that docstring laboured.
    """
    raise NotImplementedError("Step 3c: forward phase over all micro-batches, then backward phase.")
