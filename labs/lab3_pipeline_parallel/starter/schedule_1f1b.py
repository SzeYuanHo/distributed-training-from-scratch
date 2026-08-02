"""Step 5: 1F1B, the production schedule.

Same bubble as AFAB. Bounded activation memory. That trade is why every real
pipeline implementation uses this and not GPipe.

Three phases:

* **warmup** — run forwards until this stage has enough in flight to start
  alternating. In this file's convention, stage `i` completes `p - i` forwards
  before its first backward (capped by the available micro-batches). The number
  of forward-only steps before the first paired iteration is one smaller.
* **steady state** — one forward, one backward, repeating. The activation count
  stays flat.
* **drain** — the remaining backwards, once there are no forwards left.

The warmup count is the whole schedule. Get it wrong by one and the pipeline
deadlocks, because some stage waits for a message another stage already decided
not to send.
"""

from __future__ import annotations

from .schedule_afab import Op

__all__ = ["onefonebe_schedule", "warmup_steps", "run_1f1b"]


def warmup_steps(stage_index: int, n_stages: int, n_micro: int) -> int:
    """Forwards this stage runs before its first backward.

    THEORY sections 2 and 3 derive it. Two checks on whatever you write:

    * The last stage (``stage_index == n_stages - 1``) should get **1**. It can
      start its backward as soon as its own first forward completes; nothing
      downstream has to return anything.
    * Stage 0 should get **n_stages**, because the first micro-batch has to travel
      all the way down the pipeline and its gradient all the way back.

    And it must never exceed ``n_micro``. With ``n_micro < n_stages`` the pipeline
    never reaches a steady state at all, and the schedule degenerates into AFAB.
    Handle that rather than producing a warmup longer than the work available.
    """
    raise NotImplementedError("Step 5a: THEORY section 2.")


def onefonebe_schedule(stage_index: int, n_stages: int, n_micro: int) -> list[Op]:
    """The operation sequence for one stage under 1F1B.

    Returns:
        Ops in execution order. Length ``2 * n_micro``, same as AFAB: the same
        work, reordered.

    Worth checking your output by hand at ``p=4, n_micro=8``:

    ==========  ================================================
    stage 0     F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
    stage 3     F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
    ==========  ================================================

    Stage 3 is pure alternation with a warmup of one. Stage 0 has a long warmup
    and a long drain, and its steady state is short. Both run 8 forwards and 8
    backwards.

    ``test_00_math.py`` checks these sequences exactly, so an off-by-one fails in
    milliseconds instead of hanging a spawned run.
    """
    raise NotImplementedError("Step 5b: warmup, then alternate, then drain.")


def run_1f1b(
    stage, n_micro: int, batches: list, targets: list | None = None, event_log=None
) -> float:
    """Execute the 1F1B schedule on this rank's stage.

    Same signature and same result as ``run_afab``. The loss must match to within
    floating-point reordering, because it is the same arithmetic in a different
    order.

    What changes from AFAB:

    **Activations are a bounded queue, not a growing list.** A forward pushes,
    a backward pops. The queue depth never exceeds the warmup count, which is
    what makes the memory bounded. If your implementation keeps a list that only
    grows, you have written AFAB with extra steps and
    ``test_peak_activation_memory_is_lower`` will notice.

    **Communication interleaves with compute.** In the steady state a stage
    receives an activation from upstream and a gradient from downstream in the
    same iteration. Issue both in one ``exchange`` call rather than sequentially,
    or you have reintroduced the ordering dependency ``batch_isend_irecv`` was
    supposed to remove.

    **Every test here runs under a timeout.** A warmup off by one deadlocks. Make
    sure that failure is loud.
    """
    raise NotImplementedError("Step 5c.")
