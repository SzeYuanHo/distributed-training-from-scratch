# Hazard 1 — Postmortem

## 1. Symptom

Four ranks start, print the world size, and stop. No traceback, no output, and the
CPU goes idle rather than spinning. I waited two minutes before killing it. It dies
on the 30 second process-group timeout if you leave it.

## 2. First hypothesis

Written before looking: a straggler. I assumed one rank was just slow to reach the
first all-reduce and the others were waiting on it, so the fix would be a longer
timeout. Wrong — idle CPU should have told me nothing was being computed anywhere.

## 3. How I confirmed it

`TORCH_DISTRIBUTED_DEBUG=DETAIL` first. The last collective logged differed by rank:
rank 0 had issued one more all-reduce than everyone else, and the step number where
they diverged was 0, 3, 6 — every third step.

The culprit is the rank that is *not* blocked in a collective. Ranks 1-3 were all
parked in the same all-reduce at the top of the next `train_step`, which makes the
all-reduce look guilty. Rank 0 had moved past it into the logging branch and was
sitting in a collective of its own that nobody else would ever join. Three stacks
agreeing and one disagreeing is the signal; the odd one out is where the divergence
happened.

## 4. Root cause

The gradient-norm all-reduce sits inside `if step % 3 == 0 and rank == 0`. Rank 0
issues it, the other three never do, and both sides block forever.

## 5. The fix

Hoist the collective out of the rank test. Every rank computes and reduces the norm;
only rank 0 prints the result.

Why that one: the alternative is to compute the norm locally on rank 0 and not
communicate at all, which is cheaper but reports a norm over one rank's gradients
instead of the job's. The number would be quietly wrong, and a diagnostic you cannot
trust is worse than no diagnostic.

## 6. The general lesson

Logging is control flow, and on a distributed job control flow is a contract between
ranks. Any branch that is not taken identically everywhere must not contain a
collective. `if rank == 0:` is the shape to look for, and it is nearly always
wrapping something diagnostic, which is why it survives review — the reviewer reads
it as "printing", not as "communicating".

## 7. How would you catch this earlier?

At 512 ranks I would not dump every stack by hand. I would enable the Flight Recorder
ring buffer (`TORCH_NCCL_TRACE_BUFFER_SIZE=1000`) and its timeout dump
(`TORCH_NCCL_DUMP_ON_TIMEOUT=1`) so the watchdog writes the last collectives per rank
and the divergence becomes a diff rather than an investigation. A
short process-group timeout in development, so this fails in 30 seconds instead of
looking like slowness. And for a diagnostic path specifically, `monitored_barrier`
names the ranks that failed to arrive instead of hanging.

## 8. Time spent

Hypothesis to fix: about 35 minutes, most of it spent on the straggler theory before
I noticed the CPU was idle.
