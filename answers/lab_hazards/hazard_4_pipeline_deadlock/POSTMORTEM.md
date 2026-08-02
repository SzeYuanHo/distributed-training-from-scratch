# Hazard 4 — Postmortem

## 1. Symptom

It stops at the first size. On gloo the script never gets past the 1 KB exchange —
`batch=1` hangs and the 8 and 64 cases never run. No error until the timeout fires.

The size it stops at is the interesting part. On a transport with an eager protocol
the small sizes would sail through and only the large one would block, which is the
version of this bug that survives code review.

## 2. First hypothesis

Written before investigating: a ring-index mistake. If `send_to` and `recv_from` were
computed with the same sign, some rank would be waiting on a peer that was not
talking to it, and the wrap-around at rank 0 seemed the likely place for an
off-by-one.

Wrong. The indices are fine; every send does have a matching receive. They are just
posted in an order that cannot make progress.

## 3. How I confirmed it

Two ranks and a stack dump was enough: both were inside `send`, neither had reached
`recv`. That rules out an index error immediately — a mismatched peer would leave one
rank in `send` and the other in `recv`, not both in the same call.

Then the size sweep, which is the more useful evidence. Raising the payload to 4 MB
kept it hanging in this environment, so a version that happens to pass at 1 KB is
not necessarily fixed; the transport may simply have hidden the ordering defect.

## 4. Root cause

Every rank calls blocking `send` before any rank calls `recv`. In the observed run,
each send needed its peer to post the matching receive, so all four ranks waited on
an operation the next rank would only reach after its own send returned. A cycle,
and nothing broke it.

Whether a small payload appears to work is a property of the backend, build and
transport, not a correctness property of this code. This gloo run blocked
immediately. Some eager transports may hide the problem for small messages. NCCL's
completion and progress semantics differ from MPI eager buffering, so this result
does not establish a universal NCCL threshold.

## 5. The fix

Build both halves as `P2POp`s and hand them to `batch_isend_irecv` together, then wait
on the returned work objects.

Why that one: odd/even ordering also works and costs nothing, but it breaks at odd
world sizes. In a ring of three, ranks 0 and 2 are neighbours and both even, so both
send first and neither receives — the same deadlock with an extra rule to remember.
`test_it_works_at_an_odd_world_size` exists for exactly that. Handing the whole
exchange to the backend lets it order the transfers, so the property holds by
construction rather than by convention.

Replacing it with an all-gather would also pass, and would move N times the bytes and
throw away the topology that makes pipeline parallelism worth anything.

## 6. The general lesson

`send` returning is not a property of the API, it is a property of the buffer. Any
reasoning of the form "the send will return and then we receive" is an assumption
about a threshold nobody wrote down.

The class of bug is a correctness problem that presents as a scale problem: it passes
every test written against small tensors, then hangs the first time the tensors are
real. Test the exchange at a size past any plausible eager threshold, and at an odd
world size.

## 7. How would you catch this earlier?

Short process-group timeouts everywhere in development, so a deadlock is a 30 second
failure rather than a job that looks slow. At scale, the Flight Recorder dump on
timeout shows the last collective per rank, and every rank sitting in a send is an
unmistakable signature.

The cheap structural guard: no blocking `send` in a symmetric exchange, ever. If both
sides of a link send, it goes through `batch_isend_irecv`.

## 8. Time spent

Hypothesis to fix: about 15 minutes. Seeing both ranks in `send` made it obvious;
most of the time went on the odd-world-size case afterwards.
