# Hazard 4 — Hints

Hypothesis in `POSTMORTEM.md` first, including a guess at what the message size has to
do with it.

---

<details>
<summary><b>Hint 1 — where to look</b></summary>

`exchange_activations` is four lines. Every rank runs the same four lines.

Trace it for two ranks, by hand, writing down what each rank is doing at each moment:

```
rank 0: send(to=1)     rank 1: send(to=0)
rank 0: recv(from=1)   rank 1: recv(from=0)
```

Rank 0 calls `send` first. Rank 1 calls `send` first. Neither has called `recv` yet.

Now: does `send` return before the peer receives?
</details>

---

<details>
<summary><b>Hint 2 — why the size matters, and why it might not here</b></summary>

Sometimes it matters. Sometimes it does not, and which case you are in depends on the
transport.

A transport may keep an internal buffer. When a message fits, `send` copies it in and
returns immediately, and the peer collects it whenever it gets round to receiving. This is
the *eager* protocol. When the message is too big to buffer, `send` cannot return until
the receiver is ready to take the data directly, because there is nowhere to put it. This
is the *rendezvous* protocol.

The gloo configuration used while preparing this lab blocked at 1 KB. That observation is
not a contract for every gloo build. Some transports and MPI configurations use an eager
protocol, so a small case may pass and a larger one may expose the bug. Thresholds and
completion semantics vary, and NCCL should not be described as though it were an MPI eager
buffer with a universal cutoff.

Either way the underlying mistake is the same, and you can see it by tracing two ranks by
hand:

```
rank 0: send(to=1)     rank 1: send(to=0)
rank 0: recv(from=1)   rank 1: recv(from=0)
```

Both call `send`. Neither has called `recv`. Nobody is receiving.

**Why the eager case is the dangerous one.** Your unit tests use small tensors. Production
does not. A bug that only appears above a buffer threshold passes CI and fails on the
first real run.
</details>

---

<details>
<summary><b>Hint 3 — the answer, and three ways to fix it</b></summary>

Symmetric blocking send-then-receive deadlocks whenever progress requires a peer to post
its receive. Because the API does not promise a portable buffering threshold, correct code
must not rely on a small send completing first.

**Fix A: odd/even ordering.** Break the symmetry so someone is always receiving.

```python
if rank % 2 == 0:
    collectives.send(tensor, dst=send_to)
    collectives.recv(inbox, src=recv_from)
else:
    collectives.recv(inbox, src=recv_from)
    collectives.send(tensor, dst=send_to)
```

Free, and correct for a chain or a ring with an even world size. At an odd world size the
ring has two adjacent even ranks and you are back where you started. Worth knowing about,
worth not relying on.

**Fix B: non-blocking, then wait.**

```python
send_req = collectives.isend(tensor, dst=send_to)
recv_req = collectives.irecv(inbox, src=recv_from)
send_req.wait()
recv_req.wait()
```

Both operations are posted before either blocks, so the transport can pair them up. Works
at any world size. The ordering of the two `wait()` calls does not matter here, and in
more complicated patterns it can, which is the reason for fix C.

**Fix C: `batch_isend_irecv`. This is the recommended one.**

```python
import torch.distributed as dist

ops = [
    collectives.P2POp(dist.isend, tensor, send_to),
    collectives.P2POp(dist.irecv, inbox, recv_from),
]
for req in collectives.batch_isend_irecv(ops):
    req.wait()
```

The backend sees every transfer at once and orders them itself. The deadlock becomes
impossible by construction rather than by your reasoning being correct, which matters
because the reasoning gets harder as the communication pattern does.

Any of the three passes the test. C is what Lab 3 requires and what production pipeline
implementations use.
</details>

---

## The general lesson

Two of them, and the second is the more portable.

**Symmetric blocking communication deadlocks.** Any pattern where every rank does the same
blocking operation in the same order, with no one receiving, will hang. Break the symmetry
or use non-blocking primitives.

**A bug that depends on message size will not show up in your tests.** This is the general
form and it is worth internalising beyond this specific hazard. The eager/rendezvous
switch is one instance. Others in this same family:

- a collective that fits in one packet against one that does not
- a tensor that fits in the L2 cache against one that does not
- an allocation that the caching allocator already has a block for against one it does not
- a batch that fits in VRAM against one that does not

If your test suite only exercises small inputs, it is not testing the regime you will run
in. Lab 0's `test_symmetric_send_deadlocks` uses a 32 MB payload specifically so that it
demonstrates the failure rather than accidentally passing, and its docstring says what to
do if your transport buffers even that.
