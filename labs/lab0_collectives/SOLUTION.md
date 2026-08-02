# Lab 0 — Solution notes

Read this when you are stuck, not before. It explains the approach to each step and
shows the two or three lines that are genuinely hard to guess. It is deliberately not
copy-pasteable: there is no complete function here, and assembling one from these
fragments takes about as long as writing it yourself, which is the point.

---

## Step 1: `chunk_boundaries`

Two remainder schemes work and one of them is much less error-prone.

**Scheme A, pad to a multiple.** Round the length up to `N * ceil(length/N)`, work with
equal chunks, and slice the padding off at the end. The ring loop stays simple. You pay
for it in an allocation, and in remembering to slice, and in the fact that the padded
elements participate in the sum and must therefore be zero.

**Scheme B, uneven chunks.** The first `length % N` chunks get one extra element. No
padding, no allocation, and every chunk index computation now needs the offset. This is
what NCCL does.

Scheme B is worth the extra thought, and the arithmetic is a two-liner once you see it:

```python
base, remainder = divmod(length, n_ranks)
# chunk r has `base + (1 if r < remainder else 0)` elements
# and starts at `r * base + min(r, remainder)`
```

That `min(r, remainder)` is the part people get wrong. It counts how many of the earlier
chunks got the extra element, and it is not `r * (remainder > 0)`.

Check it against `length=7, n_ranks=3`: base=2, remainder=1. Chunk 0 starts at 0 and has
3 elements, chunk 1 starts at 3 with 2, chunk 2 starts at 5 with 2. Total 7. Good.

---

## Step 1b: `reduce_scatter_numpy`

The index arithmetic is the whole exercise, and it is worth deriving rather than
looking up.

In step `s` of `N-1`, rank `r` sends one chunk and receives one chunk. The question is
which. Work backwards from the postcondition: after all `N-1` steps, rank `r` must hold
the fully-reduced chunk `r`. That chunk had to visit all `N` ranks, arriving at `r` last.
So it started at `r+1`, then `r+2`, and so on around the ring.

Which means in step `s`, rank `r` should be *receiving* the chunk that will eventually
land on `r` only in the final step, and *sending* something a step further along.

The pair that satisfies this:

```python
send_idx = (rank - step) % n_ranks
recv_idx = (rank - step - 1) % n_ranks
```

Do not take that on faith. Trace `N=3`, and write out the table:

| step | rank 0 sends / receives | rank 1 | rank 2 |
|---|---|---|---|
| 0 | send 0, recv 2 | send 1, recv 0 | send 2, recv 1 |
| 1 | send 2, recv 1 | send 0, recv 2 | send 1, recv 0 |

After step 1, rank 0 has accumulated chunk 1 from both other ranks. Rank 1 has chunk 2,
rank 2 has chunk 0. Which is a rotation of what the postcondition asked for.

**That rotation is the trap.** If your test fails with the right values in the wrong
slots, you have an off-by-one in the direction, not a broken algorithm. Flip the sign of
`step` or shift by one and re-derive rather than guessing.

Accumulate into what you received, then send that in the next round. A common mistake is
sending the original chunk every time, which is correct at `N=2` and wrong for every
larger `N`.

---

## Step 3b: `broadcast_tree`

Two independent difficulties, and separating them helps.

**Relabelling.** Doing the arithmetic with an arbitrary `src` is unpleasant. Relabel so
`src` becomes 0:

```python
virtual = (rank - src) % world
# ... do everything in virtual space ...
real_peer = (virtual_peer + src) % world
```

Now round `k` is: ranks with `virtual < 2**k` send to `virtual + 2**k`.

**Ranks that do not exist.** At `N=6`, in round 2, virtual rank 1 would send to virtual
rank 5 (fine) and virtual rank 2 would send to virtual rank 6 (which does not exist).
The sender must skip; the non-existent receiver obviously cannot block. Get this wrong in
the other direction, where a receiver waits for a message nobody sends, and you have a
hang.

Structure it as: each rank computes, for each round, whether it sends, receives, or does
nothing. Then the loop body is three branches and no special cases.

---

## Steps 4 and 5: the ring, distributed

If step 1 is solid this is translation. Three things are new:

**Buffers must exist before the receive.** `recv` fills a tensor you supply; there is no
size negotiation. Allocate the receive buffers before the loop, not inside it, or you
will allocate `2(N-1)` tensors per all-reduce and wonder why the profile is all
`aten::empty`.

**The symmetric-send deadlock is real here.** Every rank running `send(right)` then
`recv(left)` blocks once the chunk exceeds the transport buffer. Three fixes:

1. Odd/even ordering: even ranks send then receive, odd ranks receive then send. Simple,
   and it breaks at odd world sizes unless you are careful.
2. `isend` and `irecv` with an explicit `wait()`. Post both, then wait. This is the one
   to use here.
3. `batch_isend_irecv`, which hands the whole exchange to the backend. Best for Lab 3's
   pipeline; slightly heavy for a two-message exchange.

Whichever you pick, leave a comment saying which and why. You will be back here in Lab 3.

**In-place all-reduce means copying back.** `all_reduce_ring` is documented as in place.
`reduce_scatter_ring` returns a new chunk, and `all_gather_ring` returns a new full
buffer, so the last line is `tensor.copy_(gathered)`. Forgetting it produces a function
that computes the right answer and returns nothing useful, and the failure looks like the
algorithm is broken when it is not.

---

## Step 6: recursive doubling

The bit trick is `peer = rank ^ (1 << k)`. XOR with a single set bit flips exactly that
bit, which is precisely "the rank whose id differs from mine in bit k". After `log2(N)`
rounds you have paired with every rank exactly once, along the edges of a hypercube.

Two variants, and the docstring asks you to choose:

**Whole-buffer variant.** In each round, exchange the entire buffer with your peer and
sum. `log2(N)` rounds, each moving `M` bytes, so `log2(N)·M` bytes per rank. Six lines.

**Halving-doubling variant.** In each round of the first phase, exchange only the half of
your buffer that your peer will end up owning. Data halves each round.
`2·log2(N)` rounds moving `2(N-1)/N·M` bytes total. Considerably more index bookkeeping.

At `N=8`: whole-buffer moves `3M`, halving-doubling moves `1.75M`. At `N=64`:
whole-buffer moves `6M`, halving-doubling still moves under `2M`. The whole-buffer
variant is fine for small `N` and a disaster for large `N`, which is the answer the
writeup question is fishing for.

Implement whichever you like. The test accepts either and tells you which one it thinks
you wrote, from the send count.

---

## Step 8: fitting alpha and beta

The trap is stated in the docstring and it is worth spelling out. Your measurements span
roughly `1e-5` to `1e-1` seconds. An unweighted least-squares fit minimises absolute
squared error, so a 50% error on a 10-microsecond point contributes essentially nothing
next to a 1% error on a 100-millisecond point. Alpha comes out as whatever noise is left,
often negative.

Three defensible fixes:

**Weight by `1/t²`.** Turns absolute error into relative error. One extra argument to
`np.polyfit` via `w=1/times`, and the change in the fitted alpha is dramatic.

**Fit alpha separately.** Take the smallest three or four points, where `β·m` is
negligible, and let alpha be their median. Then fit beta on the large points with alpha
fixed. Physically motivated and easy to explain.

**Fit in log space.** Fits a power law rather than a line, which is a different model.
Do not do this unless you mean to.

Whichever you pick, note it in a comment. When your predicted crossover misses the
measurement by 3x, this is the first place to look.

The negative-alpha clamp is not defensive programming for its own sake. A curve with a
protocol switch in it genuinely fits a negative intercept, and returning that produces a
predicted time below zero for small messages, which then produces a nonsense crossover,
which then sits in your writeup unnoticed.

---

## What the writeup should not say

"The measurements roughly matched the model." They will not, and saying so means you did
not look.

Concretely, on gloo over localhost you should expect:

- **Alpha in the tens of microseconds.** It is dominated by python, not by the network:
  a `dist.send` call goes through the c10d dispatcher, a work object, and a future.
- **Beta corresponding to a few GB/s.** Loopback is a memcpy, so the ceiling is DRAM
  bandwidth, and you will be well under it.
- **A visible kink** somewhere in the low hundreds of kilobytes, where gloo switches
  protocol. The single-line model cannot represent that, which is one of the three things
  section 1 of THEORY.md asked you to write down.
- **Your ring several times slower than `dist.all_reduce`** at every size. Account for
  where that goes rather than treating it as a defect.

The last one is worth sitting with. Your implementation is algorithmically identical to
the real one and measurably worse, and every part of the difference is engineering:
segment pipelining, protocol switching, avoiding a python frame per message, reusing
buffers. That gap between "correct algorithm" and "production implementation" is most of
what the rest of this repo is about.
