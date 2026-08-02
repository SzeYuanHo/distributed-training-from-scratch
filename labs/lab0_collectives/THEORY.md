# Lab 0 — Theory

Work through this with a pen before you write code. The blanks marked `____` are for
you. Nothing here is checked by a test, and that is deliberate: the tests check whether
your code is right, this checks whether you are.

Notation follows Grama §2.5, translated into the α–β form everyone else uses:

- `α` = latency per message, in seconds. Grama writes `t_s`.
- `β` = inverse bandwidth, seconds per byte. Grama writes `t_w` per word.
- `N` = number of ranks.
- `M` = size of the buffer being reduced, in bytes.

Cost of one point-to-point message of `m` bytes: `T = α + β·m`.

---

## 1. Warm-up: what does the model ignore?

The α–β model is a straight line. Real networks are not. Before you fit it, write down
three things it cannot capture:

1. `____________________________________________`
2. `____________________________________________`
3. `____________________________________________`

(Hints if you are stuck, in decreasing order of obviousness: congestion when many ranks
send at once; the difference between a message that fits in a single packet and one that
does not; the fact that a rank cannot send and receive on the same link simultaneously
on some hardware; protocol switching between eager and rendezvous at a size threshold.)

Keep these. When your measurements miss your predictions in section 6, this list is
where you look first.

---

## 2. Naive all-reduce: gather to root, then broadcast

Every rank sends its buffer to rank 0. Rank 0 sums them. Rank 0 sends the result back to
everyone.

**Messages received by rank 0:** `____`

**Bytes received by rank 0:** `____`

**Total time, assuming rank 0 processes messages serially:**

```
T_naive = ____________________
```

**Why this is the wrong algorithm**, in one sentence:

`____________________________________________________________`

Now, a subtlety. If the network lets rank 0 receive from all N-1 peers simultaneously at
full bandwidth, is this algorithm still bad? Explain what actually limits it:

`____________________________________________________________`

---

## 3. Recursive halving-doubling

Round `k` (from 0): each rank exchanges with the rank whose id differs in bit `k`.

The reduce-scatter phase halves the data each round while doubling the distance. The
all-gather phase reverses it.

**Number of rounds in each phase:** `____`

**In round k of reduce-scatter, how many bytes does each rank send?**

```
bytes(k) = ____________________
```

**Sum over all rounds:**

```
Σ bytes = ____________________
```

**Total time:**

```
T_rd = ____·α + ____·β·M
```

**Why N must be a power of two:** `_______________________________________`

What is the smallest change that would make it work for arbitrary N, and what does that
change cost? (Real libraries do this. Look up "binary blocks algorithm" if you want the
answer, but try first.)

`____________________________________________________________`

---

## 4. Ring all-reduce

Rank `r` sends only to rank `(r+1) mod N` and receives only from `(r-1) mod N`. The
buffer is split into N chunks.

**Phase 1, reduce-scatter.** After N-1 steps, rank r holds the fully-reduced chunk
`____`. In step `s`, rank `r` sends chunk index `____` and receives chunk index `____`.

**Phase 2, all-gather.** Another N-1 steps circulating the reduced chunks.

**Messages sent per rank, both phases:** `____`

**Bytes sent per rank, both phases:**

```
____________________
```

**Total time:**

```
T_ring = ____·α + ____·β·M
```

### 4.1 The result that matters

Show that the per-rank byte volume `2·(N-1)/N·M`:

- is monotonically increasing in N: `____________________`
- is bounded above by `2M`: `____________________`
- equals `____` when N=2 and `____` when N=1024

Now state, in your own words, why this makes data parallelism viable at 512 GPUs when a
gather-to-root scheme would not:

`____________________________________________________________`
`____________________________________________________________`

### 4.2 The remainder

If `M` bytes do not divide evenly by `N`, you have a choice. Write down two schemes and
what each costs:

Scheme A: `____________________________________________`
Cost: `____________________`

Scheme B: `____________________________________________`
Cost: `____________________`

Which one will you implement, and why?

`____________________________________________________________`

(This is not a rhetorical exercise. `test_00_math.py` tests length 7 on 3 ranks and
length 100 on 8 ranks. One of these schemes is noticeably easier to get right.)

---

## 5. The crossover

Set the two total costs equal:

```
T_ring(M*) = T_rd(M*)

2(N-1)·α + 2(N-1)/N·β·M*  =  2·log₂N·α + 2(N-1)/N·β·M*
```

Something interesting happens. Work it through:

`____________________________________________________________`

Look at the β terms. What do you notice?

`____________________________________________________________`

So the comparison as written above is degenerate, and that tells you the textbook
recursive-doubling and ring algorithms move the *same* number of bytes. The real
crossover comes from the fact that the two algorithms do not achieve the same effective
bandwidth. Write the corrected model, with `β_ring` and `β_rd` as separate constants:

```
T_ring(M) = ____________________
T_rd(M)   = ____________________
```

Solve for `M*`:

```
M* = ____________________
```

**Your numeric prediction** for gloo on localhost with N=4. Guess α and β first from
what you know about loopback (a localhost round trip is tens of microseconds; loopback
bandwidth is a few GB/s), then compute:

```
α  ≈ ____ s
β  ≈ ____ s/byte
M* ≈ ____ bytes
```

Put `M*` into `predictions.yaml` as `crossover_bytes_n4`. You will find out how wrong
you are in about an hour, and that is the point.

---

## 6. Algorithm bandwidth against bus bandwidth

For an all-reduce of `M` bytes taking `T` seconds across N ranks:

```
algorithm bandwidth  =  M / T
bus bandwidth        =  ____________________
```

**Which one does NCCL report?** `____`

**Why does the distinction exist at all?** Bus bandwidth is constructed so that a
perfect ring implementation reports a number close to the hardware's link bandwidth,
independent of N. Verify that: substitute `T_ring` with α=0 and confirm the bus
bandwidth comes out as `1/β`.

`____________________________________________________________`

**A colleague says "we hit 180 GB/s all-reduce on 8 GPUs."** List the three questions you
need answered before that number means anything:

1. `____________________`
2. `____________________`
3. `____________________`

---

## 7. Broadcast, the same exercise in miniature

**Linear broadcast:** rank 0 sends to each of the other N-1 ranks in a loop.

Time: `____________________`

**Tree broadcast:** rank 0 sends to rank N/2, then both halves recurse.

Rounds: `____`   Time: `____________________`

**Scatter-then-all-gather broadcast:** rank 0 scatters M/N to each rank, then all ranks
all-gather.

Time: `____________________`

**Which is fastest for small M?** `____`   **For large M?** `____`

Now predict the message counts the `CommCounter` will report for each, at N=8, and check
your answer against the test output:

```
linear:  ____ sends from rank 0
tree:    ____ sends from rank 0, ____ total across all ranks
```

---

## 8. Before you run the sweep

Fill in `predictions.yaml`. Every key, including the ones you are unsure about. A blank
entry scores nothing; a wrong entry teaches you something.

The reconciliation table will show you where your model of the machine is wrong. That is
the deliverable of this lab, more than the code is.
