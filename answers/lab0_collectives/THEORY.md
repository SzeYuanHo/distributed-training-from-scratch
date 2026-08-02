# Lab 0 — Theory answers

Notation:

- `α` is latency per message in seconds.
- `β` is inverse bandwidth in seconds per byte.
- `N` is the rank count.
- `M` is the reduced buffer size in bytes.

One point-to-point message of `m` bytes costs `T = α + βm`.

## 1. What the model ignores

1. Contention when several ranks share a link or memory bus.
2. Eager/rendezvous and packet-size protocol changes.
3. Duplex limits, software overhead, cache effects and topology.

These effects are where I would look first when the fitted line misses a measured
crossover.

## 2. Naive all-reduce

Rank 0 receives `N - 1` messages and `(N - 1)M` bytes.

With serial receives followed by a serial broadcast,

```text
T_naive = 2(N - 1)(α + βM)
```

It is the wrong algorithm because all traffic is funnelled through one root while
the other links are mostly idle. Even if receives can overlap, the root's injection
bandwidth, memory bandwidth and bisection bandwidth remain the bottleneck.

## 3. Recursive halving-doubling

Each phase has `log₂N` rounds. In reduce-scatter round `k`,

```text
bytes(k) = M / 2^(k + 1)
```

so

```text
sum(bytes(k), k=0..log₂N-1) = M(1 - 1/N)
T_rd = 2log₂N·α + 2(N - 1)/N·βM
```

The bit-partition halves the rank set exactly each round, so the direct form
requires a power-of-two `N`. For arbitrary `N`, binary blocks first pair off the
surplus ranks, run the power-of-two algorithm on the remaining block, then return
the result; that adds preprocessing and postprocessing messages.

## 4. Ring all-reduce

I use the indexing convention in which rank `r` finishes reduce-scatter with
reduced chunk `r`. In step `s`, it sends chunk
`(r - s - 1) mod N` and receives chunk `(r - s - 2) mod N`.

Across reduce-scatter and all-gather, each rank sends `2(N - 1)` messages and

```text
2(N - 1)/N · M
```

bytes. Therefore

```text
T_ring = 2(N - 1)·α + 2(N - 1)/N·βM
```

### 4.1 The bounded byte volume

Let `V(N) = 2M(1 - 1/N)`.

- `V(N + 1) - V(N) = 2M/(N(N + 1)) > 0`, so it increases monotonically.
- `1 - 1/N < 1`, so `V(N) < 2M`.
- `V(2) = M`.
- `V(1024) = 2046M/1024 = 1.998046875M`.

Each added rank gets a smaller chunk, compensating for the extra hop. The per-rank
volume therefore approaches `2M` rather than growing like `NM`, which keeps data
parallel gradient synchronisation viable at hundreds of ranks.

### 4.2 Uneven lengths

Scheme A pads to `N·ceil(length/N)` with zeros. It keeps equal chunks but allocates
and transmits up to `N - 1` extra elements.

Scheme B gives one extra element to each of the first `length mod N` chunks. It
uses no padding or extra traffic, but every offset is
`r·base + min(r, remainder)`.

I implemented scheme B because it preserves the exact buffer and handles empty
chunks without a special padded value.

## 5. The crossover

With one shared `β`, the bandwidth terms cancel. For `N > 2` the ring then has
more latency hops, so there is no finite crossover; at `N = 2` the models tie.
The useful model needs the algorithms' different achieved bandwidths:

```text
T_ring(M) = 2(N - 1)α + 2(N - 1)/N·β_ring·M
T_rd(M)   = 2log₂N·α + 2(N - 1)/N·β_rd·M
```

When `β_rd > β_ring`,

```text
M* = 2((N - 1) - log₂N)α
     / (2(N - 1)/N · (β_rd - β_ring))
```

If `β_rd <= β_ring`, the ring never overtakes recursive doubling in this model.

My pre-measurement localhost guesses for `N = 4` are:

```text
α       ≈ 3.0e-5 s
β_ring  ≈ 5.0e-10 s/byte
β_rd    ≈ 8.0e-10 s/byte
M*      ≈ 1.33e5 bytes
```

## 6. Algorithm and bus bandwidth

```text
algorithm bandwidth = M / T
bus bandwidth       = 2(N - 1)/N · M / T
```

NVIDIA's `nccl-tests` reports both algorithm bandwidth and the derived bus
bandwidth. For the flat-ring correction used here, with `α = 0`,
`T_ring = 2(N - 1)Mβ/N`, so substituting gives `1/β`, independent of `N`.

Before comparing a quoted “180 GB/s” result I would ask:

1. Is it algorithm bandwidth or NCCL bus bandwidth?
2. What message size, dtype and timing boundary produced it?
3. What rank count, topology and collective algorithm were used?

## 7. Broadcast

For a serial linear broadcast,

```text
T_linear = (N - 1)(α + βM)
```

A doubling tree takes `ceil(log₂N)` rounds and approximately

```text
T_tree = ceil(log₂N)(α + βM)
```

Scatter followed by all-gather takes approximately

```text
T_scatter_gather = 2(N - 1)α + 2(N - 1)/N·βM
```

The tree is favoured for small messages; scatter plus all-gather can win for large
messages because it distributes the bandwidth work.

At `N = 8`, linear broadcast makes rank 0 issue `7` sends. The tree makes rank 0
issue `3` sends and makes `7` sends in total.

## 8. Before the sweep

The accompanying prediction file records the numerical guesses before any
benchmark is run. The point of reconciliation is to explain the error, especially
protocol kinks and Python overhead, rather than to retrofit the prediction.
