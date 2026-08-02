# Lab 3 notes

## Bubble

For four stages the last stage waits for three forward slots at the start, and
stage 0 waits for three backward slots at the end.

With forward and backward both counted as one slot:

```
T_pipeline = 2m + 2(p - 1)
T_ideal    = 2m
```

The overhead relative to ideal work is

```
overhead ratio = (T_pipeline - T_ideal) / T_ideal = (p - 1) / m
```

The idle fraction of elapsed pipeline time—the quantity comparable to
`1 - EventLog` compute occupancy—is

```
idle fraction = (T_pipeline - T_ideal) / T_pipeline
              = (p - 1) / (m + p - 1)
```

| p | m | overhead / ideal | idle / elapsed |
|---:|---:|---:|---:|
| 4 | 1 | 3 | 0.75 |
| 4 | 4 | 0.75 | 3/7 ≈ 0.429 |
| 4 | 16 | 0.1875 | 3/19 ≈ 0.158 |
| 8 | 16 | 0.4375 | 7/23 ≈ 0.304 |

`m=1` is just one batch moving through the stages; there is no steady state.
Very large `m` makes each matmul smaller and AFAB has to retain all `m`
activation sets.

## Activation count

AFAB holds `m` micro-batches at its peak, hence `O(m)` activation memory.

For 1F1B:

```
warmup(i) = min(p - i, m)
```

At `p=4`, that is four forwards on stage 0 and one on stage 3. A stage then
keeps roughly `p-i` batches in flight, with a worst case of `p`, so the bound is
`O(p)`. This is useful because the saved memory can be spent on a larger `m`,
which reduces the bubble.

Two forwards in a row add one live graph. Two backwards in a row consume the
queue and eventually wait for work that has not been forwarded. Alternating is
the steady ordering that holds the live count flat. A backward at stage `i`
cannot start until the same micro-batch has crossed every downstream stage and
its gradient has returned, which accounts for the `p-i` warmup forwards.

## Interleaving

With `p=4, v=2`, device 0 owns virtual chunks 0 and 4. There are eight virtual
stages and the usual interleaved overhead estimate is

```
overhead ratio = (p - 1) / (v m)
idle fraction  = (p - 1) / (v m + p - 1)
```

Point-to-point traffic grows by about `v`. For activation size `A`, my rough
comparison was

```
time saved = T_ideal (p - 1) (1 - 1/v) / m
time added = 2m(p - 1)(v - 1)(alpha + A / bandwidth)
```

A link bandwidth on its own is not enough to produce a unique `v*`; the compute
slot time and `A` are also needed. I used `v=2` as the pre-run guess for the
25 GB/s profile.

## Stage balance

Embedding lookup FLOPs are negligible. Per token, one transformer block is

```
24h^2 + 4sh
```

and the final vocabulary projection adds `2hV`.

For `h=384`, `s=128`, six layers split two per stage:

```
V=72:
stage 0  7,471,104
stage 1  7,471,104
stage 2  7,526,400
```

Stage 2 is about `1.007x` slower. With `V=128000`:

```
stage 0    7,471,104
stage 1    7,471,104
stage 2  105,775,104
```

The final stage is then `14.16x` the lightest stage. Equal layer counts are only
balanced when the work outside the blocks is small.

## Point-to-point ordering

The symmetric version is:

```
send(to=rank+1)
recv(from=rank-1)
```

At two ranks both sends can wait for a receive, so neither rank advances.

| Fix | Cost |
|---|---|
| odd/even order | no extra transfer, but edge cases live in the schedule |
| `isend` + `irecv` | request bookkeeping and explicit waits |
| `batch_isend_irecv` | one batched call and backend scheduling |

I used `batch_isend_irecv`; it also works when the communication pattern is not
a neat even-sized chain.
