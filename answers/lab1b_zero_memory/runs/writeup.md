# Run notes

> **Maintenance status.** The two impossible or mismatched checks described below were
> repaired: the balance bound now respects indivisible parameters, and communication is
> compared as algorithmic wire volume with matching optimizer settings. The completed
> ZeRO-1 answer passes the current contract. The original reasoning is retained because
> it explains why those tests needed correction.

The ledger was the one part of this curriculum I predicted well, which is not
surprising — it is arithmetic I did on paper first, and there is no hardware in it to
be wrong about.

| | predicted | measured |
|---|---|---|
| states, plain AdamW | 172,320,000 B | 171,589,632 B |
| states, ZeRO-1 at N=4 | 75,390,000 B | 75,070,464 B |

Both within half a percent, and the gap is just my rounding of the parameter count.
16 bytes per parameter held exactly, and so did the fp32 case, which is the result
worth keeping: mixed precision does not halve state memory. It halves activations and
wire bytes and buys tensor cores. I had written 8 bytes for the fp32 row the first
time through THEORY and had to go back when the test told me otherwise.

## Two reduction factors

The total-state prediction is 2.286x. The benchmark's optimizer-state-only result is
3.967x. They answer different questions and are now recorded under different keys.

I predicted the reduction in *total* state: parameters and gradients stay replicated
under ZeRO-1 and only the 12-byte optimiser group shards, so 16/(2+2+12/4) = 2.29.
The benchmark divides `optimizer_state_bytes` at one rank by `local_state_bytes` at
four, which is optimiser state only, and that shards cleanly by N. It should approach
4 and it does.

So both numbers are right about their own question. The total-state factor is the one
that matters when deciding whether a model fits; 3.967 is the one that shows the
optimizer-state sharding works. Reconciliation leaves the total-state prediction
unmeasured instead of presenting the optimizer-only result as a miss.

## Measured peaks are RSS

The plotted points are process RSS, so they include the interpreter, torch, the
dataset and the allocator. On a model whose states are a few megabytes that is tens
of megabytes of constant offset. The trend against N is real; the absolute values are
not comparable with the analytic totals and I did not try. Worth rerunning on the GPU
where `max_memory_allocated` means something.

## Historical test-contract defects

**The original `test_shard_parameters_balances_bytes`.** It built one 10,000-element
tensor plus twenty 100-element ones and asked for an imbalance under 12x across 4
ranks. Greedy largest-first gives `[10000, 700, 700, 600]`, so 16.7x.

No whole-parameter assignment can pass. For imbalance below 12 the smallest shard needs
more than 10000/12 = 833 elements. The big tensor occupies one rank on its own, and
the other three share 2000 elements, so the best possible smallest shard is 666.
Whole-parameter sharding cannot do it, and `shard_parameters`' signature commits to
whole parameters — the docstring says so explicitly, and notes that real ZeRO shards
byte ranges instead precisely to get perfect balance. Splitting tensors across ranks
would have changed the exercise. The maintained bound now accounts for the largest
indivisible parameter.

**The original `test_zero1_communication_equals_ddp[2]`.** It reported 1.50x while
requiring under 1.4x. It passed at N=4 (1.25x), which was the clue.

The wire volume genuinely is equal — that is the whole point, and I derived it in
THEORY section 3. What differs is the counting. `common/collectives` records the
*input buffer* size: `reduce_scatter_tensor` gets the full P bytes, `all_gather_into_tensor`
gets P/N. So ZeRO-1 counts P + P/N against DDP's P, which is 1.5 at N=2 and 1.25 at
N=4, converging to 1 as N grows. It is an artefact of measuring buffers rather than
bytes on the wire.

All-reducing and slicing would have satisfied that counter while no longer being
ZeRO-1. The maintained test compares algorithmic wire volume with matching optimizer
settings. The original collective counts remain useful evidence of the structure:
`{'reduce_scatter': 10, 'all_gather': 10}` against DDP's `{'all_reduce': 520}`, which
is 10 steps of one flat collective each instead of one per parameter.
