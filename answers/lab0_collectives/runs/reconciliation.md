```
Predictions vs measurements: lab0_collectives
==================================================================================================
quantity                                     predicted        measured     units     error        
--------------------------------------------------------------------------------------------------
alpha_seconds                                    3e-05       0.0002218         s    639.4%  MISS
beta_seconds_per_byte                            5e-10       1.731e-09    s/byte    246.2%  MISS
ring_allreduce_bytes_per_rank_n4_1mb         1.573e+06       1.573e+06     bytes      0.0%  ok
ring_allreduce_sends_per_rank_n4                     6    not recorded  messages            unmeasured
naive_allreduce_bytes_at_root_n4_1mb         3.146e+06    not recorded     bytes            unmeasured
broadcast_linear_sends_n8                            7    not recorded  messages            unmeasured
broadcast_tree_sends_total_n8                        7    not recorded  messages            unmeasured
allreduce_ring_1mb_n4_seconds                0.0009664         0.00321         s    232.2%  MISS
allreduce_ring_16mb_n4_seconds                 0.01276         0.09461         s    641.3%  MISS
crossover_bytes_n4                             133,333       1.055e+05     bytes     20.9%  close
ring_vs_torch_slowdown_factor                        5    not recorded         x            unmeasured
--------------------------------------------------------------------------------------------------

Off by more than 50%. These are the ones worth understanding:
  alpha_seconds   (you wrote: Thirty microseconds. Added up a syscall, a loopback round trip and a scheduler wakeup on the far side. Guessing before I measure.
)
  beta_seconds_per_byte   (you wrote: 2 GB/s effective. Loopback is a memcpy plus protocol, so somewhere well under DRAM and well above a real NIC.
)
  allreduce_ring_1mb_n4_seconds   (you wrote: 6 alpha plus 1.5 MiB times beta.)
  allreduce_ring_16mb_n4_seconds   (you wrote: 6 alpha plus 1.5 times 16 MiB times beta.)
  Write down why in runs/<lab>/writeup.md before moving on.

Predictions without a recorded counterpart: ring_allreduce_sends_per_rank_n4, naive_allreduce_bytes_at_root_n4_1mb, broadcast_linear_sends_n8, broadcast_tree_sends_total_n8, ring_vs_torch_slowdown_factor
  Some are analytic or inspection results. Check the relevant derivation or test; rerun a benchmark only for quantities it records.
```
