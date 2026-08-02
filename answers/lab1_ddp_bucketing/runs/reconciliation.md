```
Predictions vs measurements: lab1_ddp_bucketing
====================================================================================================
quantity                                    predicted        measured        units     error        
----------------------------------------------------------------------------------------------------
total_gradient_bytes_tiny                   3.276e+06    not recorded        bytes            unmeasured
simple_ddp_allreduces_per_step                     52    not recorded  collectives            unmeasured
bucketed_allreduces_per_step_1mb                    4               4  collectives      0.0%  ok
no_sync_allreduces_per_4_microsteps                 4    not recorded  collectives            unmeasured
optimal_bucket_bytes                        1.049e+06       1.049e+06        bytes      0.0%  ok
optimal_bucket_count                                4               4      buckets      0.0%  ok
single_process_step_ms                             45           151.9           ms    237.6%  MISS
ddp_2rank_step_ms                                  95             256           ms    169.4%  MISS
tier1_speedup_vs_single_process                  0.47          0.5936            x     26.3%  close
exposed_comm_fraction                            0.65    not recorded     fraction            unmeasured
tier2_scaling_efficiency_2xT4                    0.75    not recorded     fraction            unmeasured
----------------------------------------------------------------------------------------------------

Off by more than 50%. These are the ones worth understanding:
  single_process_step_ms   (you wrote: Tiny preset, batch 8, block 128 on this CPU.)
  ddp_2rank_step_ms   (you wrote: Pre-benchmark estimate for the same work per rank plus gloo all-reduces.)
  Write down why in runs/<lab>/writeup.md before moving on.

Predictions without a recorded counterpart: total_gradient_bytes_tiny, simple_ddp_allreduces_per_step, no_sync_allreduces_per_4_microsteps, exposed_comm_fraction, tier2_scaling_efficiency_2xT4
  Some are analytic or inspection results. Check the relevant derivation or test; rerun a benchmark only for quantities it records.
```
