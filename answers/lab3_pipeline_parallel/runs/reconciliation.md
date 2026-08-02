```
Predictions vs measurements: lab3_pipeline_parallel
================================================================================================================
quantity                                              predicted        measured          units     error        
----------------------------------------------------------------------------------------------------------------
theoretical_bubble_p2_m4                                    0.2             0.2       fraction      0.0%  ok
theoretical_bubble_p4_m8                                 0.2727          0.2727       fraction      0.0%  ok
theoretical_bubble_p4_m16                                0.1579          0.1579       fraction      0.0%  ok
measured_bubble_p4_m8                                      0.52          0.3695       fraction     28.9%  close
afab_max_inflight_microbatches_p4_m8                          8    not recorded  micro-batches            unmeasured
onefonebe_max_inflight_microbatches_p4_m8                     4    not recorded  micro-batches            unmeasured
onefonebe_warmup_steps_stage0_p4                              4    not recorded       forwards            unmeasured
onefonebe_warmup_steps_stage3_p4                              1    not recorded       forwards            unmeasured
slowest_stage_index_p3_small                                  2    not recorded    stage index            unmeasured
stage_imbalance_factor_p3_small                           1.007    not recorded              x            unmeasured
stage_imbalance_factor_p3_vocab128k                       14.16    not recorded              x            unmeasured
activation_memory_ratio_afab_over_1f1b_p4_m16                 4    not recorded              x            unmeasured
interleaved_v_star_infiniband                                 2    not recorded         chunks            unmeasured
measured_bubble_1f1b_p4_m8                        not predicted          0.3695       fraction            unpredicted
measured_bubble_afab_p4_m8                        not predicted          0.2465       fraction            unpredicted
measured_bubble_p2_m1                             not predicted          0.5152       fraction            unpredicted
measured_bubble_p2_m16                            not predicted          0.1697       fraction            unpredicted
measured_bubble_p2_m2                             not predicted           0.342       fraction            unpredicted
measured_bubble_p2_m4                             not predicted          0.2821       fraction            unpredicted
measured_bubble_p2_m8                             not predicted          0.1982       fraction            unpredicted
measured_bubble_p4_m1                             not predicted          0.7662       fraction            unpredicted
measured_bubble_p4_m16                            not predicted          0.2648       fraction            unpredicted
measured_bubble_p4_m2                             not predicted          0.6028       fraction            unpredicted
measured_bubble_p4_m4                             not predicted          0.4835       fraction            unpredicted
peak_memory_1f1b                                  not predicted       2.661e+08          bytes            unpredicted
peak_memory_afab                                  not predicted       2.674e+08          bytes            unpredicted
theoretical_bubble_p2_m1                          not predicted             0.5       fraction            unpredicted
theoretical_bubble_p2_m16                         not predicted         0.05882       fraction            unpredicted
theoretical_bubble_p2_m2                          not predicted          0.3333       fraction            unpredicted
theoretical_bubble_p2_m8                          not predicted          0.1111       fraction            unpredicted
theoretical_bubble_p4_m1                          not predicted            0.75       fraction            unpredicted
theoretical_bubble_p4_m2                          not predicted             0.6       fraction            unpredicted
theoretical_bubble_p4_m4                          not predicted          0.4286       fraction            unpredicted
----------------------------------------------------------------------------------------------------------------

Measured but never predicted: measured_bubble_1f1b_p4_m8, measured_bubble_afab_p4_m8, measured_bubble_p2_m1, measured_bubble_p2_m16, measured_bubble_p2_m2, measured_bubble_p2_m4, measured_bubble_p2_m8, measured_bubble_p4_m1, measured_bubble_p4_m16, measured_bubble_p4_m2, measured_bubble_p4_m4, peak_memory_1f1b, peak_memory_afab, theoretical_bubble_p2_m1, theoretical_bubble_p2_m16, theoretical_bubble_p2_m2, theoretical_bubble_p2_m8, theoretical_bubble_p4_m1, theoretical_bubble_p4_m2, theoretical_bubble_p4_m4

Predictions without a recorded counterpart: afab_max_inflight_microbatches_p4_m8, onefonebe_max_inflight_microbatches_p4_m8, onefonebe_warmup_steps_stage0_p4, onefonebe_warmup_steps_stage3_p4, slowest_stage_index_p3_small, stage_imbalance_factor_p3_small, stage_imbalance_factor_p3_vocab128k, activation_memory_ratio_afab_over_1f1b_p4_m16, interleaved_v_star_infiniband
  Some are analytic or inspection results. Check the relevant derivation or test; rerun a benchmark only for quantities it records.
```
