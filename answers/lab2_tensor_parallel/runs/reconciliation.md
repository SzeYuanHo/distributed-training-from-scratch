```
Predictions vs measurements: lab2_tensor_parallel
=====================================================================================================
quantity                                     predicted        measured        units     error        
-----------------------------------------------------------------------------------------------------
allreduces_per_layer_forward                         2    not recorded  collectives            unmeasured
allreduces_per_layer_per_step                        4               4  collectives      0.0%  ok
allreduces_per_step_small_preset                    29    not recorded  collectives            unmeasured
mlp_block_forward_allreduces                         1    not recorded  collectives            unmeasured
comm_bytes_per_layer_small_b8_s256           1.258e+07    not recorded        bytes            unmeasured
comm_bytes_per_step_small_tp2                 7.55e+07    not recorded        bytes            unmeasured
comm_bytes_per_step_small_tp4                 7.55e+07    not recorded        bytes            unmeasured
tp_scaling_ceiling_t_for_t4_pair                     4    not recorded        ranks            unmeasured
tp_scaling_ceiling_t_for_a100_nvlink                11    not recorded        ranks            unmeasured
full_logits_bytes_tier2_b8_s512               1.18e+06    not recorded        bytes            unmeasured
full_logits_bytes_real_vocab                 2.097e+09    not recorded        bytes            unmeasured
vocab_parallel_ce_collectives                        3    not recorded  collectives            unmeasured
tp2_step_time_vs_single_process                     12    not recorded            x            unmeasured
measured_payload_l4_b4_s64_h256_tp2       not predicted       4.722e+06        bytes            unpredicted
measured_payload_l4_b4_s64_h256_tp4       not predicted       4.722e+06        bytes            unpredicted
-----------------------------------------------------------------------------------------------------

Measured but never predicted: measured_payload_l4_b4_s64_h256_tp2, measured_payload_l4_b4_s64_h256_tp4

Predictions without a recorded counterpart: allreduces_per_layer_forward, allreduces_per_step_small_preset, mlp_block_forward_allreduces, comm_bytes_per_layer_small_b8_s256, comm_bytes_per_step_small_tp2, comm_bytes_per_step_small_tp4, tp_scaling_ceiling_t_for_t4_pair, tp_scaling_ceiling_t_for_a100_nvlink, full_logits_bytes_tier2_b8_s512, full_logits_bytes_real_vocab, vocab_parallel_ce_collectives, tp2_step_time_vs_single_process
  Some are analytic or inspection results. Check the relevant derivation or test; rerun a benchmark only for quantities it records.
```
