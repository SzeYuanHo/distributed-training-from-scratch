```
Predictions vs measurements: lab4_capstone_2d
=======================================================================================================================
quantity                                    predicted        measured                           units     error        
-----------------------------------------------------------------------------------------------------------------------
best_config_7b_on_8xa100                          811    not recorded  encoded as dp*100 + tp*10 + pp            unmeasured
memory_per_gpu_best_config                  7.286e+10    not recorded                           bytes            unmeasured
comm_bytes_per_step_best_config              2.45e+10    not recorded                           bytes            unmeasured
does_ranking_change_on_two_nodes                    1    not recorded             1 for yes, 0 for no            unmeasured
dp_group_allreduces_per_step_2x2                   28    not recorded                     collectives            unmeasured
tp_group_allreduces_per_step_2x2                   13    not recorded                     collectives            unmeasured
model_flops_per_token_coefficient                   6               6                   dimensionless      0.0%  ok
predicted_mfu_2xt4                               0.12    not recorded                        fraction            unmeasured
hfu_minus_mfu_with_recompute                    0.333    not recorded                        fraction            unmeasured
measured_tokens_per_s_single_device     not predicted           2,586                        tokens/s            unpredicted
-----------------------------------------------------------------------------------------------------------------------

Measured but never predicted: measured_tokens_per_s_single_device

Predictions without a recorded counterpart: best_config_7b_on_8xa100, memory_per_gpu_best_config, comm_bytes_per_step_best_config, does_ranking_change_on_two_nodes, dp_group_allreduces_per_step_2x2, tp_group_allreduces_per_step_2x2, predicted_mfu_2xt4, hfu_minus_mfu_with_recompute
  Some are analytic or inspection results. Check the relevant derivation or test; rerun a benchmark only for quantities it records.
```
