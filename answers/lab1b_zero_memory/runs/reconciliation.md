```
Predictions vs measurements: lab1b_zero_memory
==================================================================================================
quantity                                   predicted        measured       units     error        
--------------------------------------------------------------------------------------------------
bytes_per_param_mixed_adamw                       16              16       bytes      0.0%  ok
bytes_per_param_fp32_adamw                        16    not recorded       bytes            unmeasured
states_bytes_small_preset_n1               1.723e+08       1.716e+08       bytes      0.4%  ok
states_bytes_small_preset_zero1_n4         7.539e+07       7.507e+07       bytes      0.4%  ok
zero1_reduction_factor_n4                      2.286    not recorded           x            unmeasured
ddp_bytes_per_step_small_n4                3.231e+07    not recorded       bytes            unmeasured
zero1_bytes_per_step_small_n4              3.231e+07    not recorded       bytes            unmeasured
activation_bytes_small_b8_s256             2.863e+08    not recorded       bytes            unmeasured
activation_bytes_small_b8_s512             8.242e+08    not recorded       bytes            unmeasured
measured_over_analytic_ratio_n1                  2.2    not recorded           x            unmeasured
max_params_8gb_plain_adamw                 4.048e+08    not recorded  parameters            unmeasured
min_ranks_for_1_5b_zero1                          14    not recorded       ranks            unmeasured
optimizer_state_reduction_n4           not predicted           3.967           x            unpredicted
--------------------------------------------------------------------------------------------------

Measured but never predicted: optimizer_state_reduction_n4

Predictions without a recorded counterpart: bytes_per_param_fp32_adamw, zero1_reduction_factor_n4, ddp_bytes_per_step_small_n4, zero1_bytes_per_step_small_n4, activation_bytes_small_b8_s256, activation_bytes_small_b8_s512, measured_over_analytic_ratio_n1, max_params_8gb_plain_adamw, min_ranks_for_1_5b_zero1
  Some are analytic or inspection results. Check the relevant derivation or test; rerun a benchmark only for quantities it records.
```
