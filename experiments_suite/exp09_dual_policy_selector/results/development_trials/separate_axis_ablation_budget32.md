# Separate-axis POLY ablation

No joint prior or joint posterior is used by P1--P5.

## MJ

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B2_priorless_separate_axis_harmful | 0.8327 | 0.9069 | 0.7185 | 9.43 | 0.00 |
| B1_priorless_joint_gp_ucb_harmful | 0.8318 | 0.9059 | 0.7173 | 9.46 | 0.00 |
| B0_random_harmful | 0.8232 | 0.8965 | 0.6943 | 10.20 | 0.00 |
| P6_cosine_ood_gated_dynamic | 0.8143 | 0.8869 | 0.7042 | 9.87 | 12.97 |
| P5_response_state_dynamic | 0.7555 | 0.8228 | 0.6767 | 10.72 | 37.22 |
| A_no_cosine | 0.7420 | 0.8081 | 0.6757 | 10.75 | 39.20 |
| P3_repeated_harmless | 0.7417 | 0.8078 | 0.6719 | 10.87 | 33.13 |
| A_no_local_search | 0.7417 | 0.8078 | 0.6719 | 10.87 | 33.13 |
| P4_stagnation_local_search | 0.7390 | 0.8048 | 0.6682 | 10.99 | 37.22 |
| A_no_response_state | 0.7390 | 0.8048 | 0.6682 | 10.99 | 37.22 |
| P2_cosine_success_cluster | 0.7270 | 0.7918 | 0.6576 | 11.32 | 0.00 |
| A_no_repeat | 0.7270 | 0.7918 | 0.6576 | 11.32 | 0.00 |
| A_no_success_gate | 0.7188 | 0.7828 | 0.6398 | 11.88 | 0.00 |
| P1_global_success_prior | 0.7022 | 0.7648 | 0.6287 | 12.23 | 0.00 |

## LG

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| P6_cosine_ood_gated_dynamic | 0.8574 | 0.9620 | 0.7848 | 7.32 | 22.52 |
| B1_priorless_joint_gp_ucb_harmful | 0.8529 | 0.9571 | 0.7703 | 7.78 | 0.00 |
| B0_random_harmful | 0.8520 | 0.9560 | 0.7732 | 7.68 | 0.00 |
| B2_priorless_separate_axis_harmful | 0.8515 | 0.9554 | 0.7705 | 7.77 | 0.00 |
| P5_response_state_dynamic | 0.8500 | 0.9538 | 0.7910 | 7.11 | 37.49 |
| P3_repeated_harmless | 0.8265 | 0.9274 | 0.7791 | 7.48 | 34.31 |
| A_no_local_search | 0.8265 | 0.9274 | 0.7791 | 7.48 | 34.31 |
| P4_stagnation_local_search | 0.8265 | 0.9274 | 0.7787 | 7.49 | 37.49 |
| A_no_response_state | 0.8265 | 0.9274 | 0.7787 | 7.49 | 37.49 |
| A_no_cosine | 0.8243 | 0.9249 | 0.7817 | 7.40 | 39.32 |
| P1_global_success_prior | 0.8206 | 0.9208 | 0.7756 | 7.59 | 0.00 |
| P2_cosine_success_cluster | 0.8162 | 0.9158 | 0.7721 | 7.70 | 0.00 |
| A_no_repeat | 0.8162 | 0.9158 | 0.7721 | 7.70 | 0.00 |
| A_no_success_gate | 0.8147 | 0.9142 | 0.7477 | 8.48 | 0.00 |

