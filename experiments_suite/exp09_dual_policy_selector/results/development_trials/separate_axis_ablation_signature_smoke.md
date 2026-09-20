# Separate-axis POLY ablation

No joint prior or joint posterior is used by P1--P5.

## MJ

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B1_priorless_joint_gp_ucb_harmful | 0.7463 | 0.8128 | 0.5815 | 5.40 | 0.00 |
| B0_random_harmful | 0.7243 | 0.7888 | 0.5407 | 5.87 | 0.00 |
| B2_priorless_separate_axis_harmful | 0.6085 | 0.6627 | 0.3859 | 7.67 | 0.00 |
| A_no_cosine | 0.5800 | 0.6316 | 0.4689 | 6.66 | 38.90 |
| P2_cosine_success_cluster | 0.5772 | 0.6286 | 0.4670 | 6.68 | 0.00 |
| A_no_repeat | 0.5772 | 0.6286 | 0.4670 | 6.68 | 0.00 |
| P3_repeated_harmless | 0.5607 | 0.6106 | 0.4520 | 6.86 | 34.52 |
| A_no_local_search | 0.5607 | 0.6106 | 0.4520 | 6.86 | 34.52 |
| P4_stagnation_local_search | 0.5597 | 0.6096 | 0.4511 | 6.87 | 35.88 |
| P5_response_state_dynamic | 0.5597 | 0.6096 | 0.4511 | 6.87 | 35.88 |
| A_no_response_state | 0.5597 | 0.6096 | 0.4511 | 6.87 | 35.88 |
| P1_global_success_prior | 0.5184 | 0.5646 | 0.4094 | 7.35 | 0.00 |
| A_no_success_gate | 0.4301 | 0.4685 | 0.2766 | 8.90 | 0.00 |

## LG

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B0_random_harmful | 0.7985 | 0.8960 | 0.6705 | 4.35 | 0.00 |
| B1_priorless_joint_gp_ucb_harmful | 0.7971 | 0.8944 | 0.6662 | 4.40 | 0.00 |
| A_no_cosine | 0.7853 | 0.8812 | 0.6902 | 4.11 | 39.12 |
| P1_global_success_prior | 0.7824 | 0.8779 | 0.6859 | 4.16 | 0.00 |
| P4_stagnation_local_search | 0.7809 | 0.8762 | 0.6865 | 4.15 | 36.15 |
| P5_response_state_dynamic | 0.7809 | 0.8762 | 0.6865 | 4.15 | 36.15 |
| A_no_response_state | 0.7809 | 0.8762 | 0.6865 | 4.15 | 36.15 |
| P3_repeated_harmless | 0.7794 | 0.8746 | 0.6865 | 4.15 | 34.87 |
| A_no_local_search | 0.7794 | 0.8746 | 0.6865 | 4.15 | 34.87 |
| P2_cosine_success_cluster | 0.7691 | 0.8630 | 0.6736 | 4.30 | 0.00 |
| A_no_repeat | 0.7691 | 0.8630 | 0.6736 | 4.30 | 0.00 |
| B2_priorless_separate_axis_harmful | 0.7559 | 0.8482 | 0.5440 | 5.85 | 0.00 |
| A_no_success_gate | 0.6897 | 0.7739 | 0.4970 | 6.38 | 0.00 |

