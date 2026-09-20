# Separate-axis POLY ablation

No joint prior or joint posterior is used by P1--P5.

## MJ

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B1_priorless_joint_gp_ucb_harmful | 0.7463 | 0.8128 | 0.5815 | 5.40 | 0.00 |
| B0_random_harmful | 0.7243 | 0.7888 | 0.5407 | 5.87 | 0.00 |
| P1_global_success_prior | 0.6471 | 0.7047 | 0.5290 | 5.98 | 0.00 |
| A_no_cosine | 0.6305 | 0.6867 | 0.5202 | 6.07 | 40.00 |
| P2_cosine_success_cluster | 0.6250 | 0.6807 | 0.5156 | 6.12 | 0.00 |
| A_no_repeat | 0.6250 | 0.6807 | 0.5156 | 6.12 | 0.00 |
| P4_stagnation_local_search | 0.6158 | 0.6707 | 0.5085 | 6.21 | 37.58 |
| A_no_response_state | 0.6158 | 0.6707 | 0.5085 | 6.21 | 37.58 |
| P5_response_state_dynamic | 0.6149 | 0.6697 | 0.5084 | 6.21 | 37.58 |
| P3_repeated_harmless | 0.6131 | 0.6677 | 0.5060 | 6.23 | 34.17 |
| A_no_local_search | 0.6131 | 0.6677 | 0.5060 | 6.23 | 34.17 |
| B2_priorless_separate_axis_harmful | 0.6085 | 0.6627 | 0.3859 | 7.67 | 0.00 |
| A_no_success_gate | 0.4301 | 0.4685 | 0.2766 | 8.90 | 0.00 |

## LG

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| P1_global_success_prior | 0.8118 | 0.9109 | 0.7222 | 3.74 | 0.00 |
| P2_cosine_success_cluster | 0.8103 | 0.9092 | 0.7224 | 3.74 | 0.00 |
| A_no_repeat | 0.8103 | 0.9092 | 0.7224 | 3.74 | 0.00 |
| P4_stagnation_local_search | 0.8088 | 0.9076 | 0.7067 | 3.92 | 37.68 |
| P5_response_state_dynamic | 0.8088 | 0.9076 | 0.7067 | 3.92 | 37.68 |
| A_no_response_state | 0.8088 | 0.9076 | 0.7067 | 3.92 | 37.68 |
| P3_repeated_harmless | 0.8059 | 0.9043 | 0.7061 | 3.93 | 34.79 |
| A_no_local_search | 0.8059 | 0.9043 | 0.7061 | 3.93 | 34.79 |
| B0_random_harmful | 0.7985 | 0.8960 | 0.6705 | 4.35 | 0.00 |
| B1_priorless_joint_gp_ucb_harmful | 0.7971 | 0.8944 | 0.6662 | 4.40 | 0.00 |
| A_no_cosine | 0.7912 | 0.8878 | 0.6882 | 4.14 | 40.00 |
| B2_priorless_separate_axis_harmful | 0.7559 | 0.8482 | 0.5440 | 5.85 | 0.00 |
| A_no_success_gate | 0.6897 | 0.7739 | 0.4970 | 6.38 | 0.00 |

