# Separate-axis POLY ablation

No joint prior or joint posterior is used by P1--P5.

## MJ

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B2_priorless_separate_axis_harmful | 0.7647 | 0.8328 | 0.5697 | 5.55 | 0.00 |
| B1_priorless_joint_gp_ucb_harmful | 0.7463 | 0.8128 | 0.5815 | 5.40 | 0.00 |
| P6_cosine_ood_gated_dynamic | 0.7417 | 0.8078 | 0.5669 | 5.57 | 12.97 |
| B0_random_harmful | 0.7195 | 0.7836 | 0.5462 | 5.81 | 0.00 |
| A_no_cosine | 0.7045 | 0.7672 | 0.5893 | 5.28 | 39.20 |
| P5_response_state_dynamic | 0.7031 | 0.7658 | 0.5783 | 5.41 | 37.22 |
| P3_repeated_harmless | 0.7022 | 0.7648 | 0.5818 | 5.37 | 33.13 |
| A_no_local_search | 0.7022 | 0.7648 | 0.5818 | 5.37 | 33.13 |
| P4_stagnation_local_search | 0.6976 | 0.7598 | 0.5780 | 5.41 | 37.22 |
| A_no_response_state | 0.6976 | 0.7598 | 0.5780 | 5.41 | 37.22 |
| P2_cosine_success_cluster | 0.6884 | 0.7497 | 0.5656 | 5.56 | 0.00 |
| A_no_repeat | 0.6884 | 0.7497 | 0.5656 | 5.56 | 0.00 |
| A_no_success_gate | 0.6820 | 0.7427 | 0.5344 | 5.93 | 0.00 |
| P1_global_success_prior | 0.6599 | 0.7187 | 0.5354 | 5.91 | 0.00 |

## LG

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B2_priorless_separate_axis_harmful | 0.8118 | 0.9109 | 0.6570 | 4.52 | 0.00 |
| P5_response_state_dynamic | 0.8103 | 0.9092 | 0.7198 | 3.77 | 37.49 |
| P6_cosine_ood_gated_dynamic | 0.8074 | 0.9059 | 0.6947 | 4.07 | 22.52 |
| B0_random_harmful | 0.8049 | 0.9032 | 0.6716 | 4.34 | 0.00 |
| A_no_cosine | 0.8036 | 0.9017 | 0.7221 | 3.74 | 39.32 |
| P1_global_success_prior | 0.8015 | 0.8993 | 0.7111 | 3.87 | 0.00 |
| P3_repeated_harmless | 0.8000 | 0.8977 | 0.7169 | 3.80 | 34.31 |
| A_no_local_search | 0.8000 | 0.8977 | 0.7169 | 3.80 | 34.31 |
| P4_stagnation_local_search | 0.8000 | 0.8977 | 0.7154 | 3.81 | 37.49 |
| A_no_response_state | 0.8000 | 0.8977 | 0.7154 | 3.81 | 37.49 |
| P2_cosine_success_cluster | 0.7971 | 0.8944 | 0.7104 | 3.87 | 0.00 |
| A_no_repeat | 0.7971 | 0.8944 | 0.7104 | 3.87 | 0.00 |
| B1_priorless_joint_gp_ucb_harmful | 0.7971 | 0.8944 | 0.6662 | 4.40 | 0.00 |
| A_no_success_gate | 0.7912 | 0.8878 | 0.6515 | 4.58 | 0.00 |

