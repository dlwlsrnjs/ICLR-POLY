# Separate-axis POLY ablation

No joint prior or joint posterior is used by P1--P5.

## MJ

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B2_priorless_separate_axis_harmful | 0.7546 | 0.8218 | 0.5669 | 5.57 | 0.00 |
| B1_priorless_joint_gp_ucb_harmful | 0.7463 | 0.8128 | 0.5815 | 5.40 | 0.00 |
| B0_random_harmful | 0.7243 | 0.7888 | 0.5407 | 5.87 | 0.00 |
| P2_cosine_success_cluster | 0.6884 | 0.7497 | 0.5656 | 5.56 | 0.00 |
| A_no_repeat | 0.6884 | 0.7497 | 0.5656 | 5.56 | 0.00 |
| P5_response_state_dynamic | 0.6857 | 0.7467 | 0.5592 | 5.63 | 35.88 |
| A_no_success_gate | 0.6829 | 0.7437 | 0.5343 | 5.93 | 0.00 |
| P3_repeated_harmless | 0.6801 | 0.7407 | 0.5576 | 5.65 | 34.52 |
| A_no_local_search | 0.6801 | 0.7407 | 0.5576 | 5.65 | 34.52 |
| P4_stagnation_local_search | 0.6783 | 0.7387 | 0.5569 | 5.66 | 35.88 |
| A_no_response_state | 0.6783 | 0.7387 | 0.5569 | 5.66 | 35.88 |
| A_no_cosine | 0.6627 | 0.7217 | 0.5481 | 5.75 | 38.90 |
| P1_global_success_prior | 0.6599 | 0.7187 | 0.5354 | 5.91 | 0.00 |

## LG

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B2_priorless_separate_axis_harmful | 0.8191 | 0.9191 | 0.6570 | 4.53 | 0.00 |
| P5_response_state_dynamic | 0.8162 | 0.9158 | 0.7184 | 3.79 | 36.12 |
| P1_global_success_prior | 0.8015 | 0.8993 | 0.7111 | 3.87 | 0.00 |
| B0_random_harmful | 0.7985 | 0.8960 | 0.6705 | 4.35 | 0.00 |
| P2_cosine_success_cluster | 0.7971 | 0.8944 | 0.7104 | 3.87 | 0.00 |
| A_no_repeat | 0.7971 | 0.8944 | 0.7104 | 3.87 | 0.00 |
| A_no_cosine | 0.7971 | 0.8944 | 0.7087 | 3.89 | 39.12 |
| B1_priorless_joint_gp_ucb_harmful | 0.7971 | 0.8944 | 0.6662 | 4.40 | 0.00 |
| A_no_success_gate | 0.7941 | 0.8911 | 0.6531 | 4.56 | 0.00 |
| P4_stagnation_local_search | 0.7926 | 0.8894 | 0.7094 | 3.88 | 36.12 |
| A_no_response_state | 0.7926 | 0.8894 | 0.7094 | 3.88 | 36.12 |
| P3_repeated_harmless | 0.7926 | 0.8894 | 0.7093 | 3.89 | 34.87 |
| A_no_local_search | 0.7926 | 0.8894 | 0.7093 | 3.89 | 34.87 |

