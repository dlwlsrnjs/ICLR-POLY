# Separate-axis POLY ablation

No joint prior or joint posterior is used by P1--P5.

## MJ

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| B1_priorless_joint_gp_ucb_harmful | 0.7463 | 0.8128 | 0.5815 | 5.40 | 0.00 |
| B0_random_harmful | 0.7243 | 0.7888 | 0.5407 | 5.87 | 0.00 |
| P4_stagnation_local_search | 0.6792 | 0.7397 | 0.5698 | 5.50 | 35.88 |
| A_no_response_state | 0.6792 | 0.7397 | 0.5698 | 5.50 | 35.88 |
| P3_repeated_harmless | 0.6783 | 0.7387 | 0.5695 | 5.51 | 34.52 |
| A_no_local_search | 0.6783 | 0.7387 | 0.5695 | 5.51 | 34.52 |
| P2_cosine_success_cluster | 0.6774 | 0.7377 | 0.5642 | 5.57 | 0.00 |
| A_no_repeat | 0.6774 | 0.7377 | 0.5642 | 5.57 | 0.00 |
| P5_response_state_dynamic | 0.6719 | 0.7317 | 0.5585 | 5.63 | 35.88 |
| A_no_cosine | 0.6700 | 0.7297 | 0.5658 | 5.55 | 38.90 |
| P1_global_success_prior | 0.6627 | 0.7217 | 0.5556 | 5.66 | 0.00 |
| B2_priorless_separate_axis_harmful | 0.6278 | 0.6837 | 0.3995 | 7.52 | 0.00 |
| A_no_success_gate | 0.6112 | 0.6657 | 0.4057 | 7.44 | 0.00 |

## LG

| Method | Verified@12 | Oracle recovery | AUC | Harmful pulls | Harmless requests |
|---|---:|---:|---:|---:|---:|
| P5_response_state_dynamic | 0.8147 | 0.9142 | 0.7186 | 3.78 | 36.15 |
| P1_global_success_prior | 0.8015 | 0.8993 | 0.7172 | 3.79 | 0.00 |
| P4_stagnation_local_search | 0.8000 | 0.8977 | 0.7120 | 3.86 | 36.15 |
| A_no_response_state | 0.8000 | 0.8977 | 0.7120 | 3.86 | 36.15 |
| P3_repeated_harmless | 0.7985 | 0.8960 | 0.7121 | 3.85 | 34.87 |
| A_no_local_search | 0.7985 | 0.8960 | 0.7121 | 3.85 | 34.87 |
| B0_random_harmful | 0.7985 | 0.8960 | 0.6705 | 4.35 | 0.00 |
| B1_priorless_joint_gp_ucb_harmful | 0.7971 | 0.8944 | 0.6662 | 4.40 | 0.00 |
| A_no_cosine | 0.7956 | 0.8927 | 0.7105 | 3.87 | 39.12 |
| P2_cosine_success_cluster | 0.7941 | 0.8911 | 0.7080 | 3.90 | 0.00 |
| A_no_repeat | 0.7941 | 0.8911 | 0.7080 | 3.90 | 0.00 |
| A_no_success_gate | 0.7588 | 0.8515 | 0.5583 | 5.68 | 0.00 |
| B2_priorless_separate_axis_harmful | 0.7397 | 0.8300 | 0.5463 | 5.81 | 0.00 |

