# Confirmatory family-clustered statistics

The model family is the unit of inference (7 clusters). CIs are family-cluster bootstraps; p-values are exact two-sided sign-flip tests and are unadjusted.

| Experiment | Treatment vs control | B | Δ family-macro ASR | 95% CI | p |
|---|---|---:|---:|---:|---:|
| strict_itemheldout | advanced_nested_meanstd vs structured_gp_crossfamily_incumbent | 2 | +0.0175 | [-0.0029, +0.0428] | 0.2713 |
| strict_itemheldout | advanced_nested_cvar25 vs structured_gp_crossfamily_incumbent | 2 | +0.0160 | [-0.0046, +0.0417] | 0.3488 |
| strict_itemheldout | advanced_nested_minimax vs structured_gp_crossfamily_incumbent | 2 | +0.0169 | [-0.0044, +0.0429] | 0.3023 |
| strict_itemheldout | advanced_nested_meanstd vs structured_gp_crossfamily_incumbent | 4 | +0.0039 | [-0.0145, +0.0286] | 0.8915 |
| strict_itemheldout | advanced_nested_cvar25 vs structured_gp_crossfamily_incumbent | 4 | +0.0082 | [-0.0170, +0.0397] | 0.7209 |
| strict_itemheldout | advanced_nested_minimax vs structured_gp_crossfamily_incumbent | 4 | +0.0077 | [-0.0140, +0.0377] | 0.8605 |
| strict_itemheldout | advanced_nested_meanstd vs structured_gp_crossfamily_incumbent | 8 | -0.0060 | [-0.0146, +0.0044] | 0.3178 |
| strict_itemheldout | advanced_nested_cvar25 vs structured_gp_crossfamily_incumbent | 8 | -0.0049 | [-0.0164, +0.0066] | 0.5039 |
| strict_itemheldout | advanced_nested_minimax vs structured_gp_crossfamily_incumbent | 8 | -0.0021 | [-0.0115, +0.0072] | 0.6899 |
| strict_itemheldout | advanced_nested_meanstd vs structured_gp_crossfamily_incumbent | 16 | -0.0022 | [-0.0102, +0.0082] | 0.7054 |
| strict_itemheldout | advanced_nested_cvar25 vs structured_gp_crossfamily_incumbent | 16 | +0.0015 | [-0.0073, +0.0143] | 0.9535 |
| strict_itemheldout | advanced_nested_minimax vs structured_gp_crossfamily_incumbent | 16 | +0.0006 | [-0.0095, +0.0106] | 0.9380 |
| equal_item_cost_multifidelity | multifidelity_graph_kernel_ucb vs full_fidelity_gp | 2 | -0.0277 | [-0.0625, +0.0000] | 0.1318 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top8 vs full_fidelity_gp | 2 | -0.0272 | [-0.0671, +0.0041] | 0.3023 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top16 vs full_fidelity_gp | 2 | -0.0318 | [-0.0767, +0.0027] | 0.2558 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top32 vs full_fidelity_gp | 2 | -0.0319 | [-0.0764, +0.0032] | 0.2093 |
| equal_item_cost_multifidelity | multifidelity_graph_kernel_ucb vs full_fidelity_gp | 4 | -0.0237 | [-0.0598, +0.0053] | 0.2093 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top8 vs full_fidelity_gp | 4 | -0.0196 | [-0.0549, +0.0080] | 0.3643 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top16 vs full_fidelity_gp | 4 | -0.0231 | [-0.0615, +0.0074] | 0.3023 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top32 vs full_fidelity_gp | 4 | -0.0324 | [-0.0744, +0.0001] | 0.1783 |
| equal_item_cost_multifidelity | multifidelity_graph_kernel_ucb vs full_fidelity_gp | 8 | -0.0358 | [-0.0668, -0.0098] | 0.0388 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top8 vs full_fidelity_gp | 8 | -0.0344 | [-0.0660, -0.0068] | 0.0388 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top16 vs full_fidelity_gp | 8 | -0.0297 | [-0.0626, -0.0005] | 0.1783 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top32 vs full_fidelity_gp | 8 | -0.0309 | [-0.0622, -0.0042] | 0.1008 |
| equal_item_cost_multifidelity | multifidelity_graph_kernel_ucb vs full_fidelity_gp | 16 | -0.0418 | [-0.0816, -0.0103] | 0.0388 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top8 vs full_fidelity_gp | 16 | -0.0409 | [-0.0748, -0.0100] | 0.0233 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top16 vs full_fidelity_gp | 16 | -0.0361 | [-0.0714, -0.0037] | 0.1008 |
| equal_item_cost_multifidelity | prior_seeded_successive_halving_top32 vs full_fidelity_gp | 16 | -0.0266 | [-0.0611, +0.0002] | 0.1473 |
| gpu_familyloo_selector | ppo_context vs supervised_context | 0 | +0.0229 | [-0.0039, +0.0503] | 0.2093 |
| gpu_familyloo_selector | ppo_context vs ppo_no_context | 0 | +0.0075 | [-0.0110, +0.0290] | 0.5659 |
| gpu_familyloo_selector | ppo_context vs ppo_no_feedback | 0 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| gpu_familyloo_selector | ppo_context vs random_probe_best_observed | 0 | +0.0074 | [-0.0294, +0.0464] | 0.7519 |
| gpu_familyloo_selector | ppo_query_best_observed vs random_probe_best_observed | 0 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| gpu_familyloo_selector | ppo_query_best_observed vs ppo_no_feedback_query_best_observed | 0 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| gpu_familyloo_selector | ppo_context vs supervised_context | 2 | +0.0015 | [-0.0167, +0.0209] | 0.8605 |
| gpu_familyloo_selector | ppo_context vs ppo_no_context | 2 | +0.0034 | [-0.0148, +0.0257] | 0.8450 |
| gpu_familyloo_selector | ppo_context vs ppo_no_feedback | 2 | -0.0033 | [-0.0245, +0.0149] | 0.8450 |
| gpu_familyloo_selector | ppo_context vs random_probe_best_observed | 2 | +0.0407 | [+0.0031, +0.0782] | 0.1008 |
| gpu_familyloo_selector | ppo_query_best_observed vs random_probe_best_observed | 2 | +0.0437 | [+0.0032, +0.0846] | 0.1318 |
| gpu_familyloo_selector | ppo_query_best_observed vs ppo_no_feedback_query_best_observed | 2 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| gpu_familyloo_selector | ppo_context vs supervised_context | 4 | +0.0080 | [-0.0060, +0.0229] | 0.4729 |
| gpu_familyloo_selector | ppo_context vs ppo_no_context | 4 | +0.0112 | [-0.0068, +0.0305] | 0.3333 |
| gpu_familyloo_selector | ppo_context vs ppo_no_feedback | 4 | -0.0017 | [-0.0230, +0.0226] | 0.9070 |
| gpu_familyloo_selector | ppo_context vs random_probe_best_observed | 4 | +0.0083 | [-0.0303, +0.0391] | 0.6434 |
| gpu_familyloo_selector | ppo_query_best_observed vs random_probe_best_observed | 4 | +0.0063 | [-0.0339, +0.0442] | 0.7674 |
| gpu_familyloo_selector | ppo_query_best_observed vs ppo_no_feedback_query_best_observed | 4 | +0.0008 | [-0.0020, +0.0039] | 0.7519 |
| gpu_familyloo_selector | ppo_context vs supervised_context | 8 | +0.0007 | [-0.0183, +0.0203] | 0.9225 |
| gpu_familyloo_selector | ppo_context vs ppo_no_context | 8 | +0.0173 | [-0.0036, +0.0378] | 0.1938 |
| gpu_familyloo_selector | ppo_context vs ppo_no_feedback | 8 | +0.0158 | [-0.0072, +0.0414] | 0.2868 |
| gpu_familyloo_selector | ppo_context vs random_probe_best_observed | 8 | -0.0108 | [-0.0537, +0.0212] | 0.8140 |
| gpu_familyloo_selector | ppo_query_best_observed vs random_probe_best_observed | 8 | -0.0093 | [-0.0420, +0.0239] | 0.6434 |
| gpu_familyloo_selector | ppo_query_best_observed vs ppo_no_feedback_query_best_observed | 8 | -0.0008 | [-0.0024, +0.0000] | 1.0000 |
| gpu_familyloo_selector | ppo_context vs supervised_context | 16 | +0.0080 | [-0.0094, +0.0294] | 0.5659 |
| gpu_familyloo_selector | ppo_context vs ppo_no_context | 16 | +0.0160 | [-0.0031, +0.0370] | 0.2558 |
| gpu_familyloo_selector | ppo_context vs ppo_no_feedback | 16 | +0.0154 | [+0.0008, +0.0331] | 0.2558 |
| gpu_familyloo_selector | ppo_context vs random_probe_best_observed | 16 | -0.0196 | [-0.0571, +0.0106] | 0.3798 |
| gpu_familyloo_selector | ppo_query_best_observed vs random_probe_best_observed | 16 | -0.0073 | [-0.0391, +0.0224] | 0.7209 |
| gpu_familyloo_selector | ppo_query_best_observed vs ppo_no_feedback_query_best_observed | 16 | -0.0020 | [-0.0044, +0.0000] | 0.5039 |
