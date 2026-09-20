# Separate-axis pipeline audit

Overall: **PASS**

| Check | Status | Detail |
|---|---|---|
| `canonical_embedding_rows` | PASS | `331 texts / 331 vectors` |
| `canonical_embedding_norm` | PASS | `L2 norms` |
| `query_embedding_rows` | PASS | `565 queries / 565 vectors` |
| `query_embedding_norm` | PASS | `L2 norms` |
| `falcon3_10b/understanding/centroid_norm` | PASS | `k=4` |
| `falcon3_10b/understanding/nonempty` | PASS | `assignments=330` |
| `falcon3_10b/understanding/unique_item_assignment` | PASS | `unique=330` |
| `falcon3_10b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `falcon3_10b/willingness/centroid_norm` | PASS | `k=16` |
| `falcon3_10b/willingness/nonempty` | PASS | `assignments=71` |
| `falcon3_10b/willingness/unique_item_assignment` | PASS | `unique=71` |
| `falcon3_10b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `falcon3_3b/understanding/centroid_norm` | PASS | `k=16` |
| `falcon3_3b/understanding/nonempty` | PASS | `assignments=329` |
| `falcon3_3b/understanding/unique_item_assignment` | PASS | `unique=329` |
| `falcon3_3b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `falcon3_3b/willingness/centroid_norm` | PASS | `k=16` |
| `falcon3_3b/willingness/nonempty` | PASS | `assignments=45` |
| `falcon3_3b/willingness/unique_item_assignment` | PASS | `unique=45` |
| `falcon3_3b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `falcon3_7b/understanding/centroid_norm` | PASS | `k=12` |
| `falcon3_7b/understanding/nonempty` | PASS | `assignments=331` |
| `falcon3_7b/understanding/unique_item_assignment` | PASS | `unique=331` |
| `falcon3_7b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `falcon3_7b/willingness/centroid_norm` | PASS | `k=16` |
| `falcon3_7b/willingness/nonempty` | PASS | `assignments=91` |
| `falcon3_7b/willingness/unique_item_assignment` | PASS | `unique=91` |
| `falcon3_7b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `gemma2_27b/understanding/centroid_norm` | PASS | `k=8` |
| `gemma2_27b/understanding/nonempty` | PASS | `assignments=329` |
| `gemma2_27b/understanding/unique_item_assignment` | PASS | `unique=329` |
| `gemma2_27b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `gemma2_27b/willingness/centroid_norm` | PASS | `k=8` |
| `gemma2_27b/willingness/nonempty` | PASS | `assignments=86` |
| `gemma2_27b/willingness/unique_item_assignment` | PASS | `unique=86` |
| `gemma2_27b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `gemma2_2b_it/understanding/centroid_norm` | PASS | `k=8` |
| `gemma2_2b_it/understanding/nonempty` | PASS | `assignments=329` |
| `gemma2_2b_it/understanding/unique_item_assignment` | PASS | `unique=329` |
| `gemma2_2b_it/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `gemma2_2b_it/willingness/centroid_norm` | PASS | `k=12` |
| `gemma2_2b_it/willingness/nonempty` | PASS | `assignments=43` |
| `gemma2_2b_it/willingness/unique_item_assignment` | PASS | `unique=43` |
| `gemma2_2b_it/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `gemma2_9b_it/understanding/centroid_norm` | PASS | `k=8` |
| `gemma2_9b_it/understanding/nonempty` | PASS | `assignments=330` |
| `gemma2_9b_it/understanding/unique_item_assignment` | PASS | `unique=330` |
| `gemma2_9b_it/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `gemma2_9b_it/willingness/centroid_norm` | PASS | `k=4` |
| `gemma2_9b_it/willingness/nonempty` | PASS | `assignments=78` |
| `gemma2_9b_it/willingness/unique_item_assignment` | PASS | `unique=78` |
| `gemma2_9b_it/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `glm4_9b/understanding/centroid_norm` | PASS | `k=16` |
| `glm4_9b/understanding/nonempty` | PASS | `assignments=331` |
| `glm4_9b/understanding/unique_item_assignment` | PASS | `unique=331` |
| `glm4_9b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `glm4_9b/willingness/centroid_norm` | PASS | `k=12` |
| `glm4_9b/willingness/nonempty` | PASS | `assignments=57` |
| `glm4_9b/willingness/unique_item_assignment` | PASS | `unique=57` |
| `glm4_9b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `llama31_8b_it/understanding/centroid_norm` | PASS | `k=8` |
| `llama31_8b_it/understanding/nonempty` | PASS | `assignments=331` |
| `llama31_8b_it/understanding/unique_item_assignment` | PASS | `unique=331` |
| `llama31_8b_it/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `llama31_8b_it/willingness/centroid_norm` | PASS | `k=16` |
| `llama31_8b_it/willingness/nonempty` | PASS | `assignments=62` |
| `llama31_8b_it/willingness/unique_item_assignment` | PASS | `unique=62` |
| `llama31_8b_it/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `llama32_3b_it/understanding/centroid_norm` | PASS | `k=8` |
| `llama32_3b_it/understanding/nonempty` | PASS | `assignments=329` |
| `llama32_3b_it/understanding/unique_item_assignment` | PASS | `unique=329` |
| `llama32_3b_it/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `llama32_3b_it/willingness/centroid_norm` | PASS | `k=16` |
| `llama32_3b_it/willingness/nonempty` | PASS | `assignments=31` |
| `llama32_3b_it/willingness/unique_item_assignment` | PASS | `unique=31` |
| `llama32_3b_it/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `mistral24b/understanding/centroid_norm` | PASS | `k=8` |
| `mistral24b/understanding/nonempty` | PASS | `assignments=331` |
| `mistral24b/understanding/unique_item_assignment` | PASS | `unique=331` |
| `mistral24b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `mistral24b/willingness/centroid_norm` | PASS | `k=12` |
| `mistral24b/willingness/nonempty` | PASS | `assignments=126` |
| `mistral24b/willingness/unique_item_assignment` | PASS | `unique=126` |
| `mistral24b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `mistral7b/understanding/centroid_norm` | PASS | `k=8` |
| `mistral7b/understanding/nonempty` | PASS | `assignments=331` |
| `mistral7b/understanding/unique_item_assignment` | PASS | `unique=331` |
| `mistral7b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `mistral7b/willingness/centroid_norm` | PASS | `k=16` |
| `mistral7b/willingness/nonempty` | PASS | `assignments=70` |
| `mistral7b/willingness/unique_item_assignment` | PASS | `unique=70` |
| `mistral7b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `phi35_mini/understanding/centroid_norm` | PASS | `k=16` |
| `phi35_mini/understanding/nonempty` | PASS | `assignments=331` |
| `phi35_mini/understanding/unique_item_assignment` | PASS | `unique=331` |
| `phi35_mini/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `phi35_mini/willingness/centroid_norm` | PASS | `k=16` |
| `phi35_mini/willingness/nonempty` | PASS | `assignments=107` |
| `phi35_mini/willingness/unique_item_assignment` | PASS | `unique=107` |
| `phi35_mini/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `phi3_medium_14b/understanding/centroid_norm` | PASS | `k=16` |
| `phi3_medium_14b/understanding/nonempty` | PASS | `assignments=331` |
| `phi3_medium_14b/understanding/unique_item_assignment` | PASS | `unique=331` |
| `phi3_medium_14b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `phi3_medium_14b/willingness/centroid_norm` | PASS | `k=16` |
| `phi3_medium_14b/willingness/nonempty` | PASS | `assignments=81` |
| `phi3_medium_14b/willingness/unique_item_assignment` | PASS | `unique=81` |
| `phi3_medium_14b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_14b/understanding/centroid_norm` | PASS | `k=12` |
| `qwen25_14b/understanding/nonempty` | PASS | `assignments=331` |
| `qwen25_14b/understanding/unique_item_assignment` | PASS | `unique=331` |
| `qwen25_14b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_14b/willingness/centroid_norm` | PASS | `k=4` |
| `qwen25_14b/willingness/nonempty` | PASS | `assignments=123` |
| `qwen25_14b/willingness/unique_item_assignment` | PASS | `unique=123` |
| `qwen25_14b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_32b/understanding/centroid_norm` | PASS | `k=4` |
| `qwen25_32b/understanding/nonempty` | PASS | `assignments=330` |
| `qwen25_32b/understanding/unique_item_assignment` | PASS | `unique=330` |
| `qwen25_32b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_32b/willingness/centroid_norm` | PASS | `k=8` |
| `qwen25_32b/willingness/nonempty` | PASS | `assignments=121` |
| `qwen25_32b/willingness/unique_item_assignment` | PASS | `unique=121` |
| `qwen25_32b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_3b/understanding/centroid_norm` | PASS | `k=16` |
| `qwen25_3b/understanding/nonempty` | PASS | `assignments=331` |
| `qwen25_3b/understanding/unique_item_assignment` | PASS | `unique=331` |
| `qwen25_3b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_3b/willingness/centroid_norm` | PASS | `k=16` |
| `qwen25_3b/willingness/nonempty` | PASS | `assignments=49` |
| `qwen25_3b/willingness/unique_item_assignment` | PASS | `unique=49` |
| `qwen25_3b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_7b/understanding/centroid_norm` | PASS | `k=4` |
| `qwen25_7b/understanding/nonempty` | PASS | `assignments=330` |
| `qwen25_7b/understanding/unique_item_assignment` | PASS | `unique=330` |
| `qwen25_7b/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `qwen25_7b/willingness/centroid_norm` | PASS | `k=8` |
| `qwen25_7b/willingness/nonempty` | PASS | `assignments=98` |
| `qwen25_7b/willingness/unique_item_assignment` | PASS | `unique=98` |
| `qwen25_7b/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `global_anonymous/understanding/centroid_norm` | PASS | `k=12` |
| `global_anonymous/understanding/nonempty` | PASS | `assignments=331` |
| `global_anonymous/understanding/unique_item_assignment` | PASS | `unique=331` |
| `global_anonymous/understanding/canonical_only` | PASS | `no prompt/response fields` |
| `global_anonymous/willingness/centroid_norm` | PASS | `k=4` |
| `global_anonymous/willingness/nonempty` | PASS | `assignments=199` |
| `global_anonymous/willingness/unique_item_assignment` | PASS | `unique=199` |
| `global_anonymous/willingness/canonical_only` | PASS | `no prompt/response fields` |
| `no_joint_artifacts` | PASS | `no joint-named files` |
| `mj/state_count` | PASS | `122880/174080` |
| `mj/split_disjoint` | PASS | `{'selection': 31, 'validation': 17, 'test': 16}` |
| `lg/state_count` | PASS | `83200/108800` |
| `lg/split_disjoint` | PASS | `{'selection': 20, 'validation': 9, 'test': 11}` |
| `search_has_heldout_test` | PASS | `['mj', 'lg']` |
| `search_has_global_frozen_policy` | PASS | `P5_response_state_dynamic` |
| `full_evidence_no_joint_state` | PASS | `{'joint_prior': False, 'joint_posterior': False, 'joint_cluster': False, 'joint_gp': False, 'renderer_only_consumes_two_selected_setting_values': True}` |
| `canonical_final_is_p5` | PASS | `{'status': 'canonical_final', 'method': 'P5_response_state_dynamic', 'joint_prior': False, 'joint_posterior': False}` |
| `understanding_evidence_partition` | PASS | `{'rows': 180064, 'valid_success': 125005, 'valid_failure': 55058, 'invalid': 1}` |
| `willingness_evidence_partition` | PASS | `{'rows': 28135, 'eligible_r1': 23379, 'valid_success': 2708, 'valid_failure': 20671, 'reconstruction_failure': 4741, 'invalid_or_missing_judge': 15}` |
