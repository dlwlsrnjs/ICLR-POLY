import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments_suite/exp04_budget_queryeff/pilot_advanced_additive_bo.py"
SPEC = importlib.util.spec_from_file_location("advanced_additive_bo_tested", MODULE_PATH)
abo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = abo
assert SPEC.loader is not None
SPEC.loader.exec_module(abo)


class FakeBase:
    @staticmethod
    def model_family(tag):
        return tag.split("_")[0]


class AdvancedAdditiveBOTests(unittest.TestCase):
    def test_graph_anova_kernel_is_normalized_psd(self):
        axes = np.asarray(list(np.ndindex(4, 2, 4, 5)), dtype=int)
        rng = np.random.default_rng(3)
        sources = rng.random((6, len(axes)))
        kernel, weights = abo.build_kernel(
            sources, axes, beta=0.7, interaction=0.25, task_mix=0.35
        )
        self.assertTrue(np.allclose(np.diag(kernel), 1.0))
        self.assertGreaterEqual(float(np.linalg.eigvalsh(kernel).min()), -1e-8)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_target_family_is_excluded_from_transfer_sources(self):
        arms = ["a", "b"]
        panel = {
            "qwen_1": {"mj": {"a": 0.1, "b": 0.2}},
            "qwen_2": {"mj": {"a": 0.3, "b": 0.4}},
            "llama_1": {"mj": {"a": 0.8, "b": 0.9}},
        }
        vectors = abo.source_family_vectors(FakeBase(), panel, "qwen", "mj", arms)
        self.assertTrue(np.allclose(vectors, [[0.8, 0.9]]))

    def test_unobserved_target_values_do_not_enter_first_query(self):
        axes = np.asarray(list(np.ndindex(2, 2, 2, 2)), dtype=int)
        sources = np.stack([
            np.linspace(0.1, 0.9, len(axes)),
            np.linspace(0.2, 0.8, len(axes)),
        ])
        config = abo.Config(
            name="test", beta=0.7, interaction=0.25, task_mix=0.35,
            acquisition="ei", exploration=0.01, noise=0.0025,
            dynamic_strength=0.75,
        )
        first = np.zeros(len(axes))
        second = np.ones(len(axes))
        _, q1, *_ = abo.search_curve(first, sources, axes, config, 1, np.random.default_rng(5))
        _, q2, *_ = abo.search_curve(second, sources, axes, config, 1, np.random.default_rng(5))
        self.assertEqual(q1, q2)


if __name__ == "__main__":
    unittest.main()
