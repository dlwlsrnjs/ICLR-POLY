import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ALIGNED = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(ALIGNED))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))

import closed_compare
from grid_contract import GridContract
from prepare_understanding32 import (
    CELLS,
    LANGUAGE_ORDER,
    language_permutation,
    languages_for_item,
    load_release,
    render_puzzles,
)


class Understanding32PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ALIGNED / "shared_en_ar/configs/qwen25_7b.json").read_text()
        )
        cls.contract = GridContract(REPO, cls.config["renderer_sha256"])

    def fixture(self):
        prompt = "Please prepare a short library invitation for Saturday morning."
        item = {"id": "fixture", "prompt": prompt}
        bank = {
            ("fixture", language): f"{language} translated library invitation text"
            for language in LANGUAGE_ORDER[1:]
        }
        return item, bank

    def test_exact_32_cells_in_original_mj_lg_iteration_order(self):
        expected = tuple(
            f"g{g}_{order}_n{n}"
            for g in (3, 5, 8, 12)
            for order in ("ordered", "shuffled")
            for n in (2, 4, 6, 8)
        )
        self.assertEqual(CELLS, expected)
        self.assertEqual(len(CELLS), 32)

    def test_prompts_are_byte_identical_to_original_mj_lg_compose(self):
        item, bank = self.fixture()
        splits = {"fixture": "selection"}
        with patch.dict(os.environ, {}, clear=True):
            jobs = render_puzzles([item], bank, splits, self.contract, 17)
            self.assertEqual(len(jobs), 32)
            self.assertEqual({row["frame"] for row in jobs}, {"plain"})
            for job in jobs:
                g, order, n = job["cell"].split("_")
                languages = languages_for_item("fixture", job["cell"], 17)
                row = {
                    "item_id": "fixture",
                    "scenario": "Benign Control",
                    "questions": job["source_translations"],
                }
                expected = closed_compare._compose(
                    row, languages, int(n.removeprefix("n")), int(g.removeprefix("g")), order, ()
                )
                self.assertEqual(job["prompt"], expected)
                self.assertEqual(job["messages"], [{"role": "user", "content": expected}])
                self.assertTrue(job["arm"].endswith("__plain"))

    def test_languages_are_seeded_per_item_nested_and_shared_across_cells(self):
        first = language_permutation("item-a", 17)
        self.assertEqual(first, language_permutation("item-a", 17))
        self.assertNotEqual(first, language_permutation("item-b", 17))
        self.assertNotEqual(first, language_permutation("item-a", 18))
        n2 = languages_for_item("item-a", "g3_ordered_n2", 17)
        n4 = languages_for_item("item-a", "g12_shuffled_n4", 17)
        n6 = languages_for_item("item-a", "g5_ordered_n6", 17)
        n8 = languages_for_item("item-a", "g8_shuffled_n8", 17)
        self.assertEqual(n2, n4[:2])
        self.assertEqual(n4, n6[:4])
        self.assertEqual(n6, n8[:6])
        self.assertEqual(n8[0], "English")
        self.assertEqual(set(n8), set(LANGUAGE_ORDER))

    def test_final_release_is_complete_and_directly_loadable(self):
        release = Path(
            "/home/ljk98/POLY/prior331_runs/translation_repair_20260920/"
            "hf_release_all2317_accepted_v3"
        )
        items, bank, provenance = load_release(release)
        self.assertEqual(len(items), 331)
        self.assertEqual(len(bank), 2317)
        self.assertEqual(provenance["english_items"], 331)
        self.assertEqual(provenance["translation_pairs"], 2317)


if __name__ == "__main__":
    unittest.main()
