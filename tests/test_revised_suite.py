import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import closed_compare as cc  # noqa: E402
from gp_bai import arm_feats  # noqa: E402
sys.path.insert(0, str(REPO / "experiments_suite" / "exp03_heterogeneity"))
from analyze import analyze as analyze_heterogeneity  # noqa: E402


MJ_ORDER = ["Bengali", "Swahili", "Javanese", "Arabic", "Korean", "Thai",
            "Chinese", "Italian", "Vietnamese"]
LG_ORDER = ["Norwegian", "Finnish", "Arabic", "Russian", "German", "Japanese",
            "Chinese", "French", "Spanish"]


def benign_row(order):
    questions = {"English": "Explain how to maintain a bicycle safely."}
    questions.update({lang: f"{lang} parallel benign maintenance sentence." for lang in order})
    return {"item_id": "safe-1", "scenario": "Benign", "original": questions["English"],
            "questions": questions}


class RevisedSuiteTests(unittest.TestCase):
    def test_arm_space_and_collection_translation(self):
        for order, tlang in ((MJ_ORDER, "Bengali"), (LG_ORDER, "Norwegian")):
            arms = cc.build_arms(order, tlang)
            self.assertEqual(len(arms), 292)
            self.assertEqual(len({a[0] for a in arms}), 292)
            builders = {name: build for name, *_rest, build in arms}
            row = benign_row(order)
            self.assertEqual(builders["m_translated"](row), row["questions"][tlang])
            self.assertNotEqual(builders["m_translated"](row), row["questions"]["English"])

    def test_translation_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "translation language"):
            cc.validate_tlang([benign_row(MJ_ORDER)], "Norwegian", "MultiJail")

    def test_frozen_recipe_metric_is_applied(self):
        signals = {
            "persona": {"detail": 0.2, "fiction_hold": 0.8},
            "fiction": {"detail": 0.5, "fiction_hold": 0.9},
            "pap": {"detail": 0.4, "fiction_hold": 0.7},
        }
        self.assertAlmostEqual(cc.willingness_factor(signals, "persona+fiction", "detail"), 0.1)
        self.assertAlmostEqual(cc.willingness_factor(signals, "persona+fiction", "fiction_hold"), 0.72)
        self.assertEqual(cc.willingness_factor(signals, "comp", "detail"), 1.0)

    def test_gp_features_support_legacy_and_revised_names(self):
        legacy = arm_feats(["frag3_ordered_n2", "frag12_shuffled_n8"])
        self.assertEqual(legacy.shape, (2, 3))
        revised = arm_feats([
            "g3_ordered_n2__plain",
            "g12_shuffled_n8__persona+fiction+pap",
            "g5_ordered_n4__role",
            "m_translated",
        ])
        self.assertEqual(revised.shape, (4, 9))
        self.assertEqual(revised[1, 3:6].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(revised[2, 3], 1.0)  # role wrapper includes the persona frame
        self.assertEqual(revised[2, 6], 1.0)
        self.assertEqual(revised[3, 7:].tolist(), [1.0, 1.0])

    def test_manifest_upsert_deduplicates_output_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            man = cc._manifest(root, 292)
            first = {"phase": "probe", "path": "benign/model_mj.json", "value": 1}
            second = {"phase": "probe", "path": "benign/model_mj.json", "value": 2}
            cc._upsert_manifest_run(man, first)
            cc._upsert_manifest_run(man, second)
            self.assertEqual(man["schema"], 2)
            self.assertEqual(man["n_arms"], 292)
            self.assertEqual(man["runs"], [second])

    def test_manifest_write_merges_independent_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            left = cc._manifest(root, 292)
            right = cc._manifest(root, 292)
            cc._upsert_manifest_run(left, {"path": "attack/model_mj.json", "ts": "2026-09-10T01:00:00"})
            cc._upsert_manifest_run(right, {"path": "attack/model_lg.json", "ts": "2026-09-10T01:00:01"})
            cc._write_manifest(root, left)
            cc._write_manifest(root, right)
            saved = json.loads((root / "MANIFEST.json").read_text())
            self.assertEqual({r["path"] for r in saved["runs"]},
                             {"attack/model_mj.json", "attack/model_lg.json"})
            self.assertEqual((root / "MANIFEST.json").stat().st_mode & 0o777, 0o600)

    def test_attack_manifest_record_uses_relative_paths(self):
        root = Path("run-root")
        args = argparse.Namespace(model="model", tag="tag", collection="MultiJail")
        agg = {"verified": 0.5, "recon": 0.75, "unsafe": 0.5, "n": 64,
               "ts": "2026-09-10T01:02:03"}
        record = cc._attack_manifest_record(
            args, "arm", agg, root / "attack/_raw/tag__arm.jsonl",
            root / "attack/tag__arm.json", root)
        self.assertEqual(record["path"], "attack/tag__arm.json")
        self.assertEqual(record["raw_path"], "attack/_raw/tag__arm.jsonl")
        self.assertEqual(record["n"], 64)

    def test_offline_manifest_repair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "attack/_raw").mkdir(parents=True)
            agg = {"model": "model", "tag": "tag", "collection": "MultiJail",
                   "method": "arm", "gated": True, "verified": 0.5, "recon": 1.0,
                   "unsafe": 0.5, "n": 2, "ts": "2026-09-10T01:02:03"}
            (root / "attack/tag__arm.json").write_text(json.dumps(agg))
            (root / "attack/_raw/tag__arm.jsonl").write_text("{}\n{}\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cc.phase_repair_manifest(argparse.Namespace(root=str(root))), 0)
            saved = json.loads((root / "MANIFEST.json").read_text())
            self.assertEqual(len(saved["runs"]), 1)
            self.assertEqual(saved["runs"][0]["path"], "attack/tag__arm.json")

    def test_heterogeneity_rejects_partial_matrix(self):
        matrix = {"model_mj": {"arm_a": 0.5, "arm_b": 0.6}}
        with self.assertRaisesRegex(ValueError, "incomplete full matrix"):
            analyze_heterogeneity(matrix, expected_arms=292)
        progress = analyze_heterogeneity(matrix, expected_arms=292, allow_incomplete=True)
        self.assertFalse(progress["complete"])
        self.assertEqual(progress["arm_counts"], {"model_mj": 2})

    def test_offline_audit_uses_collection_translation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            order_path = root / "order.json"
            harm_path = root / "rows.jsonl"
            out = root / "audit"
            order_path.write_text(json.dumps({"order": MJ_ORDER}))
            harm_path.write_text(json.dumps(benign_row(MJ_ORDER)) + "\n")
            args = argparse.Namespace(collection="MultiJail", root=str(out),
                                      order=str(order_path), harm=str(harm_path),
                                      tlang="AUTO", n_items=1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cc.phase_audit(args), 0)
            report = json.loads((out / "dedupe_audit.json").read_text())
            self.assertEqual(report["n_arms"], 292)
            self.assertEqual(report["translation_language"], "Bengali")

    def test_gated_score_requires_answer_section(self):
        class Target:
            def generate(self, prompts):
                return ["[RECONSTRUCTED] unsafe request text only"] * len(prompts)

        class Judges:
            def answer_unsafe(self, originals, answers):
                self.answers = answers
                return [True] * len(answers)

            def recon_equivalent(self, originals, reconstructions):
                return [True] * len(reconstructions)

        judges = Judges()
        raw = io.StringIO()
        verified, recon, unsafe = cc._score_capture(
            Target(), judges, [benign_row(MJ_ORDER)], lambda row: "prompt", True, raw)
        self.assertEqual((verified, recon, unsafe), (0.0, 1.0, 0.0))
        self.assertEqual(judges.answers, [""])
        record = json.loads(raw.getvalue())
        self.assertFalse(record["answer_section_present"])


if __name__ == "__main__":
    unittest.main()
