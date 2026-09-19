import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import verify_repro_env as verify  # noqa: E402


class VerifyReproEnvironmentTests(unittest.TestCase):
    def test_core_locks_match_environment_checker(self):
        for name in ['requirements-collection.lock.txt', 'Old/environment-used.txt',
                     '23Arm/requirements-reference.lock.txt']:
            entries = {}
            for line in (REPO / name).read_text().splitlines():
                if line.strip() and not line.startswith('#'):
                    package, pinned = line.split('==')
                    entries[package] = pinned
            self.assertEqual(entries, verify.EXPECTED_PACKAGES, name)

    def test_reference_environment_accepts_unset_optional_knobs(self):
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(verify.check_reference_environment())

    def test_reference_environment_accepts_explicit_pinned_revisions(self):
        env = {
            "POLY_RECON_JUDGE_REV": verify.EXPECTED_MODELS["models--Qwen--Qwen2.5-7B-Instruct"],
            "POLY_GUARD_JUDGE_REV": verify.EXPECTED_MODELS["models--Qwen--Qwen3Guard-Gen-8B"],
            "POLY_TARGET_REV": verify.EXPECTED_MODELS["models--Qwen--Qwen2.5-7B-Instruct"],
        }
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(verify.check_reference_environment())

    def test_reference_environment_rejects_behavior_changes(self):
        with mock.patch.dict(os.environ, {"POLY_STRONG_RECON": "1"}, clear=True):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertFalse(verify.check_reference_environment())
        self.assertIn("[XX] env POLY_STRONG_RECON: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
