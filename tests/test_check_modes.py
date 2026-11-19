"""The fast and the full mode of `python -m auditgym check` (docs/AUDIT.md C12)."""
import subprocess
import sys
import unittest

from tools import check_modes


class ModesTest(unittest.TestCase):
    def test_every_full_only_name_matches_a_test_and_a_slow_name_never_hides_a_fast_one(self):
        ids = [test.id() for test in check_modes.discover()]
        for name in check_modes.FULL_ONLY:
            with self.subTest(name=name):
                self.assertTrue(any(i == name or i.startswith(name + ".") for i in ids), f"{name} matches no test")
        self.assertGreater(len(ids), 400)

    def test_fast_is_a_strict_subset_of_full(self):
        fast = {test.id() for test in check_modes.suite(False)}
        full = {test.id() for test in check_modes.suite(True)}
        self.assertLess(fast, full)
        self.assertFalse(any(check_modes.is_full_only(i) for i in fast))
        self.assertTrue(all(check_modes.is_full_only(i) for i in full - fast))

    def test_schemas_and_contract_tests_stay_in_the_fast_mode(self):
        fast = {test.id() for test in check_modes.suite(False)}
        for needle in ("tests.test_schemas.", "tests.env.test_evaluator.", "tests.judge.test_matcher.",
                       "tests.judge.test_attacks.", "tests.gen.test_config_storage."):
            self.assertTrue(any(i.startswith(needle) for i in fast), needle)

    def test_help_and_bad_arguments(self):
        done = subprocess.run([sys.executable, "-m", "auditgym", "--help"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0)
        self.assertIn("check --full", done.stdout)
        bad = subprocess.run([sys.executable, "-m", "auditgym", "check", "--fast"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
