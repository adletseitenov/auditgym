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

    def test_the_determinism_lint_runs_once_per_check(self):
        import contextlib
        import io
        from unittest import mock

        from tools import lint_determinism, score

        calls = []

        def fake_score(arguments):  # the real score table calls the lint through one of its rows
            calls.append(lint_determinism.main())
            return 0

        out = io.StringIO()
        with mock.patch.object(check_modes, "suite", lambda full: unittest.TestSuite()), \
                mock.patch.object(check_modes, "case_ranges_line", lambda: True), \
                mock.patch.object(score, "main", fake_score), contextlib.redirect_stdout(out):
            self.assertEqual(check_modes.run(False), 0)
        self.assertEqual(out.getvalue().count("determinism lint:"), 1)
        self.assertEqual(calls, [0])

    def test_the_secret_line_tells_whether_it_is_set_and_never_shows_the_value(self):
        from auditgym.evaluator import SECRET_ENV

        self.assertIn("not set", check_modes.secret_line({}))
        self.assertIn("not set", check_modes.secret_line({SECRET_ENV: ""}))
        line = check_modes.secret_line({SECRET_ENV: "hunter2-value"})
        self.assertIn("set", line)
        self.assertNotIn("not set", line)
        self.assertNotIn("hunter2-value", line)

    def test_check_prints_the_secret_line_once_and_without_the_value(self):
        import contextlib
        import io
        import os
        from unittest import mock

        from auditgym.evaluator import SECRET_ENV
        from tools import score

        out = io.StringIO()
        with mock.patch.dict(os.environ, {SECRET_ENV: "hunter2-value"}), \
                mock.patch.object(check_modes, "suite", lambda full: unittest.TestSuite()), \
                mock.patch.object(check_modes, "case_ranges_line", lambda: True), \
                mock.patch.object(score, "main", lambda arguments: 0), contextlib.redirect_stdout(out):
            check_modes.run(False)
        self.assertEqual(out.getvalue().count(SECRET_ENV + ":"), 1)
        self.assertNotIn("hunter2-value", out.getvalue())

    def test_check_fails_when_a_shipped_subject_leaves_the_case_ranges(self):
        import contextlib
        import io
        from unittest import mock

        from tools import case_ranges, score

        bad = [{"label": "level x", "n": 1, "ok": False, "postings": (1, 1), "documents": (1, 1), "errors": (0, 0)}]
        out = io.StringIO()
        with mock.patch.object(check_modes, "suite", lambda full: unittest.TestSuite()), \
                mock.patch.object(case_ranges, "run", lambda seeds: bad), \
                mock.patch.object(score, "main", lambda arguments: 0), contextlib.redirect_stdout(out):
            self.assertEqual(check_modes.run(False), 1)
        self.assertIn("case ranges: FAIL (1 of 1: level x)", out.getvalue())

    def test_help_and_bad_arguments(self):
        done = subprocess.run([sys.executable, "-m", "auditgym", "--help"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0)
        self.assertIn("check --full", done.stdout)
        bad = subprocess.run([sys.executable, "-m", "auditgym", "check", "--fast"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
