"""tools/score.py: every zone check is wired into the table, and one crashing check is reported as its
own FAIL with the exception class instead of ending the table."""
import contextlib
import importlib
import io
import unittest
from unittest import mock

from tools import score

MODULES = ("tools.checks_gen", "tools.checks_solver", "tools.checks_qa", "tools.checks_judge", "tools.checks_env")


class WiringTest(unittest.TestCase):
    def test_every_zone_module_is_listed(self):
        listed = {module for modules in score.ZONE_CHECKS.values() for module in modules}
        self.assertEqual(listed, set(MODULES))

    def test_every_implemented_check_is_in_the_table(self):
        score._load_zone_checks()
        table = {name: function for _, _, checks in score.CRITERIA for name, function in checks}
        for module_name in MODULES:
            for name, function in importlib.import_module(module_name).CHECKS:
                name = score.RENAMES.get(name, name)
                with self.subTest(check=name):
                    self.assertIs(table.get(name), function, f"{module_name}: {name} is not wired")

    def test_planned_names_are_unique(self):
        names = [name for _, _, checks in score.CRITERIA for name, _ in checks]
        self.assertEqual(len(names), len(set(names)))


class CrashTest(unittest.TestCase):
    def test_run_check_reports_the_exception_class(self):
        def broken():
            return [][0]

        self.assertEqual(score.run_check(broken), "crashed: IndexError: list index out of range")
        self.assertEqual(score.run_check(lambda: None), None)
        self.assertEqual(score.run_check(lambda: score.sys.exit(3)), "crashed: SystemExit: 3")

    def test_a_crash_fails_only_its_row_and_the_table_is_printed(self):
        def broken():
            raise KeyError("boom")

        criteria = [("A", 10, [("works", lambda: None), ("breaks", broken)]),
                    ("B", 5, [("also works", lambda: None)])]
        out = io.StringIO()
        with mock.patch.object(score, "CRITERIA", criteria), mock.patch.object(score, "ZONE_CHECKS", {}), \
                contextlib.redirect_stdout(out):
            code = score.main([])
        text = out.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] A: breaks -- crashed: KeyError", text)
        self.assertIn("ИТОГО", text)
        self.assertRegex(text, r"B\s+1/1\s+5\.0/5")


class ExitCodeTest(unittest.TestCase):
    """Q5: the table never reports a clean exit while something is not proven."""

    def run_main(self, criteria, argv, zone_checks=None):
        out = io.StringIO()
        zone = {} if zone_checks is None else zone_checks
        with mock.patch.object(score, "CRITERIA", criteria), mock.patch.object(score, "ZONE_CHECKS", zone), \
                contextlib.redirect_stdout(out):
            return score.main(argv), out.getvalue()

    def test_a_todo_fails_unless_allowed(self):
        criteria = [("A", 10, [("works", lambda: None), ("planned", score.todo)])]
        self.assertEqual(self.run_main(criteria, [])[0], 1)
        self.assertEqual(self.run_main(criteria, ["--strict"])[0], 1)
        code, text = self.run_main(criteria, ["--allow-todo"])
        self.assertEqual(code, 0)
        self.assertIn("[TODO] A: planned", text)

    def test_a_pending_check_counts_as_not_passed(self):
        criteria = [("A", 10, [("waits", lambda: "pending: another zone")])]
        self.assertEqual(self.run_main(criteria, [])[0], 1)
        self.assertEqual(self.run_main(criteria, ["--allow-todo"])[0], 0)

    def test_a_failed_check_fails_even_when_todo_is_allowed(self):
        criteria = [("A", 10, [("broken", lambda: "wrong")])]
        self.assertEqual(self.run_main(criteria, ["--allow-todo"])[0], 1)

    def test_a_missing_check_module_fails_unless_allowed(self):
        criteria = [("A", 10, [("works", lambda: None)])]
        zone = {"A": ("tools.checks_that_do_not_exist",)}
        code, text = self.run_main(criteria, [], zone)
        self.assertEqual(code, 1)
        self.assertIn("[MISSING] check module tools.checks_that_do_not_exist", text)
        self.assertEqual(self.run_main(criteria, ["--allow-todo"], zone)[0], 0)

    def test_everything_proven_exits_zero(self):
        self.assertEqual(self.run_main([("A", 10, [("works", lambda: None)])], [])[0], 0)

    def test_a_broken_import_inside_a_module_is_still_an_error(self):
        with mock.patch.object(score.importlib, "import_module", side_effect=ModuleNotFoundError(name="numpy")):
            with mock.patch.object(score, "CRITERIA", [("A", 1, [("x", lambda: None)])]), \
                    mock.patch.object(score, "ZONE_CHECKS", {"A": ("tools.checks_env",)}):
                with self.assertRaises(ModuleNotFoundError):
                    score._load_zone_checks()

    def test_check_command_is_lenient_unless_strict_is_set(self):
        from auditgym import __main__ as entry

        for value, expected in ((None, ["--allow-todo"]), ("0", ["--allow-todo"]), ("", ["--allow-todo"]),
                                ("1", [])):
            environment = {} if value is None else {"AUDITGYM_STRICT": value}
            with mock.patch.dict("os.environ", environment, clear=False), \
                    mock.patch.object(unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()), \
                    mock.patch.object(score, "main", return_value=0) as scoring, \
                    mock.patch("tools.lint_determinism.main", return_value=0), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if value is None:
                    import os

                    os.environ.pop("AUDITGYM_STRICT", None)
                entry.check()
            scoring.assert_called_once_with(expected)


class SampleNamesTest(unittest.TestCase):
    def test_small_checks_state_their_sample(self):
        names = [name for _, _, checks in score.CRITERIA for name, _ in checks]
        for planned in ("reset returns (observation, info) with task, manifest, policy, budget",
                        "no truth, seed, provenance or paths in observations, info, errors",
                        "ids, ordering and counts do not separate affected records from the rest",
                        "reset isolates episodes (A, B, A again equals A)"):
            self.assertTrue(any(name.startswith(planned + " [") and name.endswith("]") for name in names), planned)

    def test_sample_numbers_come_from_the_modules(self):
        from tools import checks_gen

        names = [name for _, _, checks in score.CRITERIA for name, _ in checks]
        want = f"[{len(checks_gen.CONFIGS) * len(checks_gen.SEEDS)} generated episodes]"
        self.assertTrue(any(n.startswith("after injection the deviation equals") and n.endswith(want)
                            for n in names), names)

    def test_labelled_names_are_the_ones_the_modules_find(self):
        score._load_zone_checks()
        todo = [name for _, _, checks in score.CRITERIA for name, function in checks if function is score.todo]
        self.assertLessEqual(len(todo), 4)  # only the four packaging checks may still be planned
        self.assertTrue(all(name.startswith(('one command', 'config and schema', 'measured failure',
                                             'limitations are')) for name in todo), todo)


if __name__ == "__main__":
    unittest.main()
