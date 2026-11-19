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


if __name__ == "__main__":
    unittest.main()
