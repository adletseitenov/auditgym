"""DECISIONS Y: on Python older than 3.11 `python -m auditgym` and judge_adapter stop at once with a clear
message instead of failing later with an obscure error."""
from __future__ import annotations

import pathlib
import runpy
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


class OldPythonTest(unittest.TestCase):
    def refused(self, run):
        with mock.patch.object(sys, "version_info", (3, 10, 4, "final", 0)):
            with self.assertRaises(SystemExit) as caught:
                run()
        message = str(caught.exception.code)
        self.assertIn("needs Python 3.11 or newer", message)
        self.assertIn("this is Python 3.10", message)
        self.assertIn("python3.11", message)

    def test_main_module_refuses_python_3_10(self):
        self.refused(lambda: runpy.run_module("auditgym.__main__", run_name="not_main"))

    def test_judge_adapter_refuses_python_3_10(self):
        self.refused(lambda: runpy.run_path(str(ROOT / "judge_adapter.py"), run_name="not_main"))

    def test_the_current_interpreter_is_let_through(self):
        namespace = runpy.run_module("auditgym.__main__", run_name="not_main")
        self.assertIn("main", namespace)


if __name__ == "__main__":
    unittest.main()
