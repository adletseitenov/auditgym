"""docs/ADAPTER.md lists all 92 checks (C01-C20, D01-D20, Block 3 tests 1-20, M01-M12, P01-P20) with the test
that implements each; this keeps the table honest: exactly these IDs, every test name resolves to a test."""
from __future__ import annotations

import importlib
import pathlib
import re
import unittest

DOC = pathlib.Path(__file__).resolve().parent.parent.parent / "docs" / "ADAPTER.md"
ROW = re.compile(r"^\| (?P<id>(?:C|D|M|P)\d\d|B3-\d\d) \| .*? \| `(?P<test>test_\w+\.\w+\.test_\w+)` \| (?P<status>.+) \|$")


def expected_ids() -> list[str]:
    return ([f"C{n:02d}" for n in range(1, 21)] + [f"D{n:02d}" for n in range(1, 21)]
            + [f"B3-{n:02d}" for n in range(1, 21)] + [f"M{n:02d}" for n in range(1, 13)]
            + [f"P{n:02d}" for n in range(1, 21)])


class AdapterTableTest(unittest.TestCase):
    def rows(self) -> list[re.Match]:
        lines = DOC.read_text(encoding="utf-8").splitlines()
        return [m for m in (ROW.match(line) for line in lines) if m]

    def test_the_table_has_exactly_the_92_checks_once(self):
        ids = [m["id"] for m in self.rows()]
        self.assertEqual(sorted(ids), sorted(expected_ids()))
        self.assertEqual(len(ids), 92)

    def test_every_named_test_exists_and_every_status_is_pass(self):
        for m in self.rows():
            module, cls, method = m["test"].split(".")
            holder = getattr(importlib.import_module(f"tests.adapter.{module}"), cls, None)
            self.assertTrue(holder is not None and callable(getattr(holder, method, None)), m["test"])
            self.assertTrue(m["status"].startswith("pass"), (m["id"], m["status"]))


if __name__ == "__main__":
    unittest.main()
