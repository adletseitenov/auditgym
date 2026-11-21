"""Block 4: the twenty paired checks P01-P20 through judge_adapter (tools/adapter_pairs.py plays them).

P01-P05 reorder records of D01-D05, P06-P10 rename ids of D06-D10, P11-P20 do both on D11-D20. Each pair is its
own test so a failure names the pair.
"""
from __future__ import annotations

import tempfile
import unittest

from tools import adapter_pairs


class PairsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_matrix_has_twenty_pairs_in_the_published_split(self):
        kinds = [adapter_pairs.PAIR_KIND[row[0]] for row in adapter_pairs.MATRIX]
        self.assertEqual(kinds, ["order"] * 5 + ["ids"] * 5 + ["both"] * 10)


def _make(index: int):
    def test(self):
        failures = adapter_pairs.run_pair(index, self.scratch.name)
        self.assertEqual(failures, [])
    return test


for _index in range(len(adapter_pairs.MATRIX)):
    setattr(PairsTest, f"test_P{_index + 1:02d}_{adapter_pairs.MATRIX[_index][0]}", _make(_index))


if __name__ == "__main__":
    unittest.main()
