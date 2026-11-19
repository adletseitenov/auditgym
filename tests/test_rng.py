import os
import subprocess
import sys
import unittest

from auditgym.rng import Rng


class RngTest(unittest.TestCase):
    def test_same_seed_same_stream(self):
        self.assertEqual([Rng(1).u64() for _ in range(5)], [Rng(1).u64() for _ in range(5)])

    def test_different_seeds_differ(self):
        self.assertNotEqual(Rng(1).u64(), Rng(2).u64())

    def test_child_does_not_depend_on_parent_draws(self):
        parent = Rng(5)
        before = parent.child("errors").u64()
        parent.u64()
        self.assertEqual(before, parent.child("errors").u64())

    def test_children_are_independent(self):
        root = Rng(5)
        self.assertNotEqual(root.child("a").u64(), root.child("b").u64())

    def test_randint_covers_both_ends(self):
        rng = Rng(3)
        seen = {rng.randint(1, 3) for _ in range(200)}
        self.assertEqual(seen, {1, 2, 3})

    def test_shuffle_is_a_permutation(self):
        items = list(range(50))
        Rng(9).shuffle(items)
        self.assertEqual(sorted(items), list(range(50)))
        self.assertNotEqual(items, list(range(50)))

    def test_golden_values(self):
        # Fixed forever: a change here means every published episode changes.
        self.assertEqual(Rng(42, "root").u64(), GOLDEN_FIRST)
        self.assertEqual(Rng(42).token(8), GOLDEN_TOKEN)

    def test_independent_of_hash_seed(self):
        code = "from auditgym.rng import Rng; print(Rng(42).u64(), Rng('a').token(6))"
        outputs = set()
        for hash_seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            outputs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                       text=True, check=True).stdout)
        self.assertEqual(len(outputs), 1)


GOLDEN_FIRST = 3372235924571515089
GOLDEN_TOKEN = "hr26bcgv"

if __name__ == "__main__":
    unittest.main()
