"""The hand-written mutation table of tools.mutate_gen must not rot: every snippet occurs exactly once in its file."""
import unittest

from tools import mutate_gen


class MutateGenTableTest(unittest.TestCase):
    def test_every_snippet_occurs_exactly_once(self):
        for m in mutate_gen.MUTATIONS:
            text = (mutate_gen.ROOT / m.file).read_text(encoding="utf-8")
            with self.subTest(mutation=m.id):
                self.assertEqual(text.count(m.old), 1, f"{m.id}: snippet must occur once in {m.file}")
                self.assertNotEqual(m.old, m.new, f"{m.id}: the mutation changes nothing")

    def test_table_check_passes(self):
        mutate_gen._check_table()


if __name__ == "__main__":
    unittest.main()
