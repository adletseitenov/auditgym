"""The «Надёжность награды» checks for tools/score.py pass and match the planned names."""
import unittest

from tools import checks_judge, score


class JudgeChecksTest(unittest.TestCase):
    def test_names_match_the_plan(self):
        planned = [name for criterion, _, checks in score.CRITERIA
                   if criterion == "Надёжность награды" for name, _ in checks]
        self.assertEqual([name for name, _ in checks_judge.CHECKS], planned)

    def test_every_check_passes(self):
        for name, check in checks_judge.CHECKS:
            with self.subTest(name):
                self.assertIsNone(check())


if __name__ == "__main__":
    unittest.main()
