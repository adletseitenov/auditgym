"""Every judge mutant of tools.mutants is killed by the attack table or a reward property."""
import unittest

from tools import mutants


class JudgeMutantsTest(unittest.TestCase):
    def test_every_mutant_is_killed(self):
        survivors = [name for name, failed in mutants.run() if not failed]
        self.assertEqual(survivors, [])

    def test_the_unmutated_judge_passes_every_detector(self):
        self.assertEqual(mutants._killed_by(), [])

    def test_patches_are_undone(self):
        from auditgym import findings, judge, matcher

        before = (matcher.same_type, judge.terminal_reward, findings.MAX_FINDINGS)
        mutants.run(mutants.MUTANTS[:1] + mutants.MUTANTS[10:11] + mutants.MUTANTS[19:20])
        self.assertEqual((matcher.same_type, judge.terminal_reward, findings.MAX_FINDINGS), before)


if __name__ == "__main__":
    unittest.main()
