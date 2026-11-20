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

        before = (matcher.same_type, matcher.match, judge.terminal_reward, judge.judge, findings.MAX_FINDINGS)
        picked = {"matcher: type not checked", "matcher: greedy first-come matching", "reward: no step cost",
                  "judge: a structure error reports no missed issues", "structure: no limit of 30 findings"}
        chosen = [mutant for mutant in mutants.MUTANTS if mutant[0] in picked]
        self.assertEqual(len(chosen), len(picked))
        mutants.run(chosen)
        self.assertEqual((matcher.same_type, matcher.match, judge.terminal_reward, judge.judge,
                          findings.MAX_FINDINGS), before)


if __name__ == "__main__":
    unittest.main()
