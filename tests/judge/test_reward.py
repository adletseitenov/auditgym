"""J2: metrics and terminal reward against values computed by hand (CASE §05)."""
import unittest
from fractions import Fraction

from auditgym.reward import metrics, metrics_exact, terminal_reward, terminal_reward_exact

# (tp, fp, fn) -> (P, R, F1), worked out by hand.
METRICS = [
    ((0, 0, 0), (0, 0, 1)),                       # empty truth, empty findings: F1 = 1
    ((0, 3, 0), (0, 0, 0)),                       # empty truth, findings: F1 = 0
    ((0, 0, 4), (0, 0, 0)),                       # silence on errors: P denominator 0
    ((4, 0, 0), (1, 1, 1)),
    ((1, 1, 0), (Fraction(1, 2), 1, Fraction(2, 3))),
    ((3, 1, 1), (Fraction(3, 4), Fraction(3, 4), Fraction(3, 4))),
    ((4, 26, 0), (Fraction(2, 15), 1, Fraction(8, 34))),
    ((1, 29, 0), (Fraction(1, 30), 1, Fraction(2, 31))),
    ((2, 0, 10), (1, Fraction(1, 6), Fraction(4, 14))),
    ((12, 18, 0), (Fraction(2, 5), 1, Fraction(24, 42))),
    ((0, 30, 12), (0, 0, 0)),
    ((5, 5, 5), (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))),
]

# (F1, steps) -> reward, worked out by hand.
REWARDS = [
    (1, 0, 1), (1, 1, Fraction(499, 500)), (1, 10, Fraction(49, 50)), (1, 100, Fraction(4, 5)),
    (0, 0, -1), (0, 100, -1), (Fraction(1, 2), 0, 0), (Fraction(1, 2), 1, Fraction(-1, 500)),
    (Fraction(2, 3), 10, Fraction(1, 3) - Fraction(1, 50)), (Fraction(3, 4), 50, Fraction(2, 5)),
    (1, 1000, -1), (Fraction(1, 10), 5, Fraction(-81, 100)),
]


class MetricsTest(unittest.TestCase):
    def test_hand_values(self):
        for counts, (p, r, f1) in METRICS:
            with self.subTest(counts=counts):
                exact = metrics_exact(*counts)
                self.assertEqual((exact["precision"], exact["recall"], exact["f1"]),
                                 (Fraction(p), Fraction(r), Fraction(f1)))
                floats = metrics(*counts)
                self.assertEqual(floats["f1"], Fraction(f1).numerator / Fraction(f1).denominator)
                self.assertEqual((floats["precision"], floats["recall"]),
                                 (Fraction(p).numerator / Fraction(p).denominator,
                                  Fraction(r).numerator / Fraction(r).denominator))

    def test_rejects_bad_counts(self):
        for bad in ((-1, 0, 0), (True, 0, 0), (1.0, 0, 0), ("1", 0, 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                metrics(*bad)


class RewardTest(unittest.TestCase):
    def test_hand_values(self):
        for f1, steps, reward in REWARDS:
            with self.subTest(f1=f1, steps=steps):
                self.assertEqual(terminal_reward_exact(f1, steps), reward)
                self.assertAlmostEqual(terminal_reward(f1, steps), reward.numerator / reward.denominator,
                                       places=12)

    def test_reward_is_the_literal_float_formula(self):
        """R4: the organizers' formula typed into Python is the reward, bit for bit."""
        differs_from_exact = 0
        for tp in range(0, 8):
            for fp in range(0, 8):
                for fn in range(0, 8):
                    f1 = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
                    for steps in (0, 1, 3, 7, 10, 33, 50, 99, 100):
                        want = max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * steps))
                        self.assertEqual(terminal_reward(metrics(tp, fp, fn)["f1"], steps).hex(), want.hex(),
                                         (tp, fp, fn, steps))
                        exact = terminal_reward_exact(Fraction(2 * tp, 2 * tp + fp + fn) if tp + fp + fn else 1,
                                                      steps)
                        differs_from_exact += want != exact.numerator / exact.denominator
        self.assertGreater(differs_from_exact, 0, "the grid must contain a case where float and exact differ")

    def test_judge_reward_is_the_literal_float_formula(self):
        from auditgym.judge import judge
        from tests.judge.helpers import episode, oracle, submit

        ep = episode(3)
        for k in range(0, 4):
            findings = oracle(ep)[:2] + [dict(oracle(ep)[0], affected_ids=[f"f{j:02d}"]) for j in range(k)]
            for steps in (0, 1, 7, 33, 100):
                result = judge(ep, submit(ep, findings), steps)
                f1 = 2 * result["tp"] / (2 * result["tp"] + result["fp"] + result["fn"])
                self.assertEqual(result["reward"].hex(), max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * steps)).hex())

    def test_bit_identical_repeat(self):
        first = [terminal_reward(Fraction(a, 30), s).hex() for a in range(31) for s in range(101)]
        second = [terminal_reward(Fraction(a, 30), s).hex() for a in range(31) for s in range(101)]
        self.assertEqual(terminal_reward(1, 100), 0.8)
        self.assertEqual(first, second)

    def test_always_in_range(self):
        for a in range(0, 61):
            for s in range(0, 201, 7):
                self.assertTrue(-1.0 <= terminal_reward(Fraction(a, 60), s) <= 1.0)

    def test_rejects_bad_arguments(self):
        for f1, steps in ((Fraction(3, 2), 0), (-0.1, 0), (True, 0), ("1", 0), (1, -1), (1, 1.0), (1, True)):
            with self.subTest(f1=f1, steps=steps), self.assertRaises(ValueError):
                terminal_reward(f1, steps)


if __name__ == "__main__":
    unittest.main()
