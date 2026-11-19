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

    def test_rejects_bad_counts(self):
        for bad in ((-1, 0, 0), (True, 0, 0), (1.0, 0, 0), ("1", 0, 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                metrics(*bad)


class RewardTest(unittest.TestCase):
    def test_hand_values(self):
        for f1, steps, reward in REWARDS:
            with self.subTest(f1=f1, steps=steps):
                self.assertEqual(terminal_reward_exact(f1, steps), reward)
                self.assertEqual(terminal_reward(f1, steps), reward.numerator / reward.denominator)

    def test_float_is_exact_nearest(self):
        # Float arithmetic would give 0.7999999999999998 here.
        self.assertEqual(terminal_reward(1, 100), 0.8)
        self.assertEqual(terminal_reward(1, 1), 0.998)
        self.assertEqual(terminal_reward(Fraction(2, 3), 0).hex(), (1 / 3).hex())

    def test_bit_identical_repeat(self):
        first = [terminal_reward(Fraction(a, 30), s).hex() for a in range(31) for s in range(101)]
        second = [terminal_reward(Fraction(a, 30), s).hex() for a in range(31) for s in range(101)]
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
