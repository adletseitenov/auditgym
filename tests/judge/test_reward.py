"""J2: metrics and terminal reward against values computed by hand from CASE.md section 05."""
import unittest
from fractions import Fraction

from auditgym.reward import f1_exact, metrics, terminal_reward

# (tp, fp, fn) -> (precision, recall, f1), computed by hand.
METRIC_CASES = [
    ((0, 0, 0), (0.0, 0.0, 1.0)),          # empty truth, empty answer: F1 = 1
    ((0, 3, 0), (0.0, 0.0, 0.0)),          # empty truth, some findings: F1 = 0
    ((0, 0, 2), (0.0, 0.0, 0.0)),          # silence on books with errors
    ((1, 0, 0), (1.0, 1.0, 1.0)),
    ((1, 1, 1), (0.5, 0.5, 0.5)),          # 2/(2+1+1)
    ((2, 1, 0), (2 / 3, 1.0, 0.8)),        # 4/(4+1)
    ((1, 29, 0), (1 / 30, 1.0, 2 / 31)),   # flag everything around one real error
    ((3, 0, 1), (1.0, 0.75, 6 / 7)),
]

# (f1, inspection_steps) -> reward, computed by hand.
REWARD_CASES = [
    ((1, 0), 1.0),
    ((1, 14), 0.972),                      # 1 - 0.028
    ((1, 100), 0.8),
    ((0, 0), -1.0),
    ((0, 50), -1.0),                       # -1.1 clipped
    ((Fraction(1, 2), 10), -0.02),         # 0 - 0.02
    ((Fraction(4, 5), 100), 0.4),          # 1.6 - 1 - 0.2
    ((Fraction(2, 31), 30), 4 / 31 - 1.06),  # flag-everything F1 at 30 steps: about -0.931
]


class MetricsTest(unittest.TestCase):
    def test_hand_computed_metrics(self):
        for counts, expected in METRIC_CASES:
            with self.subTest(counts):
                got = metrics(*counts)
                for key, value in zip(("precision", "recall", "f1"), expected):
                    self.assertAlmostEqual(got[key], value, places=12, msg=key)

    def test_f1_exact_is_a_fraction(self):
        self.assertEqual(f1_exact(2, 1, 0), Fraction(4, 5))
        self.assertEqual(f1_exact(0, 0, 0), 1)

    def test_bad_counts_are_rejected(self):
        for counts in ((-1, 0, 0), (0, True, 0), (0, 0, 1.0)):
            with self.subTest(counts), self.assertRaises(ValueError):
                metrics(*counts)


class TerminalRewardTest(unittest.TestCase):
    def test_hand_computed_rewards(self):
        for (f1, steps), expected in REWARD_CASES:
            with self.subTest(f1=f1, steps=steps):
                self.assertAlmostEqual(terminal_reward(f1, steps), expected, places=12)

    def test_exact_values_are_bit_identical_to_decimal_literals(self):
        # Exact arithmetic: no 0.9720000000000001 from float steps.
        self.assertEqual(terminal_reward(1, 14), 0.972)
        self.assertEqual(terminal_reward(Fraction(4, 5), 100), 0.4)

    def test_float_f1_from_metrics_gives_the_same_bits(self):
        for counts in ((2, 1, 0), (1, 29, 0), (3, 0, 1), (5, 2, 3)):
            with self.subTest(counts):
                exact = terminal_reward(f1_exact(*counts), 37)
                self.assertEqual(terminal_reward(metrics(*counts)["f1"], 37), exact)

    def test_reward_stays_in_range(self):
        for steps in range(0, 101):
            for f1 in (0, Fraction(1, 3), 1):
                self.assertTrue(-1.0 <= terminal_reward(f1, steps) <= 1.0)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            terminal_reward(1, -1)
        with self.assertRaises(ValueError):
            terminal_reward(Fraction(3, 2), 0)
        with self.assertRaises(TypeError):
            terminal_reward(True, 0)
        with self.assertRaises(TypeError):
            terminal_reward("1", 0)


if __name__ == "__main__":
    unittest.main()
