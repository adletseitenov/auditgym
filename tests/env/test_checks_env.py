"""tools/checks_env.py: every env check passes. A "pending:" message is not a pass (audit Q7)."""
import unittest

from tools import checks_env


class EnvChecksTest(unittest.TestCase):
    def test_every_check_passes(self):
        self.assertGreaterEqual(len(checks_env.CHECKS), 15)
        for name, function in checks_env.CHECKS:
            with self.subTest(name):
                self.assertIsNone(function())

    def test_the_leak_check_runs_on_companies_with_errors(self):
        from auditgym.generate import generate

        self.assertTrue(generate(48151623, checks_env.WITH_ERRORS)["private"]["truth"]["issues"])


if __name__ == "__main__":
    unittest.main()
