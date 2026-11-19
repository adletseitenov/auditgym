"""tools/checks_env.py: every env check passes, or waits for another zone ("pending:")."""
import unittest

from tools import checks_env


class EnvChecksTest(unittest.TestCase):
    def test_checks_pass_or_are_pending(self):
        self.assertGreaterEqual(len(checks_env.CHECKS), 15)
        for name, function in checks_env.CHECKS:
            message = function()
            if message is not None:
                self.assertTrue(message.startswith("pending:"), f"{name}: {message}")


if __name__ == "__main__":
    unittest.main()
