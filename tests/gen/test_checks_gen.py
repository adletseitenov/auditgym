import io
import unittest

from tools import checks_gen


class ChecksGenTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(len(checks_gen.CHECKS), 6)
        for name, function in checks_gen.CHECKS:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(function))

    def test_generator_checks_pass(self):
        for name, function in checks_gen.CHECKS:
            with self.subTest(check=name):
                self.assertIsNone(function())



if __name__ == "__main__":
    unittest.main()
