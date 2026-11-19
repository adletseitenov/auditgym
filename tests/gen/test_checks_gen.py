import unittest

from tools import checks_gen


class ChecksGenTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(len(checks_gen.CHECKS), 4)
        for name, function in checks_gen.CHECKS:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(function))

    def test_implemented_checks_pass(self):
        self.assertIsNone(checks_gen.deviation_equals_expected_amount())
        self.assertIsNone(checks_gen.evidence_sets_exist_and_suffice())
        self.assertIsNone(checks_gen.any_seed_and_valid_config_gives_valid_episode())

    def test_configurable_alone_names_what_is_missing(self):
        message = checks_gen.each_error_type_configurable_alone()
        # Until G5 lands, the two remaining types are reported, and nothing else fails.
        self.assertIn(message, (None, "not implemented yet: allocation_gap, fx_error"))


if __name__ == "__main__":
    unittest.main()
