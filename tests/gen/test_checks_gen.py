import io
import unittest

from tools import checks_gen, fuzz


class ChecksGenTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(len(checks_gen.CHECKS), 7)
        for name, function in checks_gen.CHECKS:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(function))

    def test_generator_checks_pass(self):
        for name, function in checks_gen.CHECKS[:-1]:
            with self.subTest(check=name):
                self.assertIsNone(function())

    def test_fuzz_check_passes_or_waits_for_the_solver(self):
        message = checks_gen.fuzz_solver_equals_truth()
        self.assertTrue(message is None or message.startswith("pending:"), message)


class FuzzToolTest(unittest.TestCase):
    def test_grid_covers_the_required_cases(self):
        names = [name for name, _ in fuzz.grid()]
        for required in ("clean", "clean_plain", "I1", "I2", "max_density_small", "max_density_large",
                         *fuzz.TYPES, "duplicate_invoice+period_cutoff", "allocation_gap+fx_error"):
            self.assertIn(required, names)
        configs = [cfg for _, cfg in fuzz.grid()]
        self.assertTrue({cfg.get("timezone") for cfg in configs} >= {"-11:00", "+14:00", "+05:45", "-09:30"})
        self.assertTrue({len(cfg.get("currencies", [])) for cfg in configs} >= {0, 1, 2, 3})

    def test_run_reports(self):
        out = io.StringIO()
        result = fuzz.run(1, ("I2",), out=out)
        self.assertGreater(result["episodes"], 0)
        self.assertIn("attempts per episode", out.getvalue())
        self.assertIn("time per episode", out.getvalue())


if __name__ == "__main__":
    unittest.main()
