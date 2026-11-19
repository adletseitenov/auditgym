import io
import unittest

from tools import checks_gen, fuzz


class ChecksGenTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(len(checks_gen.CHECKS), 8)
        for name, function in checks_gen.CHECKS:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(function))

    def test_generator_checks_pass(self):
        for name, function in checks_gen.CHECKS[:-1]:
            with self.subTest(check=name):
                self.assertIsNone(function())

    def test_fuzz_check_passes(self):
        self.assertIsNone(checks_gen.fuzz_solver_equals_truth())


class FileRecomputationTest(unittest.TestCase):
    """Q5: the file-based checks can fail: each tampering is caught without verify()."""

    def recompute(self, change):
        import pathlib
        import tempfile

        from auditgym.generate import generate
        from auditgym.storage import write_episode
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            write_episode(generate(3, {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1,
                                                  "period_cutoff": 1}, "currencies": ["USD"]}), root)
            files = checks_gen._read(root)
            self.assertEqual(checks_gen._recompute(files), [])
            change(files)
            return checks_gen._recompute(files)

    def test_tampered_amount(self):
        for index in range(4):
            def change(files, index=index):
                files["truth"]["issues"][index]["expected_amount_minor"] += 2
            self.assertTrue(self.recompute(change), index)

    def test_tampered_row(self):
        def change(files):
            files["ledger"][0]["debit_minor" if files["ledger"][0]["debit_minor"] else "credit_minor"] += 1
        self.assertTrue(any("differ" in p or "unbalanced" in p for p in self.recompute(change)))

    def test_unknown_evidence(self):
        def change(files):
            files["truth"]["issues"][0]["evidence_sets"][0].append("nosuchid00")
        self.assertTrue(self.recompute(change))

    def test_insufficient_evidence(self):
        for index in range(4):
            def change(files, index=index):
                group = files["truth"]["issues"][index]["evidence_sets"][0]
                group[:] = files["truth"]["issues"][index]["affected_ids"] + [files["ledger"][0]["row_id"]]
            self.assertTrue(self.recompute(change), index)

    def test_wrong_count(self):
        def change(files):
            files["manifest"]["counts"]["payments"] += 1
        self.assertTrue(self.recompute(change))


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
