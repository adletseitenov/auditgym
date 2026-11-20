"""`python -m auditgym check` outside a repository clone (an installed package) says so and exits non-zero."""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class CheckWithoutCheckoutTest(unittest.TestCase):
    def test_check_from_a_copy_of_the_package_alone(self):
        with tempfile.TemporaryDirectory() as scratch:
            shutil.copytree(ROOT / "auditgym", pathlib.Path(scratch) / "auditgym",
                            ignore=shutil.ignore_patterns("__pycache__"))
            for flags in ([], ["--full"]):
                done = subprocess.run([sys.executable, "-m", "auditgym", "check", *flags], cwd=scratch,
                                      capture_output=True, text=True)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertIn("cd <repo> && python -m auditgym check", done.stderr)
                self.assertNotIn("Traceback", done.stderr)
                self.assertEqual(done.stdout, "")


if __name__ == "__main__":
    unittest.main()
