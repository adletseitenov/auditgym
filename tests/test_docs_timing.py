"""The check duration is stated in one place (README); Makefile and Dockerfile only point to it."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class CheckTimingDocsTest(unittest.TestCase):
    def test_readme_states_the_range(self):
        readme = _read("README.md")
        self.assertIn("4–7 минут", readme)
        self.assertIn("10–14 минут", readme)

    def test_no_stale_durations_elsewhere(self):
        for name in ("Makefile", "Dockerfile"):
            text = _read(name)
            self.assertIsNone(re.search(r"about \d+ ?s\b|minute and a half", text), name)
            self.assertIn("README", text, name)


if __name__ == "__main__":
    unittest.main()
