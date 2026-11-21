"""Numbers that PITCH.md and USAGE.md quote from tools must match what the tools print (DECISIONS AD).
Fast: the demo and the attack table run in under a second; mutants are counted, not run (the mutant
run itself is tests/judge/test_mutants.py and `tools.mutants`)."""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _run(module: str) -> str:
    done = subprocess.run([sys.executable, "-m", module], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return done.stdout


def _heuristic_row() -> list[str]:
    results = _read("docs/RESULTS.md")
    section = results.split("## Генерированный набор (200 эпизодов)", 1)[1].split("\n## ", 1)[0]
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "heuristic":
            return cells
    raise AssertionError("no heuristic row in the generated-set table of docs/RESULTS.md")


class DocsNumbersTest(unittest.TestCase):
    def test_usage_heuristic_numbers_are_from_results(self):
        _, _, precision, recall, f1, *_ = _heuristic_row()
        line = next(line for line in _read("docs/USAGE.md").splitlines() if line.startswith("`heuristic`"))
        for label, value in (("R", recall), ("P", precision), ("F1", f1)):
            self.assertIn(f"{label} {value}", line)

    def test_pitch_heuristic_numbers_are_from_results(self):
        _, _, precision, recall, *_ = _heuristic_row()
        pitch = _read("docs/PITCH.md")
        self.assertIn(f"находит {round(float(recall) * 100)} % ошибок", pitch)
        self.assertIn(f"точность {float(precision):.2f}", pitch)

    def test_pitch_attack_and_mutant_counts(self):
        attack = re.search(r"(\d+)/(\d+) attacks handled", _run("tools.attack"))
        self.assertIsNotNone(attack)
        self.assertEqual(attack.group(1), attack.group(2))
        from tools.mutants import MUTANTS

        pitch = _read("docs/PITCH.md")
        self.assertIn(f"{attack.group(1)}/{attack.group(2)} attacks handled", pitch)
        self.assertIn(f"Атаки на награду {attack.group(1)}/{attack.group(2)}", pitch)
        self.assertIn(f"{len(MUTANTS)}/{len(MUTANTS)} judge mutants killed", pitch)
        self.assertIn(f"мутанты оценщика {len(MUTANTS)}/{len(MUTANTS)}", pitch)

    def test_pitch_demo_numbers(self):
        demo = _run("tools.demo")
        flag_all = re.search(r"«Пометить всё»: выводов \d+, награда (-?\d\.\d{3})", demo)
        self.assertIsNotNone(flag_all)
        pitch = _read("docs/PITCH.md")
        self.assertIn(f"«Пометить всё» {flag_all.group(1).replace('-', '−')}", pitch)
        reward = re.search(r"Награда (\d\.\d{3}) =", demo)
        self.assertIn(f"награда {reward.group(1)}", pitch)
        slide = pitch.split("**Четыре ошибки и два взаимодействия**", 1)[1].split("\n5. ", 1)[0]
        # money amounts and the rate the slide quotes; the demo prints a space as a thousands separator
        # and a comma for the decimals, the rate with a point
        for amount in re.findall(r"\d{1,3}(?: \d{3})*,\d{2} (?:KZT|USD)", slide):
            self.assertIn(amount, demo, f"slide 4 quotes {amount}, the demo does not print it")
        rate = re.search(r"× (\d+\.\d+)", slide)
        self.assertIsNotNone(rate)
        self.assertIn(f"курс {rate.group(1)}", demo)
        # the slide says what the demo says: amount of the fx difference in KZT
        difference = re.search(r"на ([\d ]+,\d{2}) KZT иначе", slide)
        self.assertIn(difference.group(1), demo)


if __name__ == "__main__":
    unittest.main()
