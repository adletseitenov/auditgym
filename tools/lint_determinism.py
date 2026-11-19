"""Fails when the core package uses a source of nondeterminism or a non-stdlib import."""
from __future__ import annotations

import pathlib
import re
import sys

CORE = pathlib.Path(__file__).resolve().parent.parent / "auditgym"

# Pattern -> why it is forbidden in the core package.
FORBIDDEN = {
    r"^\s*(import|from)\s+random\b": "use auditgym.rng.Rng",
    r"^\s*(import|from)\s+(uuid|secrets)\b": "use Rng.token",
    r"\bos\.urandom\b": "use auditgym.rng.Rng",
    r"\btime\.(time|monotonic|perf_counter)\b": "the clock must not reach episode data",
    r"\bdatetime\.(now|today|utcnow)\b": "the clock must not reach episode data",
    r"\bfloat\(": "money and rates are int and Decimal",
    r"^\s*(import|from)\s+(numpy|pandas|gymnasium|requests|httpx|yaml)\b": "stdlib only",
}
# A line may opt out with a reason: `# determinism: <why this is safe>`.
ALLOW = re.compile(r"#\s*determinism:\s*\S")


def main() -> int:
    problems = []
    for path in sorted(CORE.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if ALLOW.search(line):
                continue
            for pattern, why in FORBIDDEN.items():
                if re.search(pattern, line):
                    problems.append(f"{path.relative_to(CORE.parent)}:{number}: {why}: {line.strip()}")
    for problem in problems:
        print(problem)
    print(f"determinism lint: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
