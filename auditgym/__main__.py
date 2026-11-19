"""Single entry point: `python -m auditgym check` runs every offline check."""
from __future__ import annotations

import sys
import unittest


def check() -> int:
    from tools import lint_determinism, score

    suite = unittest.defaultTestLoader.discover("tests")
    tests_ok = unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()
    lint_ok = lint_determinism.main() == 0
    score_ok = score.main([]) == 0
    ok = tests_ok and lint_ok and score_ok
    print("\nCHECK:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"]:
        return check()
    if argv[:1] == ["serve"]:
        from .serve import main as serve_main

        return serve_main(argv[1:])
    print("usage: python -m auditgym check | serve [--mode train|eval] [--results PATH]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
