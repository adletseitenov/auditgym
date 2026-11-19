"""Single entry point.

    python -m auditgym check           fast: lint, the score table and the unit tests, under two minutes
    python -m auditgym check --full    everything: all tests, fuzz, leak scan, metamorphic pairs, attacks
    python -m auditgym serve [--mode train|eval] [--results PATH]    the evaluator process

`AUDITGYM_STRICT=1 python -m auditgym check` also fails on any planned score check that is still a
TODO, on a `pending:` check and on a missing check module (the default only lists them); `--full` is
always strict. What each mode runs is in tools/check_modes.py.
"""
from __future__ import annotations

import sys

USAGE = "usage: python -m auditgym check [--full] | serve [--mode train|eval] [--results PATH]"


def main(argv: list[str]) -> int:
    if argv[:1] in (["-h"], ["--help"], ["help"]):
        print(__doc__.strip())
        return 0
    if argv[:1] == ["check"]:
        flags = argv[1:]
        if flags not in ([], ["--full"]):
            print(USAGE, file=sys.stderr)
            return 2
        from tools.check_modes import run

        return run(full=bool(flags))
    if argv[:1] == ["serve"]:
        from .serve import main as serve_main

        return serve_main(argv[1:])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
