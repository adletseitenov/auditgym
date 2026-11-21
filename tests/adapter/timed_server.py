"""An evaluator server with a short time limit, for the reproducible time-limit scenario C18.

    python tests/adapter/timed_server.py --time-limit 1.0 --results PATH [serve arguments]

The same `python -m auditgym serve`, only with the episode time limit taken from the command line
instead of the 60 s default; used through make_env(server_command=...). Test helper, never shipped
as an entry point.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from auditgym import evaluator, serve  # noqa: E402


def main(argv: list[str]) -> int:
    limit = float(argv[argv.index("--time-limit") + 1])
    rest = argv[:argv.index("--time-limit")] + argv[argv.index("--time-limit") + 2:]
    original = evaluator.Evaluator

    def short(**options):
        return original(time_limit=limit, **options)

    serve.Evaluator = short
    return serve.main(rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
