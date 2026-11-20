"""What `python -m auditgym check` runs (docs/AUDIT.md C12).

Fast (default): the whole score table (tools/score.py, which includes the determinism lint) and every unit test except
FULL_ONLY. Full (`--full`): all of that, every test, and the long statistical tools in EXTRA_TOOLS.

FULL_ONLY lists test modules and classes that are skipped in the fast mode for one of two reasons:
they call the very checks the score table already runs in the same command (the `*_checks` tests), or
they are sampling statistics that the full mode also runs through `tools.leakscan` and `tools.fuzz`.
A name here is a prefix of a test id (`module` or `module.Class`); tests/test_check_modes.py fails when a
name matches no test, so a rename cannot silently put a slow test back or drop a fast one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest

FULL_ONLY = (
    "tests.gen.test_checks_gen",              # = the "Данные и эталон" rows of the score table
    "tests.env.test_checks_env",             # = the "Контракт и изоляция" rows
    "tests.judge.test_checks",               # = the "Надёжность награды" rows
    "tests.solver.test_solver.ScoreChecksTest",
    "tests.env.test_leakscan",               # = python -m tools.leakscan
    "tests.solver.test_solver.MetamorphicTest",   # = python -m tools.metamorph
    "tests.gen.test_inject.LeakTest",        # surface statistics on hundreds of episodes
    "tests.gen.test_inject.GridTest",        # = python -m tools.fuzz
)
# (label, module arguments). Each runs as `python -m <args>` and must exit 0.
EXTRA_TOOLS = (
    ("fuzz, 100 seeds per config", "tools.fuzz --seeds 100"),
    ("leak scan", "tools.leakscan"),
    ("metamorphic pairs", "tools.metamorph"),
    ("reward attacks", "tools.attack"),
)


def _flatten(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def is_full_only(test_id: str) -> bool:
    return any(test_id == name or test_id.startswith(name + ".") for name in FULL_ONLY)


def discover(start: str = "tests") -> list[unittest.TestCase]:
    return list(_flatten(unittest.defaultTestLoader.discover(start, top_level_dir=".")))


def suite(full: bool, start: str = "tests") -> unittest.TestSuite:
    """Every test (full) or every test not in FULL_ONLY (fast)."""
    result = unittest.TestSuite()
    result.addTests(test for test in discover(start) if full or not is_full_only(test.id()))
    return result


def secret_line(environ=None) -> str:
    """Whether the evaluator secret is set (docs/DECISIONS.md H). The value is never printed."""
    from auditgym.secret import SECRET_ENV, SECRET_FILE_ENV, secret_source

    environ = os.environ if environ is None else environ
    source = secret_source(environ)
    if source == "file":
        return f"{SECRET_ENV}: set via {SECRET_FILE_ENV} (the closed run is not reproducible without it)"
    if source == "value":
        return (f"{SECRET_ENV}: set as a value (a policy started from this environment sees it too; "
                f"prefer {SECRET_FILE_ENV} readable only by the evaluator)")
    return (f"{SECRET_ENV}: not set (open episodes are reproducible from the seed; "
            f"set it for a closed run, see README «Закрытая проверка»)")


def _phase(name: str, started: float) -> None:
    print(f"[{name}: {time.perf_counter() - started:.0f} s]", flush=True)


def _extra_tool(label: str, arguments: str) -> bool:
    started = time.perf_counter()
    done = subprocess.run([sys.executable, "-m", *arguments.split()], capture_output=True, text=True)
    lines = (done.stdout + done.stderr).strip().splitlines()
    print(f"{label}: {'ok' if done.returncode == 0 else 'FAILED'} -- {lines[-1] if lines else ''}", flush=True)
    if done.returncode != 0:
        print("\n".join(lines[-40:]))
    _phase(label, started)
    return done.returncode == 0


def run(full: bool = False) -> int:
    """The body of `python -m auditgym check [--full]`: exit code 0 only when everything passed."""
    from tools import score

    began = started = time.perf_counter()
    print(secret_line(), flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(suite(full))
    print(f"unit tests: {result.testsRun} run" + ("" if full else f" (fast mode; --full also runs "
                                                   f"{len(FULL_ONLY)} more groups)"))
    _phase("tests", started)
    # Development: a planned check that is not written yet is listed but does not fail the run.
    # The submission gate is AUDITGYM_STRICT=1, and --full: any TODO, `pending:` or missing module fails.
    strict = full or os.environ.get("AUDITGYM_STRICT") == "1"
    started = time.perf_counter()
    score_ok = score.main([] if strict else ["--allow-todo"]) == 0
    _phase("score table", started)
    ok = result.wasSuccessful() and score_ok  # the lint is a row of the score table: printed once
    if full:
        for label, arguments in EXTRA_TOOLS:
            ok = _extra_tool(label, arguments) and ok
    print(f"\ntotal {time.perf_counter() - began:.0f} s ({'full' if full else 'fast'} mode)")
    print("CHECK:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1
