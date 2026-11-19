"""Tests of the baseline policies.

`python -m auditgym check` discovers tests with tests/ as the top level, so this package is
imported under the name `policies` and would hide the real `policies/` package. Appending the
real directory to the search path makes `policies.runner` and friends resolve to the real code
in that case; under the normal name `tests.policies` this has no effect.
"""
import pathlib

_REAL = pathlib.Path(__file__).resolve().parent.parent.parent / "policies"
if __name__ == "policies" and str(_REAL) not in __path__:
    __path__.append(str(_REAL))
