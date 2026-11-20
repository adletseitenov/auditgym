"""Fails when the core package uses a source of nondeterminism or a non-stdlib import.

Two passes over auditgym/*.py: line patterns (imports of forbidden modules, clock calls, float())
and an AST pass for what a line pattern misses or an alias hides: `from time import ...`,
`from os import urandom`, `date.today()`, `time.time_ns()`, the builtin `hash()`,
`__import__("random")` / `importlib.import_module("random")`, and true division `/` with an int
literal operand whose other side is not a Decimal or Fraction expression (it would give a float).
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

CORE = pathlib.Path(__file__).resolve().parent.parent / "auditgym"

# Pattern -> why it is forbidden in the core package.
FORBIDDEN = {
    r"^\s*(import|from)\s+random\b": "use auditgym.rng.Rng",
    r"^\s*(import|from)\s+(uuid|secrets)\b": "use Rng.token",
    r"\bos\.urandom\b": "use auditgym.rng.Rng",
    r"\btime\.(time|time_ns|monotonic|monotonic_ns|perf_counter|perf_counter_ns|process_time)\b":
        "the clock must not reach episode data",
    r"\b(datetime|date)\.(now|today|utcnow)\b": "the clock must not reach episode data",
    r"\bfloat\(": "money and rates are int and Decimal",
    r"^\s*(import|from)\s+(numpy|pandas|gymnasium|requests|httpx|yaml)\b": "stdlib only",
}
# A line may opt out with a reason: `# determinism: <why this is safe>`.
ALLOW = re.compile(r"#\s*determinism:\s*\S")

RANDOM_MODULES = ("random", "uuid", "secrets")
CLOCK_ATTRIBUTES = {("date", "today"), ("datetime", "today"), ("datetime", "now"), ("datetime", "utcnow"),
                    ("time", "time"), ("time", "time_ns"), ("time", "monotonic"), ("time", "monotonic_ns"),
                    ("time", "perf_counter"), ("time", "perf_counter_ns"), ("time", "process_time")}
EXACT_TYPES = ("Decimal", "Fraction")


def _is_int_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


def _uses_exact_type(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in EXACT_TYPES:
                return True
    return False


def ast_problems(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, why) for every forbidden construct in a parsed module."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if module == "time":
                found.append((node.lineno, "from time import: the clock must not reach episode data"))
            elif module == "os" and "urandom" in names:
                found.append((node.lineno, "from os import urandom: use auditgym.rng.Rng"))
            elif module.split(".")[0] in RANDOM_MODULES:
                found.append((node.lineno, f"from {module} import: use auditgym.rng.Rng"))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "hash":
                found.append((node.lineno, "hash() depends on PYTHONHASHSEED"))
            imported = None
            if isinstance(func, ast.Name) and func.id == "__import__":
                imported = node.args[0] if node.args else None
            elif isinstance(func, ast.Attribute) and func.attr == "import_module":
                imported = node.args[0] if node.args else None
            if (isinstance(imported, ast.Constant) and isinstance(imported.value, str)
                    and imported.value.split(".")[0] in RANDOM_MODULES):
                found.append((node.lineno, f"dynamic import of {imported.value}: use auditgym.rng.Rng"))
        elif isinstance(node, ast.Attribute):
            owner = node.value.id if isinstance(node.value, ast.Name) else \
                node.value.attr if isinstance(node.value, ast.Attribute) else ""
            if (owner, node.attr) in CLOCK_ATTRIBUTES:  # also dt.date.today, datetime.datetime.now
                found.append((node.lineno, f"{owner}.{node.attr}: the clock must not reach episode data"))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            literal = _is_int_literal(node.left) or _is_int_literal(node.right)
            if literal and not (_uses_exact_type(node.left) or _uses_exact_type(node.right)):
                found.append((node.lineno, "true division of an int literal gives a float: use // or Decimal"))
    return found


def file_problems(path: pathlib.Path, label: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    problems = []
    for number, line in enumerate(lines, 1):
        if ALLOW.search(line):
            continue
        for pattern, why in FORBIDDEN.items():
            if re.search(pattern, line):
                problems.append(f"{label}:{number}: {why}: {line.strip()}")
    reported = {int(p.split(":")[1]) for p in problems}
    for number, why in sorted(set(ast_problems(ast.parse(text, filename=str(path))))):
        line = lines[number - 1] if 0 < number <= len(lines) else ""
        if ALLOW.search(line) or number in reported:
            continue
        problems.append(f"{label}:{number}: {why}: {line.strip()}")
    return problems


def main() -> int:
    problems = []
    for path in sorted(CORE.rglob("*.py")):
        problems += file_problems(path, str(path.relative_to(CORE.parent)))
    for problem in problems:
        print(problem)
    print(f"determinism lint: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
