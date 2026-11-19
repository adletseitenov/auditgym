"""Checks for the score criterion «Инженерия и объяснение» (CASE §10): one command, versions, journals,
measured failure analysis, honest limitations. Each returns None on success or a failure message.

CHECKS is a list of (name, function); tools/score.py wires the module into ZONE_CHECKS. The checks read
README.md, docs/ and the Dockerfile as text and run the entry point in a subprocess, so they fail when
the files promise something the repository does not do.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCKER_COMMAND = "docker build -t auditgym . && docker run --rm --network none auditgym"
PLAIN_COMMAND = "python3 -m auditgym check"
FULL_COMMAND = "python3 -m auditgym check --full"
# Topics the «Ограничения» section of the README must cover: (what it is, regex over the section).
LIMITATION_TOPICS = (
    ("synthetic, simplified bookkeeping", r"синтетическ"),
    ("seed recovery when the package is importable", r"перебор\w* seed|восстанов\w+ seed|seed\w* (?:можно )?восстанов"),
    ("mitigation: secret and long seeds", r"секрет"),
    ("a policy in the evaluator's process sees the seed", r"в том же процессе"),
    ("inspection_steps excludes the submit step", r"inspection_steps.{0,120}submit|submit.{0,120}inspection_steps"),
    ("silence scores -1 on books with errors", r"молчани\w+.{0,200}(?:-1|−1)"),
    ("flag-everything results", r"flag_all|пометить всё"),
    ("what the reward does not prove", r"не доказывает"),
)
MEASURED_BEGIN, MEASURED_END = "<!-- measured:begin -->", "<!-- measured:end -->"
RESULTS_BEGIN, RESULTS_END = "<!-- results:begin -->", "<!-- results:end -->"
RESULTS_SOURCE, ATTACK_SOURCE = "docs/RESULTS.md", "python3 -m tools.attack"
# Environment of the offline smoke: a dead proxy, and a sitecustomize that closes the network for every
# process started from it (the evaluator process included).
_BLOCK_NETWORK = '''import socket
def _closed(*args, **kwargs):
    raise OSError("network is closed in this check")
socket.socket.connect = _closed
socket.socket.connect_ex = _closed
socket.getaddrinfo = _closed
'''
_SMOKE = '''import json
from auditgym import Env
from policies.runner import make_policy, run_episode
with Env(mode="train") as env:
    result = run_episode(env, make_policy("solver_policy"), 5, None)
print(json.dumps({"reward": result["reward"], "terminated": result["terminated"], "steps": result["steps"]}))
'''


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> Optional[str]:
    """Body of the `## <heading>` section (up to the next `## `), or None."""
    match = re.search(rf"^## {re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return match.group(1) if match else None


def one_command_runs_offline() -> Optional[str]:
    readme = _read("README.md")
    for command in (DOCKER_COMMAND, PLAIN_COMMAND, FULL_COMMAND):
        if command not in readme:
            return f"README does not give the command: {command}"
    dockerfile = _read("Dockerfile")
    lines = [line.strip() for line in dockerfile.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not any(re.fullmatch(r"FROM python:3\.11-slim@sha256:[0-9a-f]{64}", line) for line in lines):
        return "Dockerfile does not pin python:3.11-slim by digest"
    if not any(re.fullmatch(r'CMD \["python", "-m", "auditgym", "check"\]', line) for line in lines):
        return "Dockerfile CMD is not `python -m auditgym check`"
    for line in lines:
        if re.search(r"\b(pip|apt-get|apk|curl|wget|npm|git clone)\b", line):
            return f"Dockerfile fetches something from the network: {line}"
    if "dependencies = []" not in _read("pyproject.toml"):
        return "pyproject.toml declares dependencies"
    # The entry point, in a clean directory with the network closed: help works, and one whole
    # episode (generator, evaluator process, policy, judge) runs and gets a reward.
    with tempfile.TemporaryDirectory() as directory:
        pathlib.Path(directory, "sitecustomize.py").write_text(_BLOCK_NETWORK, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{directory}:{ROOT}", "PYTHONDONTWRITEBYTECODE": "1",
               "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": ""}
        try:
            helped = subprocess.run([sys.executable, "-m", "auditgym", "--help"], cwd=directory, env=env,
                                    capture_output=True, text=True, timeout=30)
            ran = subprocess.run([sys.executable, "-c", _SMOKE], cwd=directory, env=env,
                                 capture_output=True, text=True, timeout=120)
            blocked = subprocess.run([sys.executable, "-c", "import socket; socket.create_connection(('127.0.0.1', 9))"],
                                     cwd=directory, env=env, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired as error:
            return f"offline smoke timed out: {error.cmd[-1][:40]!r}"
    if blocked.returncode == 0:
        return "the network was not closed in the smoke environment"
    if helped.returncode != 0 or "check --full" not in helped.stdout:
        return "`python -m auditgym --help` does not describe `check --full`"
    if ran.returncode != 0:
        return f"offline episode failed: {ran.stderr.strip().splitlines()[-1] if ran.stderr.strip() else ran.returncode}"
    result = json.loads(ran.stdout.strip().splitlines()[-1])
    if not result["terminated"] or result["reward"] <= 0.9:
        return f"offline episode gave reward {result['reward']}"
    return None


def versions_are_in_every_episode() -> Optional[str]:
    from auditgym import SCHEMA_VERSION, __version__
    from auditgym.config import CONFIG_VERSION

    directories = sorted((ROOT / "episodes").glob("*/ep_*"))
    if len(directories) < 20:
        return f"{len(directories)} shipped episodes, 20 expected"
    for directory in directories:
        name = f"{directory.parent.name}/{directory.name}"
        manifest = json.loads((directory / "public" / "manifest.json").read_text(encoding="utf-8"))
        truth = json.loads((directory / "private" / "truth.json").read_text(encoding="utf-8"))
        provenance = json.loads((directory / "private" / "provenance.json").read_text(encoding="utf-8"))
        found = {"manifest.schema_version": (manifest.get("schema_version"), SCHEMA_VERSION),
                 "manifest.config_version": (manifest.get("config_version"), CONFIG_VERSION),
                 "truth.schema_version": (truth.get("schema_version"), SCHEMA_VERSION),
                 "provenance.generator_version": (provenance.get("generator_version"), __version__),
                 "provenance.config.config_version": ((provenance.get("config") or {}).get("config_version"),
                                                      CONFIG_VERSION)}
        for key, (actual, expected) in found.items():
            if actual != expected:
                return f"{name}: {key} is {actual!r}, current is {expected!r}"
    plan = json.loads((ROOT / "configs" / "episodes.json").read_text(encoding="utf-8"))
    if plan.get("config_version") != CONFIG_VERSION:
        return "configs/episodes.json does not record the current config_version"
    return None


def measured_failure_analysis() -> Optional[str]:
    """DESIGN.md has a table between the measured markers; each row quotes a value in backticks and names its
    source. A value from docs/RESULTS.md must appear in that file verbatim (it is written by tools/report.py),
    one from the attack table is checked by running the tool."""
    design = _read("docs/DESIGN.md")
    if MEASURED_BEGIN not in design or MEASURED_END not in design:
        return "docs/DESIGN.md has no measured failure analysis (markers missing)"
    block = design.split(MEASURED_BEGIN, 1)[1].split(MEASURED_END, 1)[0]
    rows = [[cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            for line in block.splitlines() if line.strip().startswith("|") and "---" not in line]
    rows = [row for row in rows if len(row) == 3 and row[1] != "Источник"]
    if len(rows) < 8:
        return f"{len(rows)} measured rows in docs/DESIGN.md, at least 8 needed"
    results = _read("docs/RESULTS.md")
    attack_output = None
    sources = set()
    for what, source, value in rows:
        quoted = re.fullmatch(r"`([^`]+)`", value)
        if quoted is None or not re.search(r"[0-9]", value):
            return f"measured row {what!r}: the value must be a number in backticks"
        value = quoted.group(1)
        sources.add(source)
        if source == RESULTS_SOURCE:
            if value not in results:
                return f"measured row {what!r}: `{value}` is not in docs/RESULTS.md (re-run `make report`?)"
        elif source == ATTACK_SOURCE:
            if attack_output is None:
                done = subprocess.run([sys.executable, "-m", "tools.attack"], cwd=ROOT, capture_output=True,
                                      text=True, timeout=120)
                attack_output = done.stdout
            if value not in attack_output:
                return f"measured row {what!r}: `{value}` is not printed by tools.attack"
        else:
            return f"measured row {what!r}: unknown source {source!r}"
    if sources != {RESULTS_SOURCE, ATTACK_SOURCE}:
        return f"measured rows must cite both {RESULTS_SOURCE} and {ATTACK_SOURCE}, got {sorted(sources)}"
    if not re.search(r"^## (?:[0-9]+\. )?Измеренный разбор сбоев", design, re.M):
        return "docs/DESIGN.md has no «Измеренный разбор сбоев» section"
    return readme_results_are_a_copy(results)


def readme_results_are_a_copy(results: str) -> Optional[str]:
    """The table in the README (between the results markers) is a verbatim copy of rows of docs/RESULTS.md."""
    readme = _read("README.md")
    if RESULTS_BEGIN not in readme or RESULTS_END not in readme:
        return "README has no results table (markers missing)"
    rows = [line for line in readme.split(RESULTS_BEGIN, 1)[1].split(RESULTS_END, 1)[0].splitlines()
            if line.startswith("|")]
    if len(rows) < 10:
        return f"README results table has {len(rows)} lines, at least 10 expected"
    available = set(results.splitlines())
    for row in rows:
        if row not in available:
            return f"README results row is not in docs/RESULTS.md: {row[:60]}"
    return None


def limitations_are_stated() -> Optional[str]:
    section = _section(_read("README.md"), "Ограничения")
    if section is None:
        return "README has no «Ограничения» section"
    for topic, pattern in LIMITATION_TOPICS:
        if not re.search(pattern, section, re.S | re.I):
            return f"README «Ограничения» does not cover: {topic}"
    return None


CHECKS = [
    ("one command runs everything offline in the container", one_command_runs_offline),
    ("config and schema versions are recorded in every episode", versions_are_in_every_episode),
    ("measured failure analysis is in docs/DESIGN.md with numbers from docs/RESULTS.md", measured_failure_analysis),
    ("limitations are stated in the README", limitations_are_stated),
]
