"""Metamorphic checks of reproducibility and generalization (CASE §04, §06, §10: 20 paired checks).

    python -m tools.metamorph --pairs N [--seed-base S] [--verbose]

For N generated episodes over varied configurations (the supported set is probed at run time with
try/except around generate, so the tool widens as soon as new error types and interactions land):

  repeat     two fresh server processes, same seed/config/action script, different PYTHONHASHSEED,
             TZ, LC_ALL and working directory: byte-identical protocol transcripts and results files,
             equal to the in-process evaluator
  isolation  one server process: A, B, A again; the second A is byte-identical to the first
  surface    surface_seed change: the oracle answer mapped through provenance id_map gets the same
             reward, and equals the oracle of the new episode
  transform  episode written to disk, `python -m tools.transform` (reorder, rename, id style cycles),
             loaded with Env.from_files: the oracle answer rewritten with mapping.json gets the same
             reward; the honest solver policy (pages every table through the contract, runs
             auditgym.solver.solve, submits) gets the same reward and the same findings up to the
             mapping; cursors give every record of the transformed files exactly once (in the
             evaluator's id-derived page order)

The oracle answer is built here from the episode's private truth (first evidence set of each
issue), never through the agent-facing API. Exits 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLES = ("ledger", "invoices", "payments", "fx")
ID_STYLES = ("token", "numeric", "hex", "prefixed")
CHECK_NAMES = ("repeat", "isolation", "surface", "transform")

# Candidate configurations; unsupported ones are dropped at run time (supported_configs).
CANDIDATES = [
    ("clean", {"errors": {}}),
    ("clean-multi", {"errors": {}, "currencies": ["USD", "EUR"], "timezone": "-11:00"}),
    ("dup", {"errors": {"duplicate_invoice": 1}}),
    ("cutoff", {"errors": {"period_cutoff": 1}, "timezone": "+14:00"}),
    ("dup+cutoff", {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}, "currencies": ["USD", "EUR"]}),
    ("dense", {"errors": {"duplicate_invoice": 2, "period_cutoff": 2}, "currencies": ["CNY", "RUB"],
               "timezone": "+05:30"}),
    ("gap", {"errors": {"allocation_gap": 1}}),
    ("fx", {"errors": {"fx_error": 1}, "currencies": ["USD", "EUR"]}),
    ("all-four", {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}}),
    ("I1", {"errors": {}, "interaction": "I1"}),
    ("I2", {"errors": {}, "interaction": "I2"}),
    ("I1+all", {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
                "interaction": "I1", "currencies": ["USD", "EUR"]}),
    ("I2+all", {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
                "interaction": "I2"}),
]
ENVIRONMENTS = (("0", "UTC", "C"), ("random", "Asia/Tokyo", "en_US.UTF-8"), ("4242", "America/Los_Angeles", "C.UTF-8"))


def supported_configs() -> list[tuple[str, dict]]:
    from auditgym.config import ConfigError
    from auditgym.generate import generate

    out = []
    for name, config in CANDIDATES:
        try:
            generate(1, config)
        except ConfigError:
            continue
        out.append((name, config))
    return out


# --- shared helpers (also used by tools.leakscan, tools.trajectories, tools.checks_env) -------------
def oracle_answer(episode: dict) -> dict:
    """The evaluator-side oracle: every truth issue with its first evidence set."""
    truth = episode["private"]["truth"]
    findings = [{"type": issue["type"], "affected_ids": sorted(issue["affected_ids"]),
                 "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
                 "evidence_ids": list(issue["evidence_sets"][0]), "explanation": "oracle"}
                for issue in sorted(truth["issues"], key=lambda issue: issue["issue_id"])]
    return {"episode_id": episode["public"]["manifest"]["episode_id"], "findings": findings}


def map_answer(answer: dict, mapping: dict[str, str]) -> dict:
    def m(value):
        return mapping.get(value, value)

    return {"episode_id": m(answer["episode_id"]),
            "findings": [dict(finding, affected_ids=[m(i) for i in finding["affected_ids"]],
                              evidence_ids=[m(i) for i in finding["evidence_ids"]])
                         for finding in answer["findings"]]}


def finding_keys(findings: list, mapping: dict[str, str] | None = None) -> list[tuple]:
    def m(value):
        return mapping.get(value, value) if mapping else value

    return sorted((f["type"], tuple(sorted(m(i) for i in f["affected_ids"])), f["claimed_amount_minor"],
                   f["currency"], tuple(sorted(m(i) for i in f["evidence_ids"]))) for f in findings)


def surface_mapping(first: dict, second: dict) -> dict[str, str]:
    """Public id of `first` -> public id of `second` through the internal keys of provenance id_map."""
    to_internal = {public: key for key, public in first["private"]["provenance"]["id_map"].items()}
    target = second["private"]["provenance"]["id_map"]
    mapping = {public: target[key] for public, key in to_internal.items() if key in target}
    mapping[first["public"]["manifest"]["episode_id"]] = second["public"]["manifest"]["episode_id"]
    return mapping


def page_all(step) -> tuple[dict, list, int, str | None]:
    """Read every table through list_records. Returns (tables, transcript, steps, problem)."""
    tables, transcript, steps = {}, [], 0
    for table in TABLES:
        records, cursor = [], None
        while True:
            result = step({"action": "list_records", "table": table, "cursor": cursor})
            transcript.append(result)
            steps += 1
            observation = result[0]
            if observation.get("type") != "records":
                return tables, transcript, steps, f"{table}: {observation.get('error')}"
            records.extend(observation["records"])
            cursor = observation["next_cursor"]
            if cursor is None:
                break
        tables[table] = records
    return tables, transcript, steps, None


def solver_public(first_observation: dict, tables: dict) -> dict:
    return {"manifest": first_observation["manifest"], "policy": first_observation["policy"], **tables}


def solver_answer(first_observation: dict, tables: dict) -> dict:
    from auditgym.solver import solve

    return {"episode_id": first_observation["manifest"]["episode_id"],
            "findings": solve(solver_public(first_observation, tables))}


# --- raw server processes -----------------------------------------------------------------------
class _Clock:
    def now(self):
        return 0


def _serve(requests: list[bytes], *, mode: str, results, hashseed: str, tz: str, locale: str, cwd) -> bytes:
    environment = dict(os.environ, PYTHONHASHSEED=hashseed, TZ=tz, LC_ALL=locale, LANG=locale, PYTHONPATH=str(ROOT))
    command = [sys.executable, "-m", "auditgym", "serve", "--mode", mode]
    if results is not None:
        command += ["--results", str(results)]
    result = subprocess.run(command, input=b"".join(requests), capture_output=True, env=environment,
                            cwd=str(cwd), timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"server exited with {result.returncode}")
    return result.stdout


def _line(message: dict) -> bytes:
    return (json.dumps(message, ensure_ascii=True, sort_keys=True) + "\n").encode("ascii")


def build_script(seed: int, config: dict) -> list[dict]:
    """An honest action script for one episode, built with the in-process evaluator (no clock)."""
    from auditgym.evaluator import Evaluator

    evaluator = Evaluator(mode="train", clock=_Clock())
    first, _ = evaluator.reset(seed, config)
    script = []

    def step(action):
        script.append(action)
        return evaluator.step(action)

    tables, _, _, problem = page_all(step)
    if problem:
        raise RuntimeError(problem)
    extra = [{"action": "get_record", "table": "ledger", "id": tables["ledger"][0]["row_id"]},
             {"action": "calculator", "expression": "125000 * 465.37 / 3"}, {"action": "nope"},
             {"action": "get_record", "table": "invoices", "id": "missing"}]
    if tables["invoices"]:
        extra.append({"action": "get_document", "doc_id": tables["invoices"][0]["doc_id"]})
    if tables["fx"]:
        row = tables["fx"][0]
        extra.append({"action": "get_record", "table": "fx",
                      "id": f"{row['date']}|{row['source_currency']}|{row['target_currency']}"})
    for action in extra:
        step(action)
    step({"action": "submit", "findings": solver_answer(first, tables)})
    step({"action": "list_records", "table": "ledger"})
    return script


def _requests(seed: int, config: dict, script: list[dict]) -> list[bytes]:
    return [_line({"op": "reset", "seed": seed, "config": config})] + [_line({"op": "step", "action": a}) for a in script]


# --- the four checks ------------------------------------------------------------------------------
def check_repeat(seed: int, config: dict, script: list[dict], workdir: pathlib.Path) -> str | None:
    from auditgym.evaluator import Evaluator
    from auditgym.serve import Server

    requests = _requests(seed, config, script)
    outputs, results = [], []
    for number, (hashseed, tz, locale) in enumerate(ENVIRONMENTS[:2]):
        path = workdir / f"results{number}.jsonl"
        cwd = ROOT if number == 0 else workdir
        outputs.append(_serve(requests, mode="eval", results=path, hashseed=hashseed, tz=tz, locale=locale, cwd=cwd))
        results.append(path.read_bytes())
    if outputs[0] != outputs[1]:
        return "two server processes gave different transcripts"
    if results[0] != results[1] or len(results[0].splitlines()) != 1:
        return "two server processes wrote different results"
    local_path = workdir / "results-local.jsonl"
    server = Server(Evaluator(mode="eval", results_path=local_path, clock=_Clock()))
    from auditgym.serve import _dump

    local = b"".join((_dump(server.handle(request)) + "\n").encode("ascii") for request in requests)
    if local != outputs[0]:
        return "server process differs from the in-process evaluator"
    if local_path.read_bytes() != results[0]:
        return "results of the server process differ from the in-process evaluator"
    return None


def check_isolation(seed: int, config: dict, script: list[dict], other_seed: int, other_config: dict,
                    other_script: list[dict], workdir: pathlib.Path) -> str | None:
    a = _requests(seed, config, script)
    b = _requests(other_seed, other_config, other_script)
    hashseed, tz, locale = ENVIRONMENTS[2]
    output = _serve(a + b + a, mode="train", results=None, hashseed=hashseed, tz=tz, locale=locale, cwd=workdir)
    lines = output.splitlines(keepends=True)
    if len(lines) != 2 * len(a) + len(b):
        return "server did not answer every request"
    first, second = lines[:len(a)], lines[len(a) + len(b):]
    if first != second:
        return "A after B differs from the first A"
    if lines[len(a):len(a) + len(b)] == first[:len(b)]:
        return "B gave the same transcript as A"
    return None


def _submit_reward(env, answer: dict) -> float:
    _, reward, terminated, _, info = env.step({"action": "submit", "findings": answer})
    if not terminated:
        raise RuntimeError("submit did not terminate")
    return reward


def check_surface(seed: int, config: dict, surface_seed: int) -> str | None:
    from auditgym import Env
    from auditgym.generate import generate
    from auditgym.solver import solve

    first = generate(seed, config)
    second = generate(seed, dict(config, surface_seed=surface_seed))
    if first["public"]["ledger"] and first["public"]["ledger"][0]["row_id"] == second["public"]["ledger"][0]["row_id"]:
        return "surface_seed did not change the ids"
    mapping = surface_mapping(first, second)
    mapped = map_answer(oracle_answer(first), mapping)
    own = oracle_answer(second)
    if [k[:4] for k in finding_keys(mapped["findings"])] != [k[:4] for k in finding_keys(own["findings"])]:
        return "mapped oracle differs from the oracle of the new surface"
    truth = {(i["type"], frozenset(i["affected_ids"])): [set(s) for s in i["evidence_sets"]]
             for i in second["private"]["truth"]["issues"]}
    for finding in mapped["findings"]:
        if set(finding["evidence_ids"]) not in truth.get((finding["type"], frozenset(finding["affected_ids"])), []):
            return "mapped evidence is not an evidence set of the new surface"
    if finding_keys(solve(first["public"]), mapping) != finding_keys(solve(second["public"])):
        return "solver findings differ under the surface renaming"
    rewards = []
    with Env() as env:
        env.reset(seed, config)
        rewards.append(_submit_reward(env, oracle_answer(first)))
        env.reset(seed, dict(config, surface_seed=surface_seed))
        rewards.append(_submit_reward(env, mapped))
    if rewards[0] != rewards[1]:
        return f"oracle reward {rewards[0]} became {rewards[1]}"
    if rewards[0] != 1.0:
        return f"oracle reward is {rewards[0]}, not 1.0"
    return None


def _files_view(directory) -> dict:
    """The tables of an episode directory as the evaluator serves them: the projection of the public
    files, in the evaluator's id-derived page order (not the order of the file rows)."""
    from auditgym.evaluator import _FIELDS, _project, order_key
    from auditgym.storage import load_episode

    public = load_episode(directory)["public"]
    return {table: sorted((_project(record, fields) for record in public[table]),
                          key=lambda record, table=table: order_key(table, record))
            for table, fields in _FIELDS.items()}


def _honest_run(env) -> tuple[float, list, str | None, dict]:
    first, _ = env.reset()
    tables, _, _, problem = page_all(env.step)
    if problem:
        return 0.0, [], problem, tables
    answer = solver_answer(first, tables)
    return _submit_reward(env, answer), answer["findings"], None, tables


def check_transform(seed: int, config: dict, number: int, workdir: pathlib.Path) -> str | None:
    from auditgym import Env
    from auditgym.generate import generate
    from auditgym.storage import load_episode, write_episode
    from tools.transform import load_mapping

    episode = generate(seed, config)
    src, dst = workdir / "src", workdir / "dst"
    write_episode(episode, src)
    command = [sys.executable, "-m", "tools.transform", str(src), str(dst), "--seed", str(seed * 31 + number),
               "--id-style", ID_STYLES[number % len(ID_STYLES)]]
    if number % 2:
        command.append("--unsorted-truth")
    if number % 3 == 2:
        command.append("--keep-episode-id")
    result = subprocess.run(command, capture_output=True, cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT)),
                            timeout=120)
    if result.returncode != 0:
        return f"transform failed: {result.stderr.decode(errors='replace').strip()}"
    mapping = load_mapping(dst)
    transformed = load_episode(dst)
    if transformed["public"]["ledger"] == episode["public"]["ledger"]:
        return "transform changed nothing"

    mapped = map_answer(oracle_answer(episode), mapping)
    own = oracle_answer(transformed)
    if [k[:4] for k in finding_keys(mapped["findings"])] != [k[:4] for k in finding_keys(own["findings"])]:
        return "mapped oracle differs from the oracle of the transformed truth"

    outcomes = {}
    for name, directory, answer in (("src", src, oracle_answer(episode)), ("dst", dst, mapped)):
        with Env.from_files(directory) as env:
            env.reset()
            oracle_reward = _submit_reward(env, answer)
            solver_reward, findings, problem, tables = _honest_run(env)
        if problem:
            return f"{name}: paging failed: {problem}"
        outcomes[name] = (oracle_reward, solver_reward, findings, tables)
    if outcomes["src"][0] != outcomes["dst"][0] or outcomes["src"][0] != 1.0:
        return f"oracle reward {outcomes['src'][0]} became {outcomes['dst'][0]}"
    if outcomes["src"][1] != outcomes["dst"][1]:
        return f"solver reward {outcomes['src'][1]} became {outcomes['dst'][1]}"
    if finding_keys(outcomes["src"][2], mapping) != finding_keys(outcomes["dst"][2]):
        return "solver findings differ up to the mapping"
    expected = _files_view(dst)
    for table in TABLES:
        got = outcomes["dst"][3][table]
        if json.dumps(got, sort_keys=True) != json.dumps(expected[table], sort_keys=True):
            return f"{table}: cursors do not give the transformed file records exactly once, in page order"
    counts = transformed["public"]["manifest"]["counts"]
    if any(counts[table] != len(outcomes["dst"][3][table]) for table in TABLES):
        return "paged record counts differ from manifest counts"
    return None


# --- driver -----------------------------------------------------------------------------------------
def run(pairs: int, seed_base: int = 1_000_003, verbose: bool = False, out=sys.stdout) -> dict:
    configs = supported_configs()
    results = {name: [0, 0] for name in CHECK_NAMES}
    by_config: dict[str, list[int]] = {}
    failures = []
    for number in range(pairs):
        label, config = configs[number % len(configs)]
        seed = seed_base + 7919 * number
        other_label, other_config = configs[(number + 1) % len(configs)]
        other_seed = seed + 1
        row = by_config.setdefault(label, [0, 0])
        with tempfile.TemporaryDirectory(prefix="metamorph_") as directory:
            workdir = pathlib.Path(directory)
            outcome = {}
            try:
                script = build_script(seed, config)
                other_script = build_script(other_seed, other_config)
            except Exception as exc:  # a crash is a failed pair
                script = other_script = None
                for name in CHECK_NAMES:
                    outcome[name] = f"script: {type(exc).__name__}: {exc}"
            checks = {
                "repeat": lambda: check_repeat(seed, config, script, workdir),
                "isolation": lambda: check_isolation(seed, config, script, other_seed, other_config,
                                                     other_script, workdir),
                "surface": lambda: check_surface(seed, config, seed * 2 + 11),
                "transform": lambda: check_transform(seed, config, number, workdir),
            }
            for name, function in checks.items():
                if name in outcome:
                    continue
                try:
                    outcome[name] = function()
                except Exception as exc:
                    outcome[name] = f"{type(exc).__name__}: {exc}"
        for name in CHECK_NAMES:
            results[name][1] += 1
            if outcome[name] is None:
                results[name][0] += 1
            else:
                failures.append(f"pair {number} ({label}, seed {seed}) {name}: {outcome[name]}")
        row[1] += 1
        row[0] += all(outcome[name] is None for name in CHECK_NAMES)
        if verbose:
            print(f"pair {number:3d} {label:<12} " + " ".join(
                f"{name}={'ok' if outcome[name] is None else 'FAIL'}" for name in CHECK_NAMES), file=out)
    print(f"metamorph: {pairs} pairs over {len(configs)} supported configs "
          f"({', '.join(label for label, _ in configs)})", file=out)
    print(f"{'check':<12}{'passed':>10}", file=out)
    for name in CHECK_NAMES:
        passed, total = results[name]
        print(f"{name:<12}{passed:>6}/{total:<4}", file=out)
    print(f"{'config':<12}{'pairs ok':>10}", file=out)
    for label, (passed, total) in by_config.items():
        print(f"{label:<12}{passed:>6}/{total:<4}", file=out)
    for failure in failures:
        print("FAIL " + failure, file=out)
    return {"results": results, "by_config": by_config, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.metamorph")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1_000_003)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    summary = run(arguments.pairs, arguments.seed_base, arguments.verbose)
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
