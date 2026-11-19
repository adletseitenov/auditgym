"""Trajectory logs (CASE §09: trajectories.jsonl with episode_id, step, action, observation, reward,
terminated, truncated; «В наблюдениях обучающих траекторий нет закрытого эталона»).

    python -m tools.trajectories [--out trajectories.jsonl] [--episodes DIR | --seeds 1 2 3]
                                 [--config JSON] [--policies random,solver,oracle]
                                 [--policy module:Class ...] [--oracle-out PATH] [--mode train|eval]

With no arguments it records the `random` and `solver` policies on the shipped train episodes
(`episodes/train`) into `trajectories.jsonl` (`make trajectories`). The output directory is
created when it does not exist.

Each line has exactly the seven fields, in that order. Step 0 is the reset: action null, reward
0.0, both flags false. `info` is never written, nor anything else.

For other engineers' policies: wrap the Env in a Recorder and drive it as usual.

    with Env() as env, TrajectoryWriter("trajectories.jsonl") as writer:
        recorder = Recorder(env, writer, policy)
        observation, info = recorder.reset(seed, config)
        observation, reward, terminated, truncated, info = recorder.step(action)

or call `run_policy(recorder, policy, seed, config)` for a policy with `act(observation) -> action`.

Oracle separation. A policy that reads the truth is privileged: class attribute
`privileged = True`, or "oracle" in its class name. Its trajectories contain truth in `action`, so
a privileged policy can only be recorded by a writer opened with `privileged=True`, whose file name
must contain ".oracle." (default: <out stem>.oracle.jsonl); a training writer refuses it and refuses
a file name with "oracle" in it. `scan_file` checks a written file: field set, step order,
observation shapes, no truth ids outside the records the agent received and its own actions, no
issue ids or internal keys, no seed (seeds >= 1000; smaller ones are indistinguishable from step
counters), no paths or private words; in a training file, every id the agent submits must have been
shown to it earlier in the episode (an id it never saw came from the truth).
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import re
import sys

from auditgym.rng import Rng

FIELDS = ("episode_id", "step", "action", "observation", "reward", "terminated", "truncated")
TABLES = ("ledger", "invoices", "payments", "fx")
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EPISODES = ROOT / "episodes" / "train"
DEFAULT_OUT = "trajectories.jsonl"


class TrajectoryError(Exception):
    pass


def is_privileged(policy) -> bool:
    return bool(getattr(policy, "privileged", False)) or "oracle" in type(policy).__name__.lower()


class TrajectoryWriter:
    def __init__(self, path, privileged: bool = False, mode: str = "w") -> None:
        name = pathlib.Path(path).name.lower()
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        if privileged and ".oracle." not in name:
            raise TrajectoryError("oracle trajectories go to a file named *.oracle.jsonl")
        if not privileged and "oracle" in name:
            raise TrajectoryError("a training trajectory file must not be an oracle file")
        self.privileged = privileged
        self.lines = 0
        self._handle = open(path, mode, encoding="utf-8")

    def write(self, line: dict) -> None:
        if tuple(line) != FIELDS:
            raise TrajectoryError(f"trajectory line must have exactly {FIELDS}")
        self._handle.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
        self.lines += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class Recorder:
    """An Env with the same reset/step API that writes every transition (never `info`)."""

    def __init__(self, env, writer: TrajectoryWriter, policy=None) -> None:
        if policy is not None and is_privileged(policy) and not writer.privileged:
            raise TrajectoryError("a privileged policy must write to a separate oracle file")
        self.env = env
        self.writer = writer
        self.episode_id = None
        self.step_number = 0

    def _write(self, action, observation, reward, terminated, truncated) -> None:
        self.writer.write({"episode_id": self.episode_id, "step": self.step_number, "action": action,
                           "observation": observation, "reward": reward, "terminated": terminated,
                           "truncated": truncated})

    def reset(self, seed=None, config=None):
        observation, info = self.env.reset(seed, config) if seed is not None else self.env.reset()
        self.episode_id = observation["manifest"]["episode_id"]
        self.step_number = 0
        self._write(None, observation, 0.0, False, False)
        return observation, info

    def step(self, action):
        if self.episode_id is None:
            raise TrajectoryError("reset first")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_number += 1
        self._write(action, observation, reward, terminated, truncated)
        return observation, reward, terminated, truncated, info


def run_policy(recorder: Recorder, policy, seed=None, config=None, max_steps: int = 100) -> float:
    observation, _ = recorder.reset(seed, config)
    reward = 0.0
    for _ in range(max_steps + 1):
        observation, reward, terminated, truncated, _ = recorder.step(policy.act(observation))
        if terminated or truncated:
            break
    return reward


# --- built-in policies ----------------------------------------------------------------------------
class RandomPolicy:
    """Random explorer with a fixed seed: reads pages and records it has seen, then submits a guess."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def act(self, observation: dict) -> dict:
        if observation.get("type") == "reset":
            self.rng = Rng(self.seed, "random-policy:" + observation["manifest"]["episode_id"])
            self.episode_id = observation["manifest"]["episode_id"]
            self.budget = self.rng.randint(5, 40)
            self.cursors = {table: None for table in TABLES}
            self.seen_docs, self.seen_rows = [], []
        elif observation.get("type") == "records":
            self.cursors[observation["table"]] = observation["next_cursor"]
            for record in observation["records"]:
                if "doc_id" in record:
                    self.seen_docs.append(record["doc_id"])
                elif "row_id" in record:
                    self.seen_rows.append(record["row_id"])
        self.budget -= 1
        if self.budget <= 0:
            findings = []
            if len(self.seen_docs) >= 2 and self.rng.chance(1, 2):
                pair = sorted(self.rng.sample(self.seen_docs, 2))
                findings.append({"type": "duplicate_invoice", "affected_ids": pair, "claimed_amount_minor": 0,
                                 "currency": "KZT", "evidence_ids": pair, "explanation": "random guess"})
            return {"action": "submit", "findings": {"episode_id": self.episode_id, "findings": findings}}
        choice = self.rng.below(4)
        if choice == 1 and self.seen_docs:
            return {"action": "get_document", "doc_id": self.rng.choice(self.seen_docs)}
        if choice == 2 and self.seen_rows:
            return {"action": "get_record", "table": "ledger", "id": self.rng.choice(self.seen_rows)}
        if choice == 3:
            return {"action": "calculator", "expression": f"{self.rng.below(10 ** 6)} * 1.5"}
        table = self.rng.choice(TABLES)
        return {"action": "list_records", "table": table, "cursor": self.cursors[table]}


class SolverPolicy:
    """Honest scripted policy: page every table through the contract, run auditgym.solver.solve, submit."""

    def act(self, observation: dict) -> dict:
        if observation.get("type") == "reset":
            self.first = observation
            self.tables = {table: [] for table in TABLES}
            self.table = 0
            return {"action": "list_records", "table": TABLES[0], "cursor": None}
        if observation.get("type") == "records":
            self.tables[observation["table"]].extend(observation["records"])
            if observation["next_cursor"] is not None:
                return {"action": "list_records", "table": observation["table"], "cursor": observation["next_cursor"]}
            self.table += 1
            if self.table < len(TABLES):
                return {"action": "list_records", "table": TABLES[self.table], "cursor": None}
        from auditgym.solver import solve

        public = {"manifest": self.first["manifest"], "policy": self.first["policy"], **self.tables}
        return {"action": "submit", "findings": {"episode_id": self.first["manifest"]["episode_id"],
                                                 "findings": solve(public)}}


class OracleSubmitPolicy:
    """Evaluator-side oracle (privileged): submits the truth with its first evidence sets at once."""

    privileged = True

    def __init__(self, answer: dict) -> None:
        self.answer = answer

    def act(self, observation: dict) -> dict:
        return {"action": "submit", "findings": self.answer}


# --- scan -------------------------------------------------------------------------------------------
def _ids_in(value, out: set) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _ids_in(item, out)
    elif isinstance(value, list):
        for item in value:
            _ids_in(item, out)
    elif isinstance(value, str):
        out.add(value)


def _fx_keys_in(value, out: set) -> None:
    """An fx row the agent was shown also shows its evidence key "date|source|target" (docs/AUDIT.md D1)."""
    if isinstance(value, dict):
        if all(isinstance(value.get(k), str) for k in ("date", "source_currency", "target_currency")):
            out.add(f"{value['date']}|{value['source_currency']}|{value['target_currency']}")
        for item in value.values():
            _fx_keys_in(item, out)
    elif isinstance(value, list):
        for item in value:
            _fx_keys_in(item, out)


def _without_records(observation: dict) -> dict:
    return {key: value for key, value in observation.items() if key not in ("records", "record")}


def _submitted_ids(action) -> set:
    from auditgym.actions import parse_action, ActionError

    try:
        name, arguments = parse_action(action)
    except ActionError:
        return set()
    if name != "submit" or not isinstance(arguments.get("findings"), dict):
        return set()
    ids = set()
    for finding in arguments["findings"].get("findings") or []:
        if isinstance(finding, dict):
            for field in ("affected_ids", "evidence_ids"):
                if isinstance(finding.get(field), list):
                    ids.update(i for i in finding[field] if isinstance(i, str))
    return ids


def scan_file(path, episodes: dict, paths: tuple = (), training: bool = True) -> list[str]:
    """Problems in a trajectory file. `episodes`: episode_id -> episode (with private) for the truth."""
    from tools.leakscan import FORBIDDEN_KEYS, OBS_KEYS, _keys

    problems = []
    words = ("truth", "provenance", "pre_ledger", "pre_invoices", "pre_payments", "id_map", "evidence_sets",
             "expected_amount", "issue_id", "generator_version", "surface_seed", "traceback")
    expected_step: dict[str, int] = {}
    seen: dict[str, set] = {}
    for number, raw in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        where = f"line {number}"
        try:
            line = json.loads(raw)
        except ValueError:
            problems.append(f"{where}: not JSON")
            continue
        if not isinstance(line, dict) or tuple(line) != FIELDS:
            problems.append(f"{where}: fields {list(line) if isinstance(line, dict) else line}")
            continue
        episode_id, step, observation = line["episode_id"], line["step"], line["observation"]
        if step != expected_step.get(episode_id, 0) and step != 0:
            problems.append(f"{where}: step {step} out of order")
        expected_step[episode_id] = step + 1
        if step == 0:
            seen[episode_id] = set()
        if not isinstance(observation, dict) or set(observation) != OBS_KEYS.get(observation.get("type"), set()):
            problems.append(f"{where}: observation shape")
            continue
        episode = episodes.get(episode_id)
        lowered = raw.lower()
        for word in words:
            if word in lowered:
                problems.append(f"{where}: word {word!r}")
        for item in paths + (str(ROOT),):
            if item and item in raw:
                problems.append(f"{where}: a path")
        keys = set()
        _keys(observation, keys)
        if keys & FORBIDDEN_KEYS:
            problems.append(f"{where}: key {sorted(keys & FORBIDDEN_KEYS)[0]!r}")
        if episode is None:
            problems.append(f"{where}: unknown episode")
            continue
        private = episode["private"]
        truth_ids = {i for issue in private["truth"]["issues"] for group in
                     [issue["affected_ids"], *issue["evidence_sets"]] for i in group}
        secret = {issue["issue_id"] for issue in private["truth"]["issues"]} | set(private["provenance"]["id_map"])
        outside = set()
        _ids_in([_without_records(observation), line["episode_id"]], outside)
        if outside & truth_ids:
            problems.append(f"{where}: truth id outside the records and the agent's actions")
        everything = set()
        _ids_in([observation, line["action"]], everything)
        public_ids = set()
        _ids_in(episode["public"], public_ids)
        if (everything & secret) - public_ids:
            problems.append(f"{where}: issue id or internal key")
        seed = private["provenance"].get("seed")
        if isinstance(seed, int) and abs(seed) >= 1000:
            text = json.dumps([_without_records(observation), line["reward"]])
            if re.search(rf"(?<![0-9]){abs(seed)}(?![0-9])", text):
                problems.append(f"{where}: the seed")
        if training:
            unseen = _submitted_ids(line["action"]) - seen.get(episode_id, set())
            if unseen:
                problems.append(f"{where}: submitted ids the agent was never shown (truth in a training file)")
        _ids_in(observation, seen.setdefault(episode_id, set()))
        _fx_keys_in(observation, seen[episode_id])
    return problems


# --- CLI --------------------------------------------------------------------------------------------
def _episode_dirs(directory) -> list[pathlib.Path]:
    root = pathlib.Path(directory)
    if (root / "public").is_dir():
        return [root]
    return sorted(path for path in root.iterdir() if (path / "public").is_dir())


def _load_policy(spec: str):
    module_name, _, class_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), class_name)()


def record(out, *, seeds=(), episodes_dir=None, config=None, policies=("random", "solver"), external=(),
           oracle_out=None, mode: str = "train") -> dict:
    """Record every policy on every episode. Returns {"train": path, "oracle": path or None, "episodes": {...}}."""
    from auditgym import Env
    from auditgym.generate import generate
    from auditgym.storage import load_episode
    from tools.metamorph import oracle_answer

    out = pathlib.Path(out)
    if oracle_out is None:
        oracle_out = out.with_name(out.name[:-len(out.suffix)] + ".oracle" + out.suffix if out.suffix
                                   else out.name + ".oracle.jsonl")
    sources = [("seed", seed) for seed in seeds] + [("dir", d) for d in (_episode_dirs(episodes_dir) if episodes_dir
                                                                         else [])]
    episodes = {}
    writers = {"train": TrajectoryWriter(out)}
    try:
        for kind, source in sources:
            episode = generate(source, config) if kind == "seed" else load_episode(source)
            episodes[episode["public"]["manifest"]["episode_id"]] = episode
            named = [(name, {"random": lambda: RandomPolicy(seed=7), "solver": SolverPolicy,
                             "oracle": lambda: OracleSubmitPolicy(oracle_answer(episode))}[name]())
                     for name in policies]
            named += [(spec, _load_policy(spec)) for spec in external]
            for name, policy in named:
                privileged = is_privileged(policy)
                if privileged and "oracle" not in writers:
                    writers["oracle"] = TrajectoryWriter(oracle_out, privileged=True)
                writer = writers["oracle" if privileged else "train"]
                env = Env(mode=mode) if kind == "seed" else Env.from_files(source, mode=mode)
                with env:
                    recorder = Recorder(env, writer, policy)
                    run_policy(recorder, policy, source if kind == "seed" else None, config if kind == "seed" else None)
    finally:
        for writer in writers.values():
            writer.close()
    return {"train": out, "oracle": oracle_out if "oracle" in writers else None, "episodes": episodes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.trajectories")
    parser.add_argument("--out", default=DEFAULT_OUT)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--episodes", help="episode directories (default: episodes/train of the repository)")
    source.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--config", default=None, help="JSON config for --seeds")
    parser.add_argument("--policies", default="random,solver")
    parser.add_argument("--policy", action="append", default=[], help="module:Class with act(observation)")
    parser.add_argument("--oracle-out", default=None)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    arguments = parser.parse_args(argv)
    if arguments.episodes is None and not arguments.seeds:
        arguments.episodes = str(DEFAULT_EPISODES)
    policies = tuple(p for p in arguments.policies.split(",") if p)
    config = json.loads(arguments.config) if arguments.config else None
    try:
        written = record(arguments.out, seeds=arguments.seeds or (), episodes_dir=arguments.episodes, config=config,
                         policies=policies, external=arguments.policy, oracle_out=arguments.oracle_out,
                         mode=arguments.mode)
    except TrajectoryError as exc:
        print(f"trajectories: {exc}", file=sys.stderr)
        return 2
    paths = (str(pathlib.Path(arguments.episodes).resolve()),) if arguments.episodes else ()
    problems = scan_file(written["train"], written["episodes"], paths)
    if written["oracle"] is not None:
        problems += [f"oracle file: {p}" for p in scan_file(written["oracle"], written["episodes"], paths,
                                                             training=False)]
    lines = len(pathlib.Path(written["train"]).read_text(encoding="utf-8").splitlines())
    print(f"trajectories: {lines} lines in {written['train']}"
          + (f", oracle in {written['oracle']}" if written["oracle"] else ""))
    for problem in problems:
        print("LEAK " + problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
