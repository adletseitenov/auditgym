"""Trajectory logs (CASE §09: trajectories.jsonl with episode_id, step, action, observation, reward,
terminated, truncated; «В наблюдениях обучающих траекторий нет закрытого эталона»).

    python -m tools.trajectories [--out trajectories.jsonl] [--episodes DIR | --seeds 1 2 3]
                                 [--config JSON] [--policies random_explorer,reconcile,flag_all,oracle]
                                 [--policy module:Class ...] [--oracle-out PATH] [--mode train|eval]
                                 [--split train|dev] [--all-episodes]

With no arguments it records random_explorer, reconcile, flag_all, solver_policy and the oracle
(`policies/`) on the shipped train and dev episodes (`episodes/train`, `episodes/dev`) into
`trajectories.jsonl` (`make trajectories`); the oracle goes to `trajectories.oracle.jsonl`. The
output directory is created when it does not exist. The successful policy (solver_policy, which reads
only what the environment shows) plays all 20 episodes; the failing readers (reconcile, flag_all) play
the books of DEFAULT_PICK only, so the file stays about 5 MB.

Each line has the case fields plus `policy` (the name of the policy that played the episode) and
`split` (train or dev: the directory the episode came from; `--split` names it for episodes from
elsewhere), in this order: episode_id, policy, split, step, action, observation, reward, terminated,
truncated. The last
line of an episode (terminated or truncated) also carries `metrics` {precision, recall, f1}, taken
from the terminal info of a train-mode run (an eval-mode run has no metrics to give, so the key is
absent). Step 0 is the reset: action null, reward 0.0, both flags false. Nothing else of `info`
is written.

For other engineers' policies: wrap the Env in a Recorder and drive it as usual.

    with Env() as env, TrajectoryWriter("trajectories.jsonl") as writer:
        recorder = Recorder(env, writer, policy, split=where)
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

FIELDS = ("episode_id", "policy", "split", "step", "action", "observation", "reward", "terminated", "truncated")
FINAL_FIELDS = FIELDS + ("metrics",)  # the last line of an episode of a train-mode run
METRIC_KEYS = ("precision", "recall", "f1")
BASELINES = ("random_explorer", "reconcile", "flag_all", "oracle")
DEFAULT_POLICIES = ("random_explorer", "reconcile", "flag_all", "solver_policy", "oracle")
TABLES = ("ledger", "invoices", "payments", "fx")
SPLITS = ("train", "dev")
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EPISODES = (ROOT / "episodes" / "train", ROOT / "episodes" / "dev")
# Every reading policy pages every table, about 0.2 MB per episode, so 20 episodes of one policy are
# 4 MB and several would not fit the size we keep for the file (about 5 MB). The honest solver plays all
# 20 books; random_explorer is cheap (0.6 MB for all 20) and plays all too; the failing readers
# (reconcile, flag_all) play two dev books, one with errors and one clean (the smallest of each kind), and the
# oracle (its own private file) plays everything. `--all-episodes` records every policy on every episode.
POLICY_PICK = {"reconcile": {"dev": ("ep_05", "ep_09")},
               "flag_all": {"dev": ("ep_05", "ep_09")}}
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
        if tuple(line) not in (FIELDS, FINAL_FIELDS):
            raise TrajectoryError(f"trajectory line must have exactly {FIELDS} (and `metrics` on the last line)")
        if "metrics" in line and tuple(line["metrics"]) != METRIC_KEYS:
            raise TrajectoryError(f"metrics must have exactly {METRIC_KEYS}")
        self._handle.write(json.dumps(line, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self.lines += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class Recorder:
    """An Env with the same reset/step API that writes every transition (never `info`)."""

    def __init__(self, env, writer: TrajectoryWriter, policy=None, split: str = "train") -> None:
        if policy is not None and is_privileged(policy) and not writer.privileged:
            raise TrajectoryError("a privileged policy must write to a separate oracle file")
        if split not in SPLITS:
            raise TrajectoryError(f"split must be one of {SPLITS}")
        self.env = env
        self.writer = writer
        self.split = split
        self.policy_name = str(getattr(policy, "name", None) or type(policy).__name__)
        self.episode_id = None
        self.step_number = 0

    def _write(self, action, observation, reward, terminated, truncated, info=None) -> None:
        line = {"episode_id": self.episode_id, "policy": self.policy_name, "split": self.split,
                "step": self.step_number,
                "action": action, "observation": observation, "reward": reward, "terminated": terminated,
                "truncated": truncated}
        if (terminated or truncated) and info and all(key in info for key in METRIC_KEYS):
            line["metrics"] = {key: info[key] for key in METRIC_KEYS}
        self.writer.write(line)

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
        self._write(action, observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info


def run_policy(recorder: Recorder, policy, seed=None, config=None, max_steps: int = 100) -> float:
    observation, _ = recorder.reset(seed, config)
    if hasattr(policy, "reset"):  # the policies of policies/ take the first observation here
        policy.reset(observation)
    reward = 0.0
    for _ in range(max_steps + 1):
        observation, reward, terminated, truncated, _ = recorder.step(policy.act(observation))
        if terminated or truncated:
            break
    return reward


# --- built-in policies ----------------------------------------------------------------------------
class RandomPolicy:
    """Random explorer with a fixed seed: reads pages and records it has seen, then submits a guess."""

    name = "random"

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

    name = "solver"

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
    name = "oracle"

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
    """Ids a submit cites, in every form the environment accepts (a bare list, flat, a JSON string,
    under "findings.json"), not only the documented one: truth ids hidden in another form must
    not pass the training scan."""
    from auditgym.actions import ActionError, parse_action, submit_payload

    try:
        name, arguments = parse_action(action)
        payload = submit_payload(arguments, "") if name == "submit" else None
    except ActionError:
        return set()
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
        return set()
    ids = set()
    for finding in payload["findings"]:
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
    expected_step: dict[tuple, int] = {}
    splits: dict[str, str] = {}
    seen: dict[tuple, set] = {}
    for number, raw in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        where = f"line {number}"
        try:
            line = json.loads(raw)
        except ValueError:
            problems.append(f"{where}: not JSON")
            continue
        if not isinstance(line, dict) or tuple(line) not in (FIELDS, FINAL_FIELDS):
            problems.append(f"{where}: fields {list(line) if isinstance(line, dict) else line}")
            continue
        episode_id, step, observation = line["episode_id"], line["step"], line["observation"]
        if not isinstance(line["policy"], str) or not line["policy"]:
            problems.append(f"{where}: policy name")
        if "metrics" in line and not (line["terminated"] or line["truncated"]):
            problems.append(f"{where}: metrics on a line that does not end an episode")
        if "metrics" in line and (not isinstance(line["metrics"], dict) or tuple(line["metrics"]) != METRIC_KEYS):
            problems.append(f"{where}: metrics shape")
        if line["split"] not in SPLITS:
            problems.append(f"{where}: split {line['split']!r}")
        elif splits.setdefault(episode_id, line["split"]) != line["split"]:
            problems.append(f"{where}: one episode in two splits")
        run = (episode_id, line["policy"])  # one policy plays one episode at a time
        if step != expected_step.get(run, 0) and step != 0:
            problems.append(f"{where}: step {step} out of order")
        expected_step[run] = step + 1
        if step == 0:
            seen[run] = set()
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
            # A real id of the episode the agent was never shown; an invented token (random_explorer
            # guesses one when it has seen no document) exists nowhere and carries nothing.
            unseen = (_submitted_ids(line["action"]) - seen.get(run, set())) & (public_ids | truth_ids)
            if unseen:
                problems.append(f"{where}: submitted ids the agent was never shown (truth in a training file)")
        _ids_in(observation, seen.setdefault(run, set()))
        _fx_keys_in(observation, seen[run])
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


def _builtin(name: str, episode: dict):
    """A policy by name: `random` and `solver` are the scripted ones of this module, every name of
    policies.runner.NAMES is the baseline of that name (the oracle gets the truth: evaluator side)."""
    from policies.runner import NAMES, make_policy

    if name == "random":
        return RandomPolicy(seed=7)
    if name == "solver":
        return SolverPolicy()
    if name not in NAMES:
        raise TrajectoryError(f"unknown policy {name!r}")
    return make_policy(name, truth=episode["private"]["truth"] if name == "oracle" else None)


def _split_of(path, default: str) -> str:
    """train or dev when the episode directory sits in episodes/train or episodes/dev, else `default`."""
    parent = pathlib.Path(path).resolve().parent.name
    return parent if parent in SPLITS else default


def record(out, *, seeds=(), episodes_dir=None, config=None, policies=BASELINES, external=(),
           oracle_out=None, mode: str = "train", split: str = "train", restrict=None) -> dict:
    """Record every policy on every episode. `episodes_dir` is one directory or a list of them
    (each holds episode directories or is one). The split of an episode is its parent directory when that is
    `train` or `dev`, else `split`. `restrict` maps a policy name to the set of episode directories
    (`<split>/<name>`, as in POLICY_PICK) it plays; a policy not in it plays everything. Returns {"train": path, "oracle": path or None,
    "episodes": {...}}."""
    from auditgym import Env
    from auditgym.generate import generate
    from auditgym.storage import load_episode

    out = pathlib.Path(out)
    if oracle_out is None:
        oracle_out = out.with_name(out.name[:-len(out.suffix)] + ".oracle" + out.suffix if out.suffix
                                   else out.name + ".oracle.jsonl")
    roots = [episodes_dir] if isinstance(episodes_dir, (str, pathlib.Path)) else list(episodes_dir or ())
    sources = [("seed", seed) for seed in seeds] + [("dir", d) for root in roots for d in _episode_dirs(root)]
    episodes = {}
    writers = {"train": TrajectoryWriter(out)}
    try:
        for kind, source in sources:
            episode = generate(source, config) if kind == "seed" else load_episode(source)
            episodes[episode["public"]["manifest"]["episode_id"]] = episode
            where = _split_of(source, split) if kind == "dir" else split
            if kind == "dir":
                label = f"{where}/{pathlib.Path(source).name}"
                playing = [name for name in policies if restrict is None or name not in restrict
                           or label in restrict[name]]
            else:
                playing = list(policies)
            named = [(name, _builtin(name, episode)) for name in playing]
            named += [(spec, _load_policy(spec)) for spec in external]
            for name, policy in named:
                privileged = is_privileged(policy)
                if privileged and "oracle" not in writers:
                    writers["oracle"] = TrajectoryWriter(oracle_out, privileged=True)
                writer = writers["oracle" if privileged else "train"]
                env = Env(mode=mode) if kind == "seed" else Env.from_files(source, mode=mode)
                with env:
                    recorder = Recorder(env, writer, policy, split=where)
                    run_policy(recorder, policy, source if kind == "seed" else None, config if kind == "seed" else None)
    finally:
        for writer in writers.values():
            writer.close()
    return {"train": out, "oracle": oracle_out if "oracle" in writers else None, "episodes": episodes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.trajectories")
    parser.add_argument("--out", default=DEFAULT_OUT)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--episodes", nargs="+",
                        help="episode directories (default: DEFAULT_PICK of episodes/train and episodes/dev)")
    source.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--config", default=None, help="JSON config for --seeds")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                        help="random, solver or a name of policies.runner.NAMES")
    parser.add_argument("--policy", action="append", default=[], help="module:Class with act(observation)")
    parser.add_argument("--oracle-out", default=None)
    parser.add_argument("--all-episodes", action="store_true",
                        help="with the default directories: every policy on every episode, not POLICY_PICK")
    parser.add_argument("--split", choices=SPLITS, default="train",
                        help="split label for episodes that are not in a train/ or dev/ directory (and for --seeds)")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    arguments = parser.parse_args(argv)
    restrict = None
    if arguments.episodes is None and not arguments.seeds:
        arguments.episodes = [str(path) for path in DEFAULT_EPISODES]
        if not arguments.all_episodes:
            restrict = {name: {f"{part}/{book}" for part, books in picks.items() for book in books}
                        for name, picks in POLICY_PICK.items()}
    policies = tuple(p for p in arguments.policies.split(",") if p)
    config = json.loads(arguments.config) if arguments.config else None
    try:
        written = record(arguments.out, seeds=arguments.seeds or (), episodes_dir=arguments.episodes, config=config,
                         policies=policies, external=arguments.policy, oracle_out=arguments.oracle_out,
                         mode=arguments.mode, split=arguments.split, restrict=restrict)
    except TrajectoryError as exc:
        print(f"trajectories: {exc}", file=sys.stderr)
        return 2
    paths = tuple(str(pathlib.Path(path).resolve()) for path in arguments.episodes or ())
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
