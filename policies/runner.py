"""Run one episode of a policy through an `auditgym.Env` (train or eval mode).

    from auditgym import Env
    from policies.runner import run_episode, make_policy
    with Env(mode="train") as env:
        result = run_episode(env, make_policy("solver_policy"), seed=7, config=None)

With `seed=None` the env must come from `Env.from_files(dir)`: reset() restarts that episode.
In eval mode the policy sees reward 0.0 and info {}; the real reward goes to the evaluator's
`--results` file, which `read_results(path)` reads back.
"""
from __future__ import annotations

import json
import time

MAX_STEPS = 100
NAMES = ("random_explorer", "reconcile", "flag_all", "flag_all_cutoff", "flag_all_duplicates",
         "silence", "solver_policy", "oracle")


def make_policy(name: str, truth: dict | None = None, **options):
    """A fresh policy by name. `truth` is required for the oracle and only for it."""
    if name == "random_explorer":
        from .random_explorer import RandomExplorer
        return RandomExplorer(**options)
    if name == "reconcile":
        from .reconcile import Reconcile
        return Reconcile(**options)
    if name.startswith("flag_all"):
        from .flag_all import FlagAll
        return FlagAll(variant="mixed" if name == "flag_all" else name[len("flag_all_"):])
    if name == "silence":
        from .silence import Silence
        return Silence()
    if name == "solver_policy":
        from .solver_policy import SolverPolicy
        return SolverPolicy()
    if name == "oracle":
        if truth is None:
            raise ValueError("the oracle needs the episode truth (evaluator side)")
        from .oracle import Oracle
        return Oracle(truth)
    raise ValueError(f"unknown policy {name!r}")


def _submitted(action: object):
    """The findings payload of a submit action, flat or nested (CONTRACTS §5.2), else None."""
    if not isinstance(action, dict):
        return None
    name = next((action[key] for key in ("action", "tool", "name", "type") if key in action), None)
    if name != "submit":
        return None
    arguments = dict(action)
    for key in ("args", "arguments", "params"):
        if isinstance(action.get(key), dict):
            arguments.update(action[key])
    return arguments.get("findings")


def run_episode(env, policy, seed=None, config=None, record: bool = False) -> dict:
    """Play one episode. Returns steps, reward, terminated, truncated, info, the submitted
    payload and findings, wall time, and (with record=True) the trajectory rows."""
    started = time.perf_counter()
    observation, info = env.reset() if seed is None else env.reset(seed, config)
    episode_id = (observation.get("manifest") or {}).get("episode_id")
    policy.reset(observation)
    steps, reward, terminated, truncated, payload = 0, 0.0, False, False, None
    trajectory = []
    while not (terminated or truncated):
        if steps >= MAX_STEPS:
            raise RuntimeError("environment did not end the episode at the step limit")
        action = policy.act(observation)
        if _submitted(action) is not None:
            payload = _submitted(action)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if record:
            trajectory.append({"episode_id": episode_id, "step": steps, "action": action,
                               "observation": observation, "reward": reward,
                               "terminated": terminated, "truncated": truncated})
    findings = payload.get("findings") if isinstance(payload, dict) else None
    result = {"policy": getattr(policy, "name", type(policy).__name__), "episode_id": episode_id,
              "steps": steps, "reward": reward, "terminated": terminated, "truncated": truncated,
              "info": info, "payload": payload, "findings": findings if isinstance(findings, list) else None,
              "seconds": time.perf_counter() - started}
    if record:
        result["trajectory"] = trajectory
    return result


def read_results(path) -> list[dict]:
    """Evaluator-side: the lines of a `--results` file (one JSON object per finished episode)."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
