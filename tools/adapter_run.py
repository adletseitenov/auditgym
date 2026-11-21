"""The four baseline policies of the case through `judge_adapter.make_env()` only.

    python judge_adapter.py --demo [--seeds N]      (same as python -m tools.adapter_run)

`PolicyBridge` is the policy side of the adapter: it turns what the policies of `policies/` emit
(flat internal actions) into the envelope {"name", "args"}, and the adapter's observations back into
the shape those policies read. The policy classes are the same objects `tools/score.py` runs; none of
them was rewritten, and none of them touches the evaluator, an episode directory or a truth: they
see `reset` and `step` of the adapter and nothing else. The oracle is the one exception that is
built from the truth, on the jury side (here: this script), and it still submits through `step`.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import warnings

BASELINES = ("random_explorer", "reconcile", "flag_all", "oracle")
DEMO_CONFIGS = (
    ("clean", {"errors": {}}),
    ("one-duplicate", {"errors": {"duplicate_invoice": 1}}),
    ("all-four", {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1}}),
    ("interaction-I1", {"errors": {}, "interaction": "I1"}),
)


class PolicyBridge:
    """Wraps a policy of `policies/` (reset(observation), act(observation) -> action) around an
    adapter environment. It adds nothing the adapter did not return."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self._last = (None, None)

    @staticmethod
    def to_envelope(action: dict) -> dict:
        name = action.get("action")
        if name == "submit":
            return {"name": "submit", "args": {"findings": action.get("findings")}}
        return {"name": name, "args": {key: value for key, value in action.items() if key != "action"}}

    def _policy_view(self, observation: dict) -> dict:
        """The adapter observation in the shape of the baseline policies."""
        name, table = self._last
        view = {"steps_remaining": observation["remaining_steps"]}
        if observation["error"] is not None:
            return {**view, "type": "error", "error": observation["error"]}
        result = observation["result"]
        if name == "list_records":
            return {**view, "type": "records", "table": table, "records": result["records"],
                    "next_cursor": result["next_cursor"]}
        if name == "get_record":
            return {**view, "type": "record", "table": table, "record": result}
        if name == "get_document":
            return {**view, "type": "document", "record": result}
        if name == "calculator":
            return {**view, "type": "calculation", "result": result}
        return {**view, "type": "final", "result": result}

    def run(self, env, seed, config) -> dict:
        """One episode. Returns what the policy side saw: steps taken, the last reward, flags."""
        observation, _ = env.reset(seed, config)
        view = {"type": "reset", "task": observation["task"], "manifest": observation["manifest"],
                "policy": observation["policy"], "steps_remaining": observation["remaining_steps"]}
        self.policy.reset(view)
        steps, reward, terminated, truncated, last = 0, 0.0, False, False, None
        while not (terminated or truncated):
            if steps >= 100:
                raise RuntimeError("the adapter did not end the episode at 100 actions")
            action = self.policy.act(view)
            envelope = self.to_envelope(action)
            self._last = (envelope["name"], envelope["args"].get("table"))
            observation, reward, terminated, truncated, info = env.step(envelope)
            steps += 1
            last = observation
            view = self._policy_view(observation)
        return {"steps": steps, "reward": reward, "terminated": terminated, "truncated": truncated,
                "last": last, "info": info}


def make_baseline(name: str, truth: dict | None = None):
    """Fresh policy: random explorer with a fixed seed, simple reconciliation, flag-all, oracle."""
    from policies.runner import make_policy

    if name == "random_explorer":
        return make_policy("random_explorer", seed=7)
    return make_policy(name, truth=truth)


def run_baselines(episode_dir, journal_path, names=BASELINES) -> list[dict]:
    """The baselines on one episode directory, closed mode, through make_env. The jury side reads the
    truth only for the oracle and scores through `score_episode` from the findings the policy sent."""
    import judge_adapter
    from auditgym.storage import load_episode

    truth = load_episode(episode_dir)["private"]["truth"]
    rows = []
    with judge_adapter.make_env(journal_path=journal_path) as env:
        for name in names:
            sent = []
            policy = make_baseline(name, truth if name == "oracle" else None)
            bridge = PolicyBridge(policy)
            original = bridge.to_envelope

            def spy(action, original=original, sent=sent):
                envelope = original(action)
                if envelope["name"] == "submit":
                    sent.append(envelope["args"]["findings"])
                return envelope

            bridge.to_envelope = spy
            seen = bridge.run(env, 0, {"mode": "closed", "episode_dir": str(episode_dir)})
            findings = sent[-1] if sent else None
            score = judge_adapter.score_episode(episode_dir, findings, env.inspection_steps,
                                                journal_path=str(journal_path) + ".score")
            rows.append({"policy": name, "steps": seen["steps"], "policy_saw_reward": seen["reward"],
                         "terminated": seen["terminated"], "truncated": seen["truncated"],
                         "journal_reward": (env.last_result() or {}).get("reward"), **score})
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python judge_adapter.py --demo")
    parser.add_argument("--seeds", type=int, default=1, help="episodes per demo configuration")
    arguments = parser.parse_args(argv)
    import judge_adapter

    ok = True
    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch)
        print(f"{'episode':<19}{'policy':<17}{'steps':>5} {'saw':>4} {'tp':>3}{'fp':>3}{'fn':>3} {'return':>8} {'journal':>8}")
        for label, config in DEMO_CONFIGS:
            for seed in range(arguments.seeds):
                directory = root / f"{label}-{seed}"
                with warnings.catch_warnings():  # a train episode of the demo: the no-secret warning is for hidden runs
                    warnings.filterwarnings("ignore", message="generate_episode: no secret", category=RuntimeWarning)
                    judge_adapter.generate_episode(5000 + seed, config, str(directory))
                for row in run_baselines(directory, root / "results.jsonl"):
                    same = row["journal_reward"] == row["return"]
                    ok = ok and same and row["policy_saw_reward"] == 0
                    print(f"{label + '#' + str(seed):<19}{row['policy']:<17}{row['steps']:>5} {row['policy_saw_reward']:>4} "
                          f"{row['tp']!s:>3}{row['fp']!s:>3}{row['fn']!s:>3} {row['return']:>8.3f} "
                          f"{row['journal_reward']:>8.3f}" + ("" if same else "  MISMATCH"))
    print("demo:", "ok (policy saw reward 0; the journal reward equals score_episode)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    sys.exit(main(sys.argv[1:]))
