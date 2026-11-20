"""A baseline policy must not depend on the names of ids or on the order of rows (CASE §06, §10).

The episode is rewritten by tools.transform (shuffled rows, every id renamed, several id styles);
the reward of every table-reading policy must stay exactly the same. random_explorer is left out on
purpose: its stream is derived from the seed and the episode_id, and it picks ids at random from
what it has seen, so it is random by design; the oracle does not read tables.
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

from auditgym import Env, levels
from auditgym.generate import generate
from auditgym.storage import write_episode
from policies.runner import make_policy, run_episode
from tools.metamorph import CANDIDATES
from tools.transform import ID_STYLES, transform

POLICIES = ("reconcile", "heuristic", "flag_all", "flag_all_cutoff", "flag_all_duplicates", "silence",
            "solver_policy")
CONFIGS = dict(CANDIDATES)
CASES = (
    ("level max", 11, levels.load("max")),
    ("level hard", 5, levels.load("hard")),
    ("I1", 3, CONFIGS["I1"]),
    ("I2", 3, CONFIGS["I2"]),
    ("I1+all", 7, CONFIGS["I1+all"]),
    ("I2+all", 7, CONFIGS["I2+all"]),
    ("dup", 7, CONFIGS["dup"]),
    ("all-four", 7, CONFIGS["all-four"]),
)


def _reward(name: str, directory: pathlib.Path) -> float:
    env = Env.from_files(str(directory))
    try:
        return run_episode(env, make_policy(name), None)["reward"]
    finally:
        env.close()


class RenameInvarianceTest(unittest.TestCase):
    def test_shuffled_and_renamed_episode_gives_the_same_reward(self):
        for number, (label, seed, config) in enumerate(CASES):
            with tempfile.TemporaryDirectory() as scratch:
                source, changed = pathlib.Path(scratch) / "src", pathlib.Path(scratch) / "changed"
                write_episode(generate(seed, config), source)
                transform(source, changed, seed=seed + 99, shuffle=True, rename=True,
                          id_style=ID_STYLES[number % len(ID_STYLES)])
                self.assertNotEqual((source / "public" / "ledger.csv").read_bytes(),
                                    (changed / "public" / "ledger.csv").read_bytes())
                for name in POLICIES:
                    with self.subTest(case=label, policy=name):
                        self.assertEqual(_reward(name, source), _reward(name, changed))


if __name__ == "__main__":
    unittest.main()
