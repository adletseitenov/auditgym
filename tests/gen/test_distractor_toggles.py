"""config["distractors"] as an object of named switches (docs/DECISIONS.md T): each clean pattern can be
switched on alone and shows itself and nothing else; true and false keep producing the episodes they
always did; the books verify and the oracle scores 1.0 on a clean company."""
import hashlib
import json
import unittest

from auditgym import Env
from auditgym.config import DISTRACTOR_KINDS, ConfigError, config_digest, distractor_flags, normalize_config
from auditgym.generate import generate
from auditgym.rng import Rng
from auditgym.verify import predicate_issues, verify
from auditgym.world import build_world
from policies.runner import make_policy, run_episode
from tools.patterns import distractor_patterns

DENSE = {"duplicate_invoice": 2, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 2}
# SHA-256 of json.dumps(generate(seed, config), sort_keys=True, default=str), taken on main (00cea75)
# before the object form existed: true, false and omitted `distractors` must not change a byte.
PINNED = [
    ({"errors": {}, "distractors": True, "currencies": ["USD"], "timezone": "+05:00"}, 0,
     "68864b2e315e3f79f16c393eb2f4a95808de30f0800f3a1c719378181ad1592b"),
    ({"errors": {}, "distractors": True, "currencies": ["USD"], "timezone": "+05:00"}, 7,
     "f12a02abb381495c11083ee3c775cf626765cf49b6a45a476b0cd2e7ff05c431"),
    ({"errors": {}, "distractors": True, "currencies": ["EUR", "CNY"], "timezone": "-11:00"}, 0,
     "f62d839ae16a7fbb9f68a357560c24f293e72ace45e838c0f9bf7456291fbf20"),
    ({"errors": DENSE, "distractors": True, "currencies": ["USD"], "timezone": "+05:00"}, 0,
     "d71b96b3a11f34fc79acfb01b556d9165784c834c3e0fe90b2e7ba523038cc7d"),
    ({"errors": {}, "distractors": False, "currencies": ["USD"], "timezone": "+05:00"}, 0,
     "cef53fa0d3e44fe6c3c2d43e1f3c8c606e220bc2ad5dd64e96cfd67976f4ea4b"),
    ({"errors": {}, "distractors": False, "currencies": ["EUR", "CNY"], "timezone": "-11:00"}, 7,
     "6e67e31da3cab6d51b0037c96abbad303626c80384143f14ded4555e0bf58794"),
    ({"errors": DENSE, "distractors": False, "currencies": ["USD"], "timezone": "+05:00"}, 0,
     "a0a99ba9abdd37d2b5a241d2093ccdbd5bdff08285fe35ec9c14540724481ebe"),
    ({}, 0, "1922cd7d9bca5d6380b76eb27d6060d5908c092020dfd479ddf56a1ee0fd741a"),
]
ZONES = [(["USD"], "+05:00"), (["EUR", "CNY"], "-11:00"), (["USD", "EUR"], "+14:00"), (["CNY"], "-03:00")]


def clean(value, seed):
    currencies, timezone = ZONES[seed % len(ZONES)]
    return {"errors": {}, "currencies": currencies, "timezone": timezone, "distractors": value}


class NormalizationTest(unittest.TestCase):
    def test_bools_are_stored_as_given(self):
        self.assertIs(normalize_config({})["distractors"], True)
        self.assertIs(normalize_config({"distractors": True})["distractors"], True)
        self.assertIs(normalize_config({"distractors": False})["distractors"], False)

    def test_object_is_completed_with_false(self):
        config = normalize_config({"distractors": {"refunds": True}})
        self.assertEqual(list(config["distractors"]), list(DISTRACTOR_KINDS))
        self.assertEqual({k for k, v in config["distractors"].items() if v}, {"refunds"})
        self.assertEqual(config, normalize_config({"distractors": {"refunds": True, "advances": False}}))
        self.assertEqual(config_digest(config), config_digest(normalize_config(config)))
        self.assertNotEqual(config_digest(config), config_digest(normalize_config({"distractors": False})))

    def test_flags(self):
        self.assertEqual(set(distractor_flags(True).values()), {True})
        self.assertEqual(set(distractor_flags(False).values()), {False})
        self.assertEqual(distractor_flags({"refunds": True})["refunds"], True)

    def test_bad_values(self):
        for value in (1, "true", None, [], ["refunds"], {"refund": True}, {"refunds": 1}, {"refunds": None},
                      {1: True}, {"refunds": "yes"}):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ConfigError) as caught:
                    normalize_config({"distractors": value})
                self.assertEqual(caught.exception.code, "bad_distractors")
                self.assertIn("split_payments", str(caught.exception))


class BoolFormsUnchangedTest(unittest.TestCase):
    def test_pinned_episodes(self):
        for config, seed, digest in PINNED:
            with self.subTest(seed=seed, config=json.dumps(config)[:60]):
                text = json.dumps(generate(seed, config), sort_keys=True, default=str)
                self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), digest)

    def test_omitted_equals_true(self):
        self.assertEqual(generate(3, {"errors": {}}), generate(3, {"errors": {}, "distractors": True}))

    def test_empty_object_builds_the_world_of_false(self):
        for seed in range(4):
            worlds = []
            for value in (False, {}):
                config = normalize_config(clean(value, seed))
                world = build_world(Rng(seed, "t").child("world"), config)
                worlds.append({k: v for k, v in world.items() if k not in ("flags", "selective")})
            self.assertEqual(worlds[0], worlds[1])


class SwitchesTest(unittest.TestCase):
    def test_each_switch_alone_shows_only_its_pattern_in_every_episode(self):
        for kind in DISTRACTOR_KINDS:
            for seed in range(10):
                with self.subTest(kind=kind, seed=seed):
                    episode = generate(seed, clean({kind: True}, seed))
                    self.assertEqual(distractor_patterns(episode["public"]), {kind})
                    self.assertEqual(verify(episode), [])
                    self.assertEqual(predicate_issues(episode["public"]), ([], []))
                    self.assertEqual(episode["private"]["truth"]["issues"], [])

    def test_pairs_and_all(self):
        for index, (a, b) in enumerate(zip(DISTRACTOR_KINDS, DISTRACTOR_KINDS[3:] + DISTRACTOR_KINDS[:3])):
            episode = generate(index, clean({a: True, b: True}, index))
            self.assertEqual(distractor_patterns(episode["public"]), {a, b})
        for seed in range(6):
            episode = generate(seed, clean({k: True for k in DISTRACTOR_KINDS}, seed))
            self.assertEqual(distractor_patterns(episode["public"]), set(DISTRACTOR_KINDS))
            self.assertEqual(verify(episode), [])

    def test_all_off_shows_nothing(self):
        for value in (False, {}, {"refunds": False}):
            for seed in range(4):
                self.assertEqual(distractor_patterns(generate(seed, clean(value, seed))["public"]), set())

    def test_weekend_rates_need_a_foreign_currency(self):
        episode = generate(1, {"errors": {}, "currencies": [], "distractors": {"weekend_fx": True}})
        self.assertEqual(episode["public"]["fx"], [])
        self.assertEqual(verify(episode), [])

    def test_switches_next_to_errors_keep_the_truth(self):
        config = {"errors": DENSE, "currencies": ["USD"], "distractors": {"refunds": True, "other_offsets": True}}
        for seed in range(6):
            episode = generate(seed, config)
            self.assertEqual(verify(episode), [])
            self.assertEqual(len(episode["private"]["truth"]["issues"]), 6)

    def test_same_config_same_bytes(self):
        config = clean({"split_payments": True, "weekend_fx": True}, 0)
        self.assertEqual(generate(4, config), generate(4, config))

    def test_oracle_scores_one_on_a_clean_company(self):
        for kind in DISTRACTOR_KINDS:
            config = clean({kind: True}, 1)
            with Env(mode="train") as env:
                episode = generate(2, config)
                result = run_episode(env, make_policy("oracle", truth=episode["private"]["truth"]), 2, config)
            self.assertEqual(result["reward"], 1.0, kind)


if __name__ == "__main__":
    unittest.main()
