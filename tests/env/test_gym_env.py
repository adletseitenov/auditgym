"""The Gymnasium wrapper: works without gymnasium; check_env runs only when it is installed."""
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from auditgym import gym_env
from auditgym.client import EnvError
from auditgym.gym_env import ENV_ID, AuditGymEnv
from auditgym.secret import SECRET_ENV, SECRET_FILE_ENV
from auditgym.storage import load_episode
from policies.runner import make_policy

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EPISODE = ROOT / "episodes" / "train" / "ep_01"
HAVE_GYM = gym_env.gymnasium is not None


class WrapperTest(unittest.TestCase):
    def test_imports_and_has_the_gym_surface(self):
        env = AuditGymEnv()
        try:
            self.assertEqual(env.metadata, {"render_modes": []})
            self.assertIsNone(env.render())
            for space in (env.observation_space, env.action_space):
                self.assertEqual(space.max_length, gym_env.MAX_TEXT_LENGTH)
                self.assertTrue(space.contains('{"a":1}'))
                self.assertFalse(space.contains(5))
        finally:
            env.close()

    def test_core_does_not_import_gymnasium(self):
        import subprocess
        code = ("import sys, auditgym, auditgym.client, auditgym.evaluator, auditgym.serve, auditgym.shaped;"
                "print('gymnasium' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "False")

    def test_reset_step_close_with_options_as_config(self):
        env = AuditGymEnv()
        try:
            observation, info = env.reset(seed=3, options={"errors": {"duplicate_invoice": 1}})
            self.assertIsInstance(observation, str)
            self.assertEqual(json.loads(observation)["type"], "reset")
            self.assertEqual(info, {})
            self.assertTrue(env.observation_space.contains(observation))
            for action in ('{"action": "calculator", "expression": "2*21"}',
                           {"action": "calculator", "expression": "2*21"}):
                observation, reward, terminated, truncated, info = env.step(action)
                self.assertEqual(json.loads(observation)["result"], "42")
                self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))
            observation, reward, *_ = env.step("this is not json")  # a bounded error, a step spent
            self.assertEqual(json.loads(observation)["type"], "error")
        finally:
            env.close()
        with self.assertRaises(EnvError):
            env.step("{}")  # closed

    def test_same_seed_same_observation(self):
        env = AuditGymEnv()
        try:
            self.assertEqual(env.reset(seed=11)[0], env.reset(seed=11)[0])
        finally:
            env.close()

    def test_oracle_through_the_wrapper_gets_one(self):
        truth = load_episode(EPISODE)["private"]["truth"]
        env = AuditGymEnv.from_files(EPISODE)
        try:
            observation, _ = env.reset()
            policy = make_policy("oracle", truth=truth)
            policy.reset(json.loads(observation))
            action = policy.act(json.loads(observation))
            observation, reward, terminated, truncated, info = env.step(json.dumps(action))
        finally:
            env.close()
        self.assertEqual(json.loads(observation)["type"], "final")
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))
        self.assertEqual(info["f1"], 1.0)
        self.assertNotIn("shaped_reward", info)

    def test_eval_mode_hides_reward_and_shaped(self):
        with tempfile.TemporaryDirectory() as scratch:
            truth = load_episode(EPISODE)["private"]["truth"]
            env = AuditGymEnv.from_files(EPISODE, mode="eval", results_path=scratch + "/r.jsonl", shaped_reward=True)
            try:
                observation, _ = env.reset()
                policy = make_policy("oracle", truth=truth)
                policy.reset(json.loads(observation))
                _, reward, terminated, _, info = env.step(policy.act(json.loads(observation)))
            finally:
                env.close()
        self.assertEqual((reward, terminated, info), (0.0, True, {}))

    def test_eval_with_a_secret_file_plays_the_episode_generated_under_that_secret(self):
        from auditgym.generate import generate

        config = {"errors": {"duplicate_invoice": 1}}
        with tempfile.TemporaryDirectory() as scratch:
            secret = pathlib.Path(scratch) / "secret"
            secret.write_text("jury-only-value\n", encoding="utf-8")
            results = scratch + "/r.jsonl"
            env = AuditGymEnv(mode="eval", results_path=results, secret_file=secret)
            try:
                observation, _ = env.reset(seed=4, options=config)
            finally:
                env.close()
            loaded = AuditGymEnv.from_files(EPISODE, mode="eval", results_path=results, secret_file=secret)
            try:
                self.assertEqual(json.loads(loaded.reset()[0])["type"], "reset")
            finally:
                loaded.close()
            with mock.patch.dict(os.environ, {SECRET_ENV: "", SECRET_FILE_ENV: str(secret)}):
                for build in (lambda: AuditGymEnv(mode="eval", results_path=results),
                              lambda: AuditGymEnv.from_files(EPISODE, mode="eval", results_path=results)):
                    with self.assertRaises(EnvError) as caught:
                        build()
                    self.assertEqual(caught.exception.code, "bad_secret_file")
        with_secret = generate(4, config, secret="jury-only-value")["public"]["manifest"]["episode_id"]
        without = generate(4, config)["public"]["manifest"]["episode_id"]
        self.assertEqual(json.loads(observation)["manifest"]["episode_id"], with_secret)
        self.assertNotEqual(with_secret, without)

    def test_registration_is_idempotent_and_only_with_gymnasium(self):
        gym_env.register()
        gym_env.register()
        if HAVE_GYM:
            self.assertIn(ENV_ID, gym_env.gymnasium.registry)


@unittest.skipUnless(HAVE_GYM, "gymnasium is not installed")
class GymnasiumTest(unittest.TestCase):
    def test_check_env(self):
        from gymnasium.utils.env_checker import check_env

        env = AuditGymEnv(episode_dir=EPISODE)
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_make(self):
        env = gym_env.gymnasium.make(ENV_ID)
        try:
            observation, info = env.reset(seed=1)
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
