"""The README recipe `Env(server_command=["docker", "run", "-i", ...])` against a real container.

Runs only where `docker` exists and the image is built (`docker build -t auditgym .`; another tag in
$AUDITGYM_DOCKER_IMAGE). One episode of the solver policy goes through the container's evaluator, which
signs its journal with a secret the host never passes in the environment; the host then verifies it."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from auditgym import Env
from policies.runner import make_policy, run_episode
from tests.env.helpers import ROOT

IMAGE = os.environ.get("AUDITGYM_DOCKER_IMAGE", "auditgym")


def _image_state() -> str | None:
    """None when the image can be used, else the reason to skip."""
    if not shutil.which("docker"):
        return "docker is not installed"
    try:
        found = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "docker did not answer"
    if found.returncode != 0:
        return f"image '{IMAGE}' is not built (docker build -t {IMAGE} .)"
    return None


_SKIP = _image_state()


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class DockerRecipeTest(unittest.TestCase):
    def test_episode_in_container_and_journal_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret_dir, out_dir = Path(tmp, "sec"), Path(tmp, "out")
            secret_dir.mkdir()
            out_dir.mkdir()
            secret_file = secret_dir / "secret"
            secret_file.write_text("container-secret\n", encoding="utf-8")
            command = ["docker", "run", "-i", "--rm", "--network", "none",
                       "-v", f"{secret_dir}:/run/auditgym:ro", "-v", f"{out_dir}:/out", IMAGE,
                       "python", "-m", "auditgym", "serve", "--secret-file", "/run/auditgym/secret",
                       "--results", "/out/results.jsonl"]
            with Env(mode="eval", server_command=command) as env:
                result = run_episode(env, make_policy("solver_policy"), seed=31337,
                                     config={"interaction": "I1"})
            self.assertEqual(result["reward"], 0.0)  # eval mode: the policy never sees the reward
            journal = out_dir / "results.jsonl"
            self.assertEqual(len(journal.read_text(encoding="utf-8").splitlines()), 1)
            verified = subprocess.run(
                [sys.executable, "-m", "auditgym", "verify-journal", str(journal),
                 "--secret-file", str(secret_file)],
                cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)


if __name__ == "__main__":
    unittest.main()
