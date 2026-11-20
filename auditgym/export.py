"""python -m auditgym export --seed N [--config JSON | --config-file PATH] --out DIR

Writes one episode directory (public/ + private/, the layout of episodes/train/*) built the way the
evaluator builds it: the secret comes from --secret-file, else $AUDITGYM_SECRET_FILE, else
$AUDITGYM_SECRET, read by the same function as the evaluator (auditgym/secret.py). A jury that runs closed episodes with a secret gets the reference for its
own oracle here (generate(seed, config) without the secret builds a different episode). The directory
loads with Env.from_files(DIR). The secret is read, never printed and never written; private/ holds
the truth and the seed, keep it on the evaluator side.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from .limits import MAX_SEED_BITS


def build_episode(seed: int, config, environ=None, secret_file=None) -> dict:
    """The episode the evaluator builds for (seed, config) under the secret of `environ`
    (or of `secret_file`)."""
    from .generate import generate
    from .secret import read_secret

    environ = os.environ if environ is None else environ
    return generate(seed, config, secret=read_secret(environ, secret_file))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m auditgym export", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="configuration as a JSON object")
    source.add_argument("--config-file", help="path of a JSON file with the configuration")
    parser.add_argument("--out", required=True, help="directory to create (must be empty or missing)")
    parser.add_argument("--secret-file", default=None,
                        help="file with the evaluator secret (wins over $AUDITGYM_SECRET_FILE and $AUDITGYM_SECRET)")
    args = parser.parse_args(argv)

    def fail(message: str) -> int:
        print(f"export: {message}", file=sys.stderr)
        return 2

    if args.seed.bit_length() > MAX_SEED_BITS:
        return fail(f"seed must fit into {MAX_SEED_BITS} bits")
    config = None
    try:
        if args.config_file is not None:
            config = json.loads(pathlib.Path(args.config_file).read_text(encoding="utf-8-sig"))
        elif args.config is not None:
            config = json.loads(args.config)
    except (OSError, ValueError):
        return fail("the configuration is not readable JSON")
    out = pathlib.Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        return fail(f"{out} exists and is not an empty directory")

    from .config import ConfigError
    from .secret import SecretError
    from .storage import write_episode

    try:
        episode = build_episode(args.seed, config, secret_file=args.secret_file)
    except ConfigError as error:
        return fail(f"bad configuration ({error.code})")
    except SecretError as error:
        return fail(f"cannot use the secret ({error.code})")
    write_episode(episode, out)
    print(f"wrote episode {episode['public']['manifest']['episode_id']} to {out}")
    return 0
