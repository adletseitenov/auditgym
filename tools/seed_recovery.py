"""How easy is it to recover the seed of an episode? (README and docs/DESIGN.md, «Ограничения».)

    python3 -m tools.seed_recovery [--bits 8] [--seed 100]

An agent that can import the package and knows the configuration sees `manifest.episode_id` in the first
observation. The episode id is a hash of the open records, so it identifies the seed: try every small seed
until the id matches. The script does exactly that, with the plain generator and with a secret mixed in
(`generate(seed, config, secret=...)`), and prints the cost. Nothing here is a defence; it measures the
attack that docs/AUDIT.md L1 describes.
"""
from __future__ import annotations

import argparse
import time

from auditgym import Env
from auditgym.generate import generate


def recover(target: str, bits: int, secret: str = "") -> tuple:
    """(seed or None, candidates tried, seconds)."""
    started = time.perf_counter()
    for tried, candidate in enumerate(range(2 ** bits), 1):
        if generate(candidate, None, secret)["public"]["manifest"]["episode_id"] == target:
            return candidate, tried, time.perf_counter() - started
    return None, 2 ** bits, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 -m tools.seed_recovery")
    parser.add_argument("--bits", type=int, default=8, help="search seeds below 2**bits")
    parser.add_argument("--seed", type=int, default=100, help="the hidden seed (must be below 2**bits)")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** args.bits:
        parser.error("--seed must be below 2**bits")
    with Env(mode="train") as env:
        target = env.reset(args.seed, None)[0]["manifest"]["episode_id"]
    found, tried, seconds = recover(target, args.bits)
    per_second = tried / seconds
    print(f"plain generator: seed {found} found after {tried} candidates in {seconds:.1f} s "
          f"({per_second:.0f} candidates/s, one core)")
    print(f"  at that speed: 2^20 seeds take {2 ** 20 / per_second / 3600:.1f} h, "
          f"2^32 seeds take {2 ** 32 / per_second / 86400 / 365:.1f} years (one core, full generation per candidate)")
    secret_target = generate(args.seed, None, "evaluator-secret")["public"]["manifest"]["episode_id"]
    found_secret, tried_secret, _ = recover(secret_target, min(args.bits, 6))
    print(f"with a secret mixed in: the id differs from the plain one: {secret_target != target}; "
          f"search over {tried_secret} seeds finds: {found_secret}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
