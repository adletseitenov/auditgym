"""Build the open train and dev episodes from configs/episodes.json.

Run: python -m tools.episodes --out episodes

Writes <out>/train/ep_01 .. ep_10 and <out>/dev/ep_01 .. ep_10, each with public/ and private/
sub-directories (docs/CONTRACTS.md §1). Each slot of the config file lists variants in order of
preference; the first one the generator supports is used and the skipped ones are printed, so
the set widens by itself when new error types land. Names carry no seed. Rebuilding gives
byte-identical files.

Also the evaluator-side config pools used by the report, the demo and the qa checks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN = ROOT / "configs" / "episodes.json"
SPLITS = ("train", "dev")

# Pools for measurements over many seeds. Unsupported entries are skipped with a note.
ERROR_POOL = [
    {"errors": {"duplicate_invoice": 2, "period_cutoff": 2}, "currencies": ["USD"]},
    {"errors": {"duplicate_invoice": 1, "period_cutoff": 2}, "currencies": ["USD", "EUR"], "timezone": "-11:00"},
    {"errors": {"duplicate_invoice": 3}, "currencies": ["CNY"]},
    {"errors": {"period_cutoff": 3}, "currencies": ["RUB"], "timezone": "+14:00"},
    None,
    {"errors": {"allocation_gap": 2, "fx_error": 2}, "currencies": ["USD"]},
    {"errors": {"fx_error": 3}, "currencies": ["EUR", "CNY"], "timezone": "+05:30"},
    {"errors": {"allocation_gap": 3}, "currencies": ["USD", "EUR"]},
    {"interaction": "I1", "errors": {"duplicate_invoice": 1}, "currencies": ["USD"]},
    {"interaction": "I2", "errors": {"fx_error": 1}, "currencies": ["EUR"]},
]
CLEAN_POOL = [
    {"errors": {}, "currencies": ["USD"]},
    {"errors": {}, "currencies": ["USD", "EUR"], "timezone": "-11:00"},
    {"errors": {}, "currencies": [], "timezone": "+00:00"},
    {"errors": {}, "currencies": ["CNY", "RUB"], "timezone": "+05:30"},
]
_SUPPORT: dict[str, str | None] = {}


def unsupported(config, seed: int = 0) -> str | None:
    """None when the generator builds this config, else the ConfigError code."""
    from auditgym.config import ConfigError
    from auditgym.generate import generate

    key = json.dumps([config, seed], sort_keys=True)
    if key not in _SUPPORT:
        try:
            generate(seed, config)
            _SUPPORT[key] = None
        except ConfigError as error:
            _SUPPORT[key] = error.code
    return _SUPPORT[key]


def supported_pool(pool: list, quiet: bool = False) -> list:
    out = []
    for config in pool:
        reason = unsupported(config)
        if reason is None:
            out.append(config)
        elif not quiet:
            print(f"note: skipped config {json.dumps(config, sort_keys=True)} ({reason})", file=sys.stderr)
    return out


def plan(path=PLAN, quiet: bool = False) -> list[dict]:
    """[{split, name, seed, config, note}] with the first supported variant of every slot."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = []
    for split in SPLITS:
        for number, slot in enumerate(data[split], 1):
            name = f"ep_{number:02d}"
            found = False
            for index, variant in enumerate(slot["variants"]):
                reason = unsupported(variant, slot["seed"])
                if reason is None:
                    found = True
                    break
                if not quiet:
                    print(f"note: {split}/{name}: variant {json.dumps(variant, sort_keys=True)} "
                          f"not supported ({reason}), trying the next", file=sys.stderr)
            if not found:
                raise SystemExit(f"{split}/{name}: no supported variant")
            out.append({"split": split, "name": name, "seed": slot["seed"], "config": variant,
                        "note": slot.get("note", ""), "preferred": index == 0})
    return out


def build(out, path=PLAN, quiet: bool = False) -> list[dict]:
    from auditgym.generate import generate
    from auditgym.storage import write_episode

    out = pathlib.Path(out)
    entries = plan(path, quiet=quiet)
    for split in SPLITS:
        target = out / split
        if target.exists():
            stray = [p.name for p in target.iterdir() if not p.name.startswith("ep_")]
            if stray:
                raise SystemExit(f"{target} holds files that are not episodes: {stray[:3]}")
            shutil.rmtree(target)
    for entry in entries:
        write_episode(generate(entry["seed"], entry["config"]), out / entry["split"] / entry["name"])
    return entries


def tree_bytes(root) -> dict[str, bytes]:
    root = pathlib.Path(root)
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="episodes")
    parser.add_argument("--config", default=str(PLAN))
    args = parser.parse_args(argv)
    entries = build(args.out, args.config)
    for entry in entries:
        mark = "" if entry["preferred"] else "  (fallback variant)"
        print(f"{entry['split']}/{entry['name']}: {entry['note']}{mark}")
    print(f"wrote {len(entries)} episodes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
