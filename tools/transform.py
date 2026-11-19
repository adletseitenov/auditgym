"""External "harmless transformation" of an episode directory (CASE §06, §10: reorder and rename).

    python -m tools.transform SRC DST --seed N [--no-shuffle] [--no-rename]
                              [--id-style token|numeric|hex|prefixed] [--keep-episode-id]
                              [--unsorted-truth]

This is what an organizer's script may do to an episode: it knows the file formats of
docs/CONTRACTS.md and nothing else. It does not import the generator, the surface or the
storage module; it reads and writes the files itself.

- Row order of every table is shuffled: public ledger.csv, invoices.jsonl, payments.jsonl,
  fx.csv and the private pre_ledger.csv, pre_invoices.jsonl, pre_payments.jsonl (the last two
  are optional: the tool works with or without them). `line_items` and `invoice_refs` are
  shuffled inside each document; truth issues and evidence sets are shuffled too.
- Every id is renamed consistently across public and private files: row_id, doc_id,
  invoice_id, payment_id, entity_id; in truth.json affected_ids and evidence_sets; in
  provenance.json the id_map values and any id inside steps (params, before, after). fx keys
  ("date|SRC|TGT") are content, not ids, and stay.
- episode_id: renamed by default (a new "ep_" + 12 hex), because it is an id the agent echoes
  back; `--keep-episode-id` keeps it. The environment reads it from the manifest only (cursors
  are derived from it), so either choice works. With --no-rename it is always kept.
- truth.json stays valid per contract (affected_ids sorted) unless `--unsorted-truth`, which
  leaves affected_ids in shuffled order to prove nobody depends on the stored order.
- DST/mapping.json maps every old id to its new id (and the episode id) for scoring.

Randomness comes only from auditgym.rng.Rng(seed); the same SRC and arguments give the same
bytes in DST.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys

from auditgym.rng import Rng

ID_STYLES = ("token", "numeric", "hex", "prefixed")
MAPPING_FORMAT = "auditgym-transform-1"

# Known files: relative path -> kind. Optional ones may be missing.
LEDGER_FILES = ("public/ledger.csv", "private/pre_ledger.csv")
INVOICE_FILES = ("public/invoices.jsonl", "private/pre_invoices.jsonl")
PAYMENT_FILES = ("public/payments.jsonl", "private/pre_payments.jsonl")
FX_FILES = ("public/fx.csv",)
JSON_FILES = ("public/manifest.json", "public/policy.json", "private/truth.json", "private/provenance.json")
REQUIRED = ("public/manifest.json", "public/policy.json", "public/ledger.csv", "public/invoices.jsonl",
            "public/payments.jsonl", "public/fx.csv", "private/truth.json", "private/provenance.json",
            "private/pre_ledger.csv")
KNOWN = set(LEDGER_FILES + INVOICE_FILES + PAYMENT_FILES + FX_FILES + JSON_FILES)

# Field -> namespace of the id it holds (used by the "prefixed" style).
LEDGER_IDS = {"row_id": "row", "entity_id": "ent", "invoice_id": "inv", "payment_id": "pay"}
INVOICE_IDS = {"doc_id": "doc", "invoice_id": "inv", "entity_id": "ent"}
PAYMENT_IDS = {"doc_id": "doc", "payment_id": "pay"}
PREFIXES = {"row": "row_", "doc": "doc_", "inv": "inv_", "pay": "pay_", "ent": "ent_", "x": "id_"}


class TransformError(Exception):
    pass


# --- file formats (docs/CONTRACTS.md §1-§3) ----------------------------------------------------
def _read_csv(text: str) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(text))
    rows = [dict(row) for row in reader]
    return list(reader.fieldnames or []), rows


def _write_csv(columns: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _read_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_jsonl(records: list) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _write_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1) + "\n"


# --- id collection ------------------------------------------------------------------------------
def _collect(files: dict[str, object]) -> dict[str, str]:
    """Every id in the episode -> its namespace. Table fields first, so they decide the namespace."""
    namespace: dict[str, str] = {}

    def add(value, kind: str) -> None:
        if isinstance(value, str) and value != "":
            namespace.setdefault(value, kind)

    for path in LEDGER_FILES:
        if path in files:
            for row in files[path][1]:
                for field, kind in LEDGER_IDS.items():
                    add(row.get(field), kind)
    for path in INVOICE_FILES:
        for record in files.get(path, []):
            for field, kind in INVOICE_IDS.items():
                add(record.get(field), kind)
    for path in PAYMENT_FILES:
        for record in files.get(path, []):
            for field, kind in PAYMENT_IDS.items():
                add(record.get(field), kind)
            for ref in record.get("invoice_refs") or []:
                add(ref, "inv")
    fx_keys = _fx_keys(files)
    for issue in files["private/truth.json"].get("issues", []):
        for value in issue.get("affected_ids", []):
            add(value, "doc")
        for group in issue.get("evidence_sets", []):
            for value in group:
                if value not in fx_keys:
                    add(value, "x")
    for key, value in sorted(files["private/provenance.json"].get("id_map", {}).items()):
        if value not in fx_keys:
            add(value, "doc" if str(key).startswith("D:") else "x")
    return namespace


def _fx_keys(files: dict) -> set[str]:
    return {f"{row['date']}|{row['source_currency']}|{row['target_currency']}" for row in files["public/fx.csv"][1]}


def _new_id(rng: Rng, style: str, kind: str) -> str:
    if style == "token":
        return rng.token(10)
    if style == "numeric":
        return str(10 ** 9 + rng.below(9 * 10 ** 9))
    if style == "hex":
        return rng.token(12, "0123456789abcdef")
    return PREFIXES[kind] + rng.token(8)


def build_mapping(namespace: dict[str, str], rng: Rng, style: str) -> dict[str, str]:
    """old -> new, a bijection; the draw order is a shuffle, so a new id says nothing about the old one."""
    old = sorted(namespace)
    rng.shuffle(old)
    taken = set(namespace)
    mapping: dict[str, str] = {}
    for value in old:
        while True:
            candidate = _new_id(rng, style, namespace[value])
            if candidate not in taken:
                taken.add(candidate)
                mapping[value] = candidate
                break
    return dict(sorted(mapping.items()))


# --- rewriting ----------------------------------------------------------------------------------
def _walk(value, mapping: dict[str, str]):
    """Every string value (not dict keys) that is a known id is renamed."""
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_walk(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _walk(item, mapping) for key, item in value.items()}
    return value


def _rename_fields(record: dict, fields, mapping: dict[str, str]) -> dict:
    out = dict(record)
    for field in fields:
        value = out.get(field)
        if isinstance(value, str) and value in mapping:
            out[field] = mapping[value]
    return out


def transform(src, dst, seed: int, shuffle: bool = True, rename: bool = True, id_style: str = "token",
              keep_episode_id: bool = False, unsorted_truth: bool = False) -> dict:
    """Write the transformed episode to `dst` and return the mapping document (also in dst/mapping.json)."""
    if id_style not in ID_STYLES:
        raise TransformError(f"unknown id style {id_style!r}")
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    for relative in REQUIRED:
        if not (src / relative).is_file():
            raise TransformError(f"missing {relative}")
    if dst.exists() and any(dst.iterdir()):
        raise TransformError("destination is not empty")

    files: dict[str, object] = {}
    for relative in LEDGER_FILES + FX_FILES:
        if (src / relative).is_file():
            files[relative] = _read_csv((src / relative).read_text(encoding="utf-8"))
    for relative in INVOICE_FILES + PAYMENT_FILES:
        if (src / relative).is_file():
            files[relative] = _read_jsonl((src / relative).read_text(encoding="utf-8"))
    for relative in JSON_FILES:
        files[relative] = json.loads((src / relative).read_text(encoding="utf-8"))
    # Only public/ and private/ belong to the episode; notes such as a README next to them are not copied.
    extra = sorted(path.relative_to(src).as_posix() for folder in ("public", "private")
                   for path in (src / folder).rglob("*")
                   if path.is_file() and path.relative_to(src).as_posix() not in KNOWN)
    for relative in extra:
        if relative.startswith("public/"):
            raise TransformError(f"unknown public file {relative}: its ids cannot be renamed safely")

    root = Rng(seed, "transform")
    namespace = _collect(files)
    mapping = build_mapping(namespace, root.child("ids"), id_style) if rename else {key: key for key in namespace}
    old_episode = files["public/manifest.json"]["episode_id"]
    new_episode = old_episode
    if rename and not keep_episode_id:
        new_episode = "ep_" + root.child("episode").token(12, "0123456789abcdef")
    order = root.child("order")

    def shuffled(items: list) -> list:
        items = list(items)
        if shuffle:
            order.shuffle(items)
        return items

    out: dict[str, str] = {}
    for relative in LEDGER_FILES:
        if relative in files:
            columns, rows = files[relative]
            out[relative] = _write_csv(columns, shuffled([_rename_fields(row, LEDGER_IDS, mapping) for row in rows]))
    columns, rows = files["public/fx.csv"]
    out["public/fx.csv"] = _write_csv(columns, shuffled(rows))
    for relative in INVOICE_FILES:
        if relative in files:
            records = []
            for record in files[relative]:
                record = _rename_fields(record, INVOICE_IDS, mapping)
                if isinstance(record.get("line_items"), list):
                    record["line_items"] = shuffled(record["line_items"])
                records.append(record)
            out[relative] = _write_jsonl(shuffled(records))
    for relative in PAYMENT_FILES:
        if relative in files:
            records = []
            for record in files[relative]:
                record = _rename_fields(record, PAYMENT_IDS, mapping)
                if isinstance(record.get("invoice_refs"), list):
                    record["invoice_refs"] = shuffled([mapping.get(ref, ref) for ref in record["invoice_refs"]])
                records.append(record)
            out[relative] = _write_jsonl(shuffled(records))

    manifest = dict(files["public/manifest.json"], episode_id=new_episode)
    out["public/manifest.json"] = _write_json(manifest)
    out["public/policy.json"] = _write_json(files["public/policy.json"])

    truth = dict(files["private/truth.json"], episode_id=new_episode)
    issues = []
    for issue in truth.get("issues", []):
        issue = dict(issue)
        affected = [mapping.get(value, value) for value in issue.get("affected_ids", [])]
        issue["affected_ids"] = shuffled(affected) if unsorted_truth else sorted(affected)
        issue["evidence_sets"] = shuffled([shuffled([mapping.get(value, value) for value in group])
                                           for group in issue.get("evidence_sets", [])])
        issues.append(issue)
    truth["issues"] = shuffled(issues)
    out["private/truth.json"] = _write_json(truth)

    provenance = dict(files["private/provenance.json"])
    provenance = {key: (value if key in ("seed", "config", "id_map") else _walk(value, mapping))
                  for key, value in provenance.items()}
    if isinstance(provenance.get("id_map"), dict):
        provenance["id_map"] = {key: mapping.get(value, value) for key, value in provenance["id_map"].items()}
    out["private/provenance.json"] = _write_json(provenance)

    for relative in extra:  # private files of a newer generator: rename whatever ids they hold
        text = (src / relative).read_text(encoding="utf-8")
        if relative.endswith(".json"):
            out[relative] = _write_json(_walk(json.loads(text), mapping))
        elif relative.endswith(".jsonl"):
            out[relative] = _write_jsonl(shuffled([_walk(record, mapping) for record in _read_jsonl(text)]))
        elif relative.endswith(".csv"):
            columns, rows = _read_csv(text)
            id_columns = [c for c in columns if c == "id" or c.endswith("_id")]
            out[relative] = _write_csv(columns, shuffled([_rename_fields(row, id_columns, mapping) for row in rows]))
        elif any(value in text for value in mapping if value != mapping[value]):
            raise TransformError(f"unknown private file {relative} holds ids")
        else:
            out[relative] = text

    document = {"format": MAPPING_FORMAT, "transform_seed": seed, "shuffled": shuffle, "renamed": rename,
                "id_style": id_style if rename else None,
                "episode_id": {"old": old_episode, "new": new_episode}, "ids": mapping}
    out["mapping.json"] = _write_json(document)
    for relative, content in sorted(out.items()):
        path = dst / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    return document


def load_mapping(directory) -> dict[str, str]:
    """old id -> new id, the episode id included."""
    document = json.loads((pathlib.Path(directory) / "mapping.json").read_text(encoding="utf-8"))
    mapping = dict(document["ids"])
    mapping[document["episode_id"]["old"]] = document["episode_id"]["new"]
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.transform", description=__doc__.split("\n\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-rename", action="store_true")
    parser.add_argument("--id-style", choices=ID_STYLES, default="token")
    parser.add_argument("--keep-episode-id", action="store_true")
    parser.add_argument("--unsorted-truth", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        document = transform(arguments.src, arguments.dst, arguments.seed, shuffle=not arguments.no_shuffle,
                             rename=not arguments.no_rename, id_style=arguments.id_style,
                             keep_episode_id=arguments.keep_episode_id, unsorted_truth=arguments.unsorted_truth)
    except (TransformError, OSError, ValueError, KeyError) as exc:
        print(f"transform: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"transform: {len(document['ids'])} ids, episode {document['episode_id']['new']}, "
          f"shuffled={document['shuffled']}, style={document['id_style']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
