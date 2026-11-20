"""Signed evaluator journal: an HMAC-SHA256 chain over the lines of runs/results.jsonl (audit A3).

With a journal key (auditgym/secret.py: $AUDITGYM_JOURNAL_KEY_FILE, $AUDITGYM_JOURNAL_KEY, or
derived from the evaluator secret) every line gets three more keys:

    seq    0, 1, 2, ... over the whole file
    prev   the `mac` of the line before it ("0" * 64 for the first line of the file)
    mac    HMAC-SHA256(key, the line without `mac` as canonical JSON), hex

Before each write the evaluator reads the tail of the file and continues the chain from its last
line, so sessions that append to one journal one after another form one chain. `python -m
auditgym verify-journal PATH` checks every signature and link and names the first bad line: an
edited, inserted, reordered or deleted line is caught, and so is a line appended without the key.
What a chain cannot show: lines cut from the END of the file (the shorter file is a valid chain),
and two evaluators writing to one file at the same moment (their lines fork the chain; give each
its own file). Without a key nothing is signed and verify-journal says so.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os

GENESIS = "0" * 64
TAIL_BYTES = 1 << 16
SIGNED_KEYS = ("seq", "prev", "mac")


def canonical(line: dict) -> bytes:
    body = {key: value for key, value in line.items() if key != "mac"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def mac_of(key: bytes, line: dict) -> str:
    return hmac.new(key, canonical(line), hashlib.sha256).hexdigest()


def chain_state(tail: bytes) -> tuple[str, int]:
    """(prev, seq) for the next line, from the last bytes of the journal. A journal that is empty,
    or whose last line is not a signed line, starts a new chain (verify-journal reports the break
    when there are lines before it)."""
    lines = tail.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines or len(tail) >= TAIL_BYTES and len(lines) == 1:
        return GENESIS, 0  # empty, or one line longer than the tail window: no usable link
    try:
        last = json.loads(lines[-1].decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return GENESIS, 0
    if (isinstance(last, dict) and isinstance(last.get("mac"), str) and isinstance(last.get("seq"), int)
            and not isinstance(last.get("seq"), bool)):
        return last["mac"], last["seq"] + 1
    return GENESIS, 0


def read_tail_fd(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    start = max(0, size - TAIL_BYTES)
    return os.pread(descriptor, size - start, start)


def read_tail_path(path) -> bytes:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            return handle.read()
    except FileNotFoundError:
        return b""


def seal(key: bytes, line: dict, tail: bytes) -> dict:
    prev, seq = chain_state(tail)
    sealed = {**{k: v for k, v in line.items() if k not in SIGNED_KEYS}, "seq": seq, "prev": prev}
    sealed["mac"] = mac_of(key, sealed)
    return sealed


def verify_lines(lines: list[str], key: bytes | None) -> tuple[str, int | None, str]:
    """Check a journal. Returns (status, line number or None, message):
    "ok" | "bad" (line number of the first bad line) | "unsigned" | "no_key"."""
    parsed = []
    for number, text in enumerate(lines, 1):
        try:
            row = json.loads(text)
        except (ValueError, RecursionError):
            return "bad", number, "not a JSON line"
        if not isinstance(row, dict):
            return "bad", number, "not a JSON object"
        parsed.append(row)
    signed = [row for row in parsed if "mac" in row]
    if not signed:
        return "unsigned", None, f"journal is not signed ({len(parsed)} lines): nothing to verify"
    if key is None:
        return "no_key", None, "journal is signed but no journal key is configured: cannot verify"
    prev, seq = GENESIS, 0
    for number, row in enumerate(parsed, 1):
        if not all(name in row for name in SIGNED_KEYS):
            return "bad", number, "line is not signed"
        if not isinstance(row["mac"], str) or not hmac.compare_digest(row["mac"], mac_of(key, row)):
            return "bad", number, "bad signature (line changed or written without the key)"
        if row["prev"] != prev or row["seq"] != seq:
            return "bad", number, "broken chain (a line before it was removed, inserted or reordered)"
        prev, seq = row["mac"], seq + 1
    return "ok", None, (f"journal OK: {len(parsed)} signed lines, chain intact "
                        f"(lines cut from the end of the file cannot be detected)")


def main(argv: list[str]) -> int:
    """python -m auditgym verify-journal PATH [--secret-file F]: exit 0 valid, 1 bad line,
    2 nothing to verify (unsigned journal, no key, unreadable file)."""
    import argparse
    import sys

    from .secret import SecretError, journal_key, read_secret

    parser = argparse.ArgumentParser(prog="python -m auditgym verify-journal")
    parser.add_argument("path")
    parser.add_argument("--secret-file", default=None,
                        help="the evaluator secret file (else the journal key or secret from the environment)")
    arguments = parser.parse_args(argv)
    try:
        secret = read_secret(secret_file=arguments.secret_file)
        key = journal_key(secret=secret)
    except SecretError as error:
        print(f"verify-journal: cannot use the key ({error.code})", file=sys.stderr)
        return 2
    try:
        with open(arguments.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        print("verify-journal: cannot read the journal", file=sys.stderr)
        return 2
    status, number, message = verify_lines(lines, key)
    if status == "bad":
        print(f"verify-journal: line {number}: {message}")
        return 1
    print(f"verify-journal: {message}")
    return 0 if status == "ok" else 2
