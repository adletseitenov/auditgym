"""Evaluator-side secrets: the generation secret and the journal key (docs/DECISIONS.md H).

Sources, first match wins:

    generation secret   --secret-file / Env(secret_file=...)  >  $AUDITGYM_SECRET_FILE  >  $AUDITGYM_SECRET
    journal key         $AUDITGYM_JOURNAL_KEY_FILE  >  $AUDITGYM_JOURNAL_KEY  >  derived from the secret

A file is the recommended form: the policy gets the path at most, and a file readable only by the
evaluator's user (or mounted only into the evaluator's container) cannot be read by a policy that
runs as another user. A value in $AUDITGYM_SECRET is in the environment of every process started
from that shell, a policy included: that is a leak by configuration, not something code can undo.

Values are never printed, logged or put into an error text; errors carry a fixed code only.
"""
from __future__ import annotations

import hashlib
import hmac
import os

SECRET_ENV = "AUDITGYM_SECRET"
SECRET_FILE_ENV = "AUDITGYM_SECRET_FILE"
JOURNAL_KEY_ENV = "AUDITGYM_JOURNAL_KEY"
JOURNAL_KEY_FILE_ENV = "AUDITGYM_JOURNAL_KEY_FILE"
_JOURNAL_KEY_LABEL = b"auditgym-journal-key-v1"
MAX_SECRET_BYTES = 1 << 16


class SecretError(Exception):
    """A configured secret file cannot be used. `code` is fixed text without the path or value."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _read_file(path) -> str:
    """The text of a secret file without the trailing line break. Missing, unreadable, too large,
    not UTF-8 or empty is an error: a closed run must not silently fall back to no secret."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_SECRET_BYTES + 1)
    except (OSError, TypeError, ValueError):
        raise SecretError("secret_file_unreadable") from None
    if len(data) > MAX_SECRET_BYTES:
        raise SecretError("secret_file_too_large")
    try:
        text = data.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError:
        raise SecretError("secret_file_not_utf8") from None
    if not text:
        raise SecretError("secret_file_empty")
    return text


def read_secret(environ=None, secret_file=None) -> str:
    """The generation secret ("" when none is configured)."""
    environ = os.environ if environ is None else environ
    if secret_file is not None:
        return _read_file(secret_file)
    if environ.get(SECRET_FILE_ENV, ""):
        return _read_file(environ[SECRET_FILE_ENV])
    return environ.get(SECRET_ENV, "")


def secret_source(environ=None) -> str:
    """Where the secret would come from: "file", "value" or "none" (the value is not read)."""
    environ = os.environ if environ is None else environ
    if environ.get(SECRET_FILE_ENV, ""):
        return "file"
    if environ.get(SECRET_ENV, ""):
        return "value"
    return "none"


def journal_key(environ=None, secret: str | None = None) -> bytes | None:
    """The HMAC key of the journal, or None (the journal is then written unsigned).

    `secret` is the generation secret already read by the evaluator; when it is None it is read
    from `environ`. A key derived from the secret is HMAC-SHA256(secret, label), so the journal key
    never equals the secret itself."""
    environ = os.environ if environ is None else environ
    if environ.get(JOURNAL_KEY_FILE_ENV, ""):
        return _read_file(environ[JOURNAL_KEY_FILE_ENV]).encode("utf-8")
    if environ.get(JOURNAL_KEY_ENV, ""):
        return environ[JOURNAL_KEY_ENV].encode("utf-8")
    if secret is None:
        secret = read_secret(environ)
    if not secret:
        return None
    return hmac.new(secret.encode("utf-8"), _JOURNAL_KEY_LABEL, hashlib.sha256).digest()
