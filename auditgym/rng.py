"""Deterministic random streams.

The stream is SHA-256 in counter mode, so the same seed gives the same values on every
Python version, platform and PYTHONHASHSEED. Every part of the generator takes a stream as
an argument; nothing in the package may use the standard random module, UUIDs, the OS
entropy source or the clock (tools/lint_determinism.py enforces this).
"""
from __future__ import annotations

import hashlib
from typing import MutableSequence, Sequence, TypeVar

T = TypeVar("T")


class Rng:
    """A named random stream. `child(label)` derives an independent stream."""

    def __init__(self, seed: int | str, label: str = "root") -> None:
        self._key = hashlib.sha256(f"{seed}\x1f{label}".encode("utf-8")).digest()
        self._counter = 0

    def child(self, label: str) -> "Rng":
        # Derived from the key only, so drawing from the parent does not shift the child.
        return Rng(self._key.hex(), label)

    def u64(self) -> int:
        block = hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
        self._counter += 1
        return int.from_bytes(block[:8], "big")

    def below(self, n: int) -> int:
        """Uniform integer in [0, n), without modulo bias."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            value = self.u64()
            if value < limit:
                return value % n

    def randint(self, low: int, high: int) -> int:
        """Uniform integer in [low, high], both ends included."""
        if high < low:
            raise ValueError("high must not be below low")
        return low + self.below(high - low + 1)

    def chance(self, numerator: int, denominator: int) -> bool:
        """True with probability numerator/denominator, in exact integer arithmetic."""
        return self.below(denominator) < numerator

    def choice(self, items: Sequence[T]) -> T:
        if not items:
            raise ValueError("cannot choose from an empty sequence")
        return items[self.below(len(items))]

    def shuffle(self, items: MutableSequence[T]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def sample(self, items: Sequence[T], k: int) -> list[T]:
        if k < 0 or k > len(items):
            raise ValueError("sample size out of range")
        pool = list(items)
        self.shuffle(pool)
        return pool[:k]

    def token(self, length: int, alphabet: str = "0123456789abcdefghjkmnpqrstvwxyz") -> str:
        """An opaque identifier that carries no order or position information."""
        return "".join(alphabet[self.below(len(alphabet))] for _ in range(length))
