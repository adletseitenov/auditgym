"""Money and time helpers shared by the generator, the verifier and the solver.

Money is an int in minor units, rates are Decimal. Nothing here reads the clock.
"""
from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_EVEN, Decimal


def parse_offset(offset: str) -> dt.timezone:
    """'+05:00' -> a fixed-offset timezone."""
    sign = -1 if offset[0] == "-" else 1
    hours, minutes = int(offset[1:3]), int(offset[4:6])
    return dt.timezone(sign * dt.timedelta(hours=hours, minutes=minutes))


def parse_time(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


def local_date(timestamp: str, offset: str) -> str:
    """Calendar date of an ISO timestamp in the company's fixed offset."""
    return parse_time(timestamp).astimezone(parse_offset(offset)).date().isoformat()


def in_period(timestamp: str, period_start: str, period_end: str) -> bool:
    """Half-open interval [period_start, period_end)."""
    return parse_time(period_start) <= parse_time(timestamp) < parse_time(period_end)


def convert(amount_minor: int, rate: Decimal, exponent_source: int, exponent_base: int) -> int:
    """policy.fx_formula: round(amount_minor * 10^(exp_base - exp_src) * rate), half-even."""
    scaled = Decimal(amount_minor) * (Decimal(10) ** (exponent_base - exponent_source)) * rate
    return int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def fx_key(date: str, source: str, target: str) -> str:
    return f"{date}|{source}|{target}"
