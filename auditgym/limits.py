"""Limits of one episode and one answer: one constant per limit (docs/CONTRACTS.md, docs/DECISIONS.md).

config, verify, the solver, the actions and the findings parser read these values from here so a
limit is never written twice. Nothing here depends on anything else in the package.
"""
from __future__ import annotations

# Answer
MAX_FINDINGS = 30                 # findings in one submit; also the most issues one episode may hold
MAX_EVIDENCE_IDS = 10             # evidence ids per finding, and ids per truth evidence set
MAX_EXPLANATION_CHARS = 1000
MAX_MESSAGE_BYTES = 256 * 1024    # docs/AUDIT.md R2: 30 x 10 x 1 000 Cyrillic characters must fit
MAX_JSON_DEPTH = 64               # nesting of one message; explicit so the result does not depend on
                                  # the Python version's json recursion behaviour

# Episode
MAX_STEPS = 100
PAGE_SIZE = 50
MAX_PAGES = 60                    # pages for a full read of every table
MAX_LEDGER_ROWS = 2000
INVOICES_PER_ERROR = 8            # density: at most one issue per this many invoices (DECISIONS A3)
MIN_INVOICES, MAX_INVOICES = 10, 200
MIN_BACKGROUND_PAIRS, MAX_BACKGROUND_PAIRS = 0, 1000
MAX_PERIOD_MONTHS = 3
MAX_FOREIGN_CURRENCIES = 3
MIN_YEAR, MAX_YEAR = 2000, 2100

# Ledger budget: the clean story posts at most ROWS_PER_INVOICE rows per invoice, a background
# pair two rows, and one issue at most ROWS_PER_ISSUE extra rows (a paid duplicate copy: 2 + 3).
ROWS_PER_INVOICE = 5
ROWS_PER_ISSUE = 5


def pages(count: int) -> int:
    """Pages of PAGE_SIZE needed to read `count` records (an empty table still costs no page)."""
    return -(-count // PAGE_SIZE)


def json_depth_exceeds(value: object, limit: int = MAX_JSON_DEPTH) -> bool:
    """True when lists/dicts nest deeper than `limit`. Iterative, so it never recurses itself."""
    stack = [(value, 0)]
    seen = 0
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            children = item.values()
        elif isinstance(item, (list, tuple)):
            children = item
        else:
            continue
        if depth >= limit:
            return True
        seen += 1
        if seen > 1_000_000:
            return True
        stack.extend((child, depth + 1) for child in children)
    return False
