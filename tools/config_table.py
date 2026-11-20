"""Prints the limits of a configuration and how many invoices N errors need (docs/USAGE.md, «Конфигурация»).

    python3 -m tools.config_table

The numbers come from auditgym/limits.py and from auditgym.config.normalize_config itself: a row is
`accepted` only when normalize_config accepts it, so the table cannot drift from the validation.
"""
from __future__ import annotations

from auditgym import config as cfg
from auditgym import limits

SPREAD = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")


def errors_for(total: int) -> dict:
    """`total` errors spread over the four types, round robin."""
    return {name: total // 4 + (1 if index < total % 4 else 0) for index, name in enumerate(SPREAD)}


def row(total: int) -> tuple:
    errors = errors_for(total)
    default = cfg.normalize_config({"errors": errors})["size"]["invoices"] if total * 8 <= limits.MAX_INVOICES else None
    need = max(limits.MIN_INVOICES, limits.INVOICES_PER_ERROR * total)
    try:
        cfg.normalize_config({"errors": errors, "size": {"invoices": [need, limits.MAX_INVOICES]}})
        accepted = need <= limits.MAX_INVOICES
    except cfg.ConfigError as error:
        accepted = f"no ({error.code})"
    return total, need, default, accepted


def main() -> None:
    print("limits (auditgym/limits.py):")
    for name in ("MAX_STEPS", "PAGE_SIZE", "MAX_FINDINGS", "MAX_EVIDENCE_IDS", "MAX_EXPLANATION_CHARS",
                 "MAX_MESSAGE_BYTES", "MAX_LEDGER_ROWS", "MAX_PAGES", "INVOICES_PER_ERROR", "MIN_INVOICES",
                 "MAX_INVOICES", "MAX_BACKGROUND_PAIRS", "MAX_PERIOD_MONTHS", "MAX_FOREIGN_CURRENCIES"):
        print(f"  {name} = {getattr(limits, name)}")
    print(f"most errors one config can hold: {limits.MAX_INVOICES // limits.INVOICES_PER_ERROR} "
          f"(MAX_INVOICES // INVOICES_PER_ERROR); an answer holds at most {limits.MAX_FINDINGS} findings")
    print("errors | invoices needed (>=) | default size.invoices when size is omitted | accepted")
    for total in (1, 2, 4, 6, 8, 12, 16, 20, 24, 25, 26):
        n, need, default, accepted = row(total)
        print(f"{n:>6} | {need:>20} | {str(default):>42} | {accepted}")


if __name__ == "__main__":
    main()
