"""The answer rules of the first observation (`task.answer_rules`): what the judge expects per error type.

A constant block: the same in every episode, built from no episode data. It states what the judge
checks (auditgym/matcher.py: same type, same set of affected ids, same currency, amount within
`policy.amount_tolerance_minor`, every evidence id exists, a full allowed evidence set is cited) and
which sets and amounts the generator's truth holds for each type (auditgym/inject.py). Without it a
correctly found error would still be an FP for an unavailable fact, such as the fx key of an
fx_error. tests/judge/test_answer_rules.py checks the text against the judge and the truth.

Evidence sets are written as lists of `role(subject)` tokens; `evidence_roles` says how each role
is found in the open records. Subjects: A and B are the two affected invoices of a duplicate,
`payment` and `invoice` are the two affected documents of an allocation_gap, `document` is the
single affected document.
"""
from __future__ import annotations

import copy

from .limits import MAX_EVIDENCE_IDS, MAX_FINDINGS

GENERAL = [
    "Вывод — TP, только если совпали все пять признаков: type; множество affected_ids (порядок не важен, "
    "лишних и недостающих ID нет); currency; claimed_amount_minor в допуске "
    "(|заявлено − ожидаемое| <= policy.amount_tolerance_minor[type]); evidence_ids содержат целиком хотя бы "
    "один из допустимых наборов своего типа (types.<тип>.evidence.one_of).",
    "Один вывод — одна ошибка; повтор уже засчитанного вывода — отдельный FP (повторы выводов не "
    "удаляются). affected_ids и evidence_ids — множества точных ID: порядок и повтор одного ID внутри "
    "списка множество не меняют. Один документ может входить в выводы разных типов.",
    "Лишние существующие и доступные ID в evidence_ids допустимы, пока tolerances.evidence_superset_allowed "
    "= true (иначе набор должен совпасть точно): нужно полностью включить хотя бы один допустимый набор. "
    "Несуществующий или недоступный ID делает вывод несопоставимым (FP), даже если минимальный набор есть. "
    f"В списке не более {MAX_EVIDENCE_IDS} элементов (limits.max_evidence_ids; считаются до удаления "
    f"повторов: 11 элементов — неверная структура) и не более {MAX_FINDINGS} выводов "
    "(limits.max_findings). Достаточно одного набора из one_of; все четыре набора дубля вместе — шесть ID.",
    "ID: affected_ids — doc_id документов (таблицы invoices, payments); строки — row_id таблицы ledger; "
    "ключ курса — ID таблицы fx вида «дата|ИСТ|ЦЕЛЬ».",
]

EVIDENCE_ROLES = {
    "doc": "doc_id самого документа (счёта или платежа) из affected_ids.",
    "receivable_debit": "строка ledger: account receivable, debit_minor > 0, invoice_id = invoice_id счёта, "
                        "payment_id = null (проводка самого счёта).",
    "revenue_credit": "строка ledger: account revenue, credit_minor > 0, invoice_id = invoice_id счёта, "
                      "payment_id = null (проводка самого счёта).",
    "cash_row": "строка ledger: account cash, payment_id = payment_id платежа.",
    "unallocated_row": "строка ledger: account unallocated, credit_minor > 0, payment_id = payment_id платежа.",
    "receivable_credit": "строка ledger: account receivable, credit_minor > 0, payment_id = payment_id "
                         "платежа, invoice_id = invoice_id счёта (закрытие счёта этим платежом).",
    "rate_key": "ID из таблицы fx «дата|ИСТ|ЦЕЛЬ»: дата — дата документа (issued_at счёта, timestamp "
                "платежа) в поясе policy.timezone, ИСТ — валюта документа, ЦЕЛЬ — policy.base_currency.",
}


def _duplicate_sets() -> list[list[str]]:
    return [["doc(A)", "doc(B)", f"{x}(A)", f"{y}(B)"]
            for x in ("receivable_debit", "revenue_credit") for y in ("receivable_debit", "revenue_credit")]


TYPES = {
    "duplicate_invoice": {
        "affected_ids": "doc_id обоих счетов-дублей (ровно два): совпадают контрагент, валюта, итог и позиции "
                        "с периодом услуги, оба проведены в ledger; дата выставления в сравнение не входит.",
        "claimed_amount_minor": "total_minor счёта (у обоих одинаков), в минимальных единицах валюты счёта; "
                                "допуск policy.amount_tolerance_minor.duplicate_invoice.",
        "currency": "валюта счёта (invoices.currency), не базовая валюта.",
        "evidence": {
            "one_of": _duplicate_sets(),
            "subjects": "A и B — два счёта из affected_ids; строка каждого счёта — одна из двух его собственных "
                        "проводок (Дт дебиторки или Кт выручки), любое из четырёх сочетаний.",
        },
    },
    "allocation_gap": {
        "affected_ids": "doc_id платежа и doc_id счёта, на который платёж ссылается (ровно два): "
                        "часть платежа лежит на unallocated, а счёт остаётся открытым.",
        "claimed_amount_minor": "credit_minor строки unallocated платежа, как записано в ledger (без "
                                "пересчёта), в базовой валюте; допуск policy.amount_tolerance_minor.allocation_gap.",
        "currency": "policy.base_currency (валюта ledger), не валюта платежа.",
        "evidence": {
            "one_of": [["doc(payment)", "doc(invoice)", "unallocated_row(payment)", "receivable_debit(invoice)",
                        "receivable_credit(payment)"]],
            "subjects": "payment и invoice — два документа из affected_ids; набор один, нужны все пять ID.",
        },
    },
    "fx_error": {
        "affected_ids": "doc_id одного документа (счёта или платежа в иностранной валюте), чья проводка "
                        "посчитана по неверному курсу.",
        "claimed_amount_minor": "модуль разницы между отражённой суммой и правильной: |строка − "
                                "round(amount × 10^(e_base − e_src) × курс даты документа)| по policy.fx_formula "
                                "и policy.rounding; строка — Дт дебиторки (она же Кт выручки) счёта или строка "
                                "cash платежа; в базовой валюте; допуск policy.amount_tolerance_minor.fx_error.",
        "currency": "policy.base_currency, не валюта документа.",
        "evidence": {
            "one_of": [["doc(document)", "receivable_debit(document)", "rate_key(document)"],
                       ["doc(document)", "revenue_credit(document)", "rate_key(document)"],
                       ["doc(document)", "cash_row(document)", "rate_key(document)"]],
            "subjects": "document — документ из affected_ids; для счёта подходят первые два набора, для платежа "
                        "— третий. Ключ курса обязателен в каждом наборе: без него вывод — FP.",
        },
    },
    "period_cutoff": {
        "affected_ids": "doc_id одного счёта, датированного вне периода [period_start, period_end) в поясе "
                        "policy.timezone, но проведённого в ledger.",
        "claimed_amount_minor": "total_minor счёта, в минимальных единицах валюты счёта; допуск "
                                "policy.amount_tolerance_minor.period_cutoff.",
        "currency": "валюта счёта (invoices.currency), не базовая валюта.",
        "evidence": {
            "one_of": [["doc(document)", "receivable_debit(document)"], ["doc(document)", "revenue_credit(document)"]],
            "subjects": "document — счёт из affected_ids; достаточно одной из двух его проводок.",
        },
    },
}


def answer_rules() -> dict:
    """A fresh copy of the rules block (callers may not alter the constants)."""
    return copy.deepcopy({"general": GENERAL, "evidence_roles": EVIDENCE_ROLES, "types": TYPES})
