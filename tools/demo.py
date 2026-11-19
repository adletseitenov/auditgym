"""Demo for the jury: one episode step by step, in Russian, then the same episode attacked.

Run: python -m tools.demo [--seed N] [--episode DIR]

The solver policy plays through a real Env process. This tool is evaluator-side: after the
episode it uses the truth to show the judge's breakdown. The plain-language reasons are built
from the open records only.
"""
from __future__ import annotations

import argparse
import sys

from tools import episodes as episodes_tool

DEMO_SIZE = {"invoices": [32, 40], "background_pairs": [40, 60]}
CANDIDATES = [
    {"errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
     "currencies": ["USD"], "size": DEMO_SIZE},
    {"errors": {"duplicate_invoice": 1, "period_cutoff": 1}, "currencies": ["USD"], "size": DEMO_SIZE},
]
NAMES = {"duplicate_invoice": "дубль счёта", "allocation_gap": "нераспределённый платёж",
         "fx_error": "ошибка курса", "period_cutoff": "чужой период"}
ACTIONS = {"list_records": "листаю", "get_record": "запись", "get_document": "документ",
           "calculator": "калькулятор", "submit": "отправляю ответ"}
TABLES = {"ledger": "книгу проводок", "invoices": "счета", "payments": "платежи", "fx": "курсы"}


def money(amount: int, currency: str, exponents: dict) -> str:
    exponent = exponents.get(currency, 2)
    sign = "−" if amount < 0 else ""
    units, minor = divmod(abs(amount), 10 ** exponent)
    text = f"{units:,}".replace(",", " ")
    return f"{sign}{text},{minor:0{exponent}d} {currency}" if exponent else f"{sign}{text} {currency}"


def short(stamp: str) -> str:
    return stamp[:16].replace("T", " ")


def _rows_by(ledger: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in ledger:
        if key == "invoice_id" and row.get("payment_id"):
            continue
        if row.get(key):
            out.setdefault(row[key], []).append(row)
    return out


def reason(found: dict, public: dict) -> str:
    """Why the solver flagged this, in plain words with the numbers, from the open records."""
    from auditgym.money import convert, fx_key, local_date

    policy = public["policy"]
    exp, base = policy["currency_exponents"], policy["base_currency"]
    docs = {r["doc_id"]: r for r in public["invoices"] + public["payments"]}
    by_invoice, by_payment = _rows_by(public["ledger"], "invoice_id"), _rows_by(public["ledger"], "payment_id")
    ids = found["affected_ids"]
    kind = found["type"]
    if kind == "duplicate_invoice":
        a, b = docs[ids[0]], docs[ids[1]]
        return (f"счета {ids[0]} и {ids[1]}: один клиент, одна сумма {money(a['total_minor'], a['currency'], exp)}, "
                f"те же позиции ({len(a['line_items'])} шт.), выставлены {short(a['issued_at'])} и {short(b['issued_at'])}; "
                "оба проведены в книге — выручка учтена дважды")
    if kind == "period_cutoff":
        doc = docs[ids[0]]
        date = doc.get("issued_at") or doc.get("timestamp")
        amount = doc.get("total_minor", doc.get("amount_minor"))
        rows = by_invoice.get(doc.get("invoice_id"), []) or by_payment.get(doc.get("payment_id"), [])
        posted = short(min(r["timestamp"] for r in rows)) if rows else "?"
        return (f"документ {ids[0]} на {money(amount, doc['currency'], exp)} датирован {short(date)}, "
                f"а период {policy['period_start'][:10]} — {policy['period_end'][:10]} (не включая); "
                f"проводка же стоит {posted}, внутри периода")
    if kind == "fx_error":
        doc = docs[ids[0]]
        date = doc.get("issued_at") or doc.get("timestamp")
        amount = doc.get("total_minor", doc.get("amount_minor"))
        day = local_date(date, policy["timezone"])
        rate = next((r["rate"] for r in public["fx"]
                     if fx_key(r["date"], r["source_currency"], r["target_currency"]) == fx_key(day, doc["currency"], base)), None)
        if rate is None:
            return f"документ {ids[0]}: проведённая сумма расходится с курсом политики"
        from decimal import Decimal
        correct = convert(amount, Decimal(rate), exp[doc["currency"]], exp[base])
        return (f"документ {ids[0]}: {money(amount, doc['currency'], exp)} × курс {rate} на {day} = "
                f"{money(correct, base, exp)}, а в книге сумма отличается на {money(found['claimed_amount_minor'], base, exp)}")
    if kind == "allocation_gap":
        pay = next(docs[i] for i in ids if "payment_id" in docs[i])
        inv = next(docs[i] for i in ids if "invoice_id" in docs[i] and "payment_id" not in docs[i])
        return (f"платёж {pay['doc_id']} на {money(pay['amount_minor'], pay['currency'], exp)} ссылается на счёт "
                f"{inv['doc_id']}, но {money(found['claimed_amount_minor'], base, exp)} осели на счёте «нераспределённые», "
                "а счёт остался открытым")
    return "по открытым записям выполняется определение ошибки"


def observation_text(action: dict, observation: dict) -> str:
    kind = observation.get("type")
    if kind == "records":
        more = "есть ещё страницы" if observation.get("next_cursor") else "последняя страница"
        return f"записей: {len(observation['records'])}, {more}"
    if kind == "final":
        return "эпизод завершён"
    if kind == "error":
        return f"ошибка {observation['error']['code']}"
    return kind or "?"


def pick_config() -> object:
    for config in CANDIDATES:
        if episodes_tool.unsupported(config) is None:
            return config
    return None


def run(seed: int, directory=None) -> list[str]:
    from auditgym import Env
    from auditgym.findings import parse_findings
    from auditgym.generate import generate
    from auditgym.matcher import match
    from auditgym.storage import load_episode
    from policies.runner import make_policy, run_episode

    config = None
    if directory:
        episode = load_episode(directory)
        open_env = lambda: Env.from_files(directory)  # noqa: E731
        start = {}
    else:
        config = pick_config()
        episode = generate(seed, config)
        open_env = Env
        start = {"seed": seed, "config": config}
    public, truth = episode["public"], episode["private"]["truth"]
    manifest, policy = public["manifest"], public["policy"]
    exp = policy["currency_exponents"]
    counts = manifest["counts"]
    lines = ["=== AuditGym: один эпизод аудита по шагам ===",
             "Задача агента: найти в книгах компании ошибки четырёх типов — дубль счёта, нераспределённый",
             "платёж, ошибку курса, документ чужого периода — и доказать каждую ссылками на записи. Ошибок может не быть.",
             f"Компания: период {policy['period_start'][:10]} — {policy['period_end'][:10]} (часовой пояс {policy['timezone']}), "
             f"базовая валюта {policy['base_currency']}.",
             f"Записей: проводок {counts['ledger']}, счетов {counts['invoices']}, платежей {counts['payments']}, "
             f"курсов {counts['fx']}. Лимит — 100 шагов; страница — до 50 записей.", "",
             "--- Ход скриптовой политики (solver_policy) ---"]

    with open_env() as env:
        result = run_episode(env, make_policy("solver_policy"), record=True, **start)
    for row in result["trajectory"]:
        action, observation = row["action"], row["observation"]
        name = ACTIONS.get(action.get("action"), action.get("action"))
        if action.get("action") == "list_records":
            name += " " + TABLES.get(action["table"], action["table"])
        elif action.get("action") == "submit":
            name += f" (выводов: {len(action['findings']['findings'])})"
        lines.append(f"шаг {row['step']:>2}: {name} → {observation_text(action, observation)}; "
                     f"осталось {observation.get('steps_remaining')}")

    findings = result["findings"] or []
    lines += ["", "--- Что заметила политика (по открытым записям) ---"]
    for number, found in enumerate(findings, 1):
        lines.append(f"{number}. {NAMES.get(found['type'], found['type'])}: {reason(found, public)}.")
    if not findings:
        lines.append("Ничего: книги выглядят чистыми.")

    lines += ["", "--- Отправленные выводы ---"]
    for number, found in enumerate(findings, 1):
        lines.append(f"{number}. {found['type']} | документы {', '.join(found['affected_ids'])} | "
                     f"{money(found['claimed_amount_minor'], found['currency'], exp)} | "
                     f"доказательств: {len(found['evidence_ids'])}")

    info = result["info"]
    normalized = parse_findings(result["payload"], manifest["episode_id"])
    pairs = dict(match(truth, normalized, public)["pairs"])
    issues = {issue["issue_id"]: issue for issue in truth["issues"]}
    lines += ["", "--- Оценка (считает оценщик в отдельном процессе) ---",
              f"Награда {result['reward']:.3f} = clip(2·F1 − 1 − 0.002·{info['inspection_steps']} шагов осмотра); "
              f"P={info['precision']:.2f}, R={info['recall']:.2f}, F1={info['f1']:.2f}."]
    for index in range(len(normalized)):
        verdict = (f"совпал с эталонной ошибкой {pairs[index]} ({issues[pairs[index]]['type']}) — TP"
                   if index in pairs else "не совпал ни с одной эталонной ошибкой — FP")
        lines.append(f"  вывод {index + 1}: {verdict}")
    missed = [issue for issue_id, issue in issues.items() if issue_id not in pairs.values()]
    lines.append("  пропущено эталонных ошибок: " + (", ".join(i["type"] for i in missed) if missed else "нет"))

    lines += ["", "--- Тот же эпизод: атака и молчание ---"]
    for name, title in (("flag_all", "«Пометить всё»"), ("silence", "Молчание")):
        with open_env() as env:
            other = run_episode(env, make_policy(name), **start)
        sent = len(other["findings"] or [])
        lines.append(f"{title}: выводов {sent}, награда {other['reward']:.3f} "
                     f"(P={other['info']['precision']:.2f}, R={other['info']['recall']:.2f}, F1={other['info']['f1']:.2f})")
    lines.append("Ложные обвинения и молчание проигрывают точному ответу с доказательствами.")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode", default=None, help="an episode directory instead of a generated one")
    args = parser.parse_args(argv)
    print("\n".join(run(args.seed, args.episode)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
