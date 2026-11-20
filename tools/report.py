"""Measure the baseline policies and write docs/RESULTS.md. Every number in that file comes from here.

Run: python -m tools.report --out docs/RESULTS.md [--n 200]

Sets: episodes/train and episodes/dev (built by tools.episodes; rebuilt into a temporary
directory when missing) and a larger generated set over the supported config pools. Policies run
through a real `Env` process in train mode. Truth is used only here, on the evaluator side: to
build the oracle and to split recall by error type.
"""
from __future__ import annotations

import argparse
import pathlib
import platform
import statistics
import sys
import tempfile
import time

from tools import episodes as episodes_tool

ROOT = pathlib.Path(__file__).resolve().parent.parent
POLICIES = ("oracle", "solver_policy", "reconcile", "random_explorer", "silence",
            "flag_all", "flag_all_cutoff", "flag_all_duplicates", "heuristic")
TYPED = ("reconcile", "solver_policy")
ERROR_TYPES = ("duplicate_invoice", "allocation_gap", "fx_error", "period_cutoff")
LARGE_SEED_BASE = 50_000
CLEAN_EVERY = 4  # every 4th generated episode is a clean company


def _row(result: dict, episode: dict) -> dict:
    from auditgym.findings import StructureError, parse_findings
    from auditgym.matcher import match

    truth = episode["private"]["truth"]
    issues = {issue["issue_id"]: issue["type"] for issue in truth.get("issues", [])}
    types = {kind: [0, 0] for kind in ERROR_TYPES}
    for kind in issues.values():
        types.setdefault(kind, [0, 0])[1] += 1
    if result["payload"] is not None:
        try:
            findings = parse_findings(result["payload"], episode["public"]["manifest"]["episode_id"])
            for _, issue_id in match(truth, findings, episode["public"])["pairs"]:
                types[issues[issue_id]][0] += 1
        except StructureError:
            pass
    info = result["info"]
    return {"reward": result["reward"], "precision": info.get("precision", 0.0), "recall": info.get("recall", 0.0),
            "f1": info.get("f1", 0.0), "shaped": info.get("shaped_reward"), "steps": result["steps"], "seconds": result["seconds"],
            "truncated": result["truncated"], "clean": not issues, "types": types}


def _policy(name: str, episode: dict):
    from policies.runner import make_policy

    return make_policy(name, truth=episode["private"]["truth"] if name == "oracle" else None)


def generated_cases(n: int, quiet: bool = True) -> list[tuple[int, object]]:
    errors = episodes_tool.supported_pool(episodes_tool.ERROR_POOL, quiet=quiet)
    clean = episodes_tool.supported_pool(episodes_tool.CLEAN_POOL, quiet=quiet)
    cases = []
    for k in range(n):
        if k % CLEAN_EVERY == CLEAN_EVERY - 1:
            cases.append((LARGE_SEED_BASE + k, clean[k % len(clean)]))
        else:
            cases.append((LARGE_SEED_BASE + k, errors[k % len(errors)]))
    return cases


def run_generated(names, cases, generation: dict | None = None) -> dict[str, list[dict]]:
    """Every policy on every (seed, config), one Env process per policy (reset isolates)."""
    from auditgym import Env
    from auditgym.generate import generate
    from policies.runner import run_episode

    episodes = []
    for seed, config in cases:
        started = time.perf_counter()
        episodes.append(generate(seed, config))
        if generation is not None:
            generation.setdefault("seconds", []).append(time.perf_counter() - started)
            generation.setdefault("attempts", []).append(episodes[-1]["private"]["provenance"].get("attempt", 0))
    rows: dict[str, list[dict]] = {}
    for name in names:
        with Env(mode="train", shaped_reward=True) as env:
            for (seed, config), episode in zip(cases, episodes):
                result = run_episode(env, _policy(name, episode), seed, config)
                rows.setdefault(name, []).append(_row(result, episode))
    return rows


def run_directories(names, directories) -> dict[str, list[dict]]:
    from auditgym import Env
    from auditgym.storage import load_episode
    from policies.runner import run_episode

    rows: dict[str, list[dict]] = {}
    for directory in directories:
        episode = load_episode(directory)
        for name in names:
            env = Env.from_files(directory, mode="train", shaped_reward=True)
            try:
                rows.setdefault(name, []).append(_row(run_episode(env, _policy(name, episode)), episode))
            finally:
                env.close()
    return rows


def summary(rows: list[dict]) -> dict:
    """P, R, F1 over books with errors (on clean books P and R are 0 by definition); returns
    split into books with errors, clean books and all."""
    errors = [row for row in rows if not row["clean"]]
    clean = [row for row in rows if row["clean"]]

    def mean(items: list[dict], key: str) -> float | None:
        return statistics.fmean(row[key] for row in items) if items else None

    return {"n": len(rows), "precision": mean(errors, "precision"), "recall": mean(errors, "recall"),
            "f1": mean(errors, "f1"), "reward_errors": mean(errors, "reward"), "reward_clean": mean(clean, "reward"),
            "reward": mean(rows, "reward"), "shaped_errors": mean(errors, "shaped"), "steps": mean(rows, "steps"), "ms": 1000 * mean(rows, "seconds"),
            "truncated": sum(row["truncated"] for row in rows)}


def type_recall(rows: list[dict]) -> dict[str, tuple[int, int]]:
    out = {}
    for kind in ERROR_TYPES:
        found = sum(row["types"].get(kind, [0, 0])[0] for row in rows)
        total = sum(row["types"].get(kind, [0, 0])[1] for row in rows)
        out[kind] = (found, total)
    return out


# "Flagging everything gives a bad result" (CASE 05). On books with errors silence sits at the
# floor -1, so a flag-all that hits one issue is always a little above it; "below silence" is
# required on clean books and over the whole set, and a bad absolute result on books with errors:
# on average no better than an answer with F1 = 0.1 (2 * 0.1 - 1 = -0.8). Measured: -0.93 to -0.98.
BAD = -0.8


def ordering_problems(rows: dict[str, list[dict]], bad: float = BAD) -> list[str]:
    """The expected order on books with errors (flag-all at most `bad` there), plus flag-all below
    silence over the whole set when it has clean books; empty list when it holds."""
    errors = {name: [row for row in policy_rows if not row["clean"]] for name, policy_rows in rows.items()}
    mean = {name: statistics.fmean(row["reward"] for row in policy_rows)
            for name, policy_rows in errors.items() if policy_rows}
    problems = []

    def need(condition: bool, text: str) -> None:
        if not condition:
            problems.append(text + " (" + ", ".join(f"{k}={v:.3f}" for k, v in sorted(mean.items())) + ")")

    if {"oracle", "solver_policy"} <= set(mean):
        need(mean["oracle"] >= mean["solver_policy"], "oracle < solver_policy")
    if {"solver_policy", "reconcile"} <= set(mean):
        need(mean["solver_policy"] > mean["reconcile"], "solver_policy <= reconcile")
    for name in [n for n in mean if n.startswith("flag_all")]:
        if "reconcile" in mean:
            need(mean["reconcile"] > mean[name], f"reconcile <= {name}")
        need(mean[name] <= bad, f"{name} > {bad} on books with errors")
    overall = {name: statistics.fmean(row["reward"] for row in policy_rows) for name, policy_rows in rows.items()}
    if "silence" in overall and any(row["clean"] for row in rows["silence"]):
        for name in [n for n in overall if n.startswith("flag_all")]:
            if overall[name] >= overall["silence"]:
                problems.append(f"{name} >= silence over the whole set")
    return problems


def _eval_mode_agrees(directory) -> bool:
    """Eval mode: the policy sees 0.0 and {}, the evaluator log holds the train-mode reward."""
    from auditgym import Env
    from policies.runner import read_results, run_episode, make_policy

    with tempfile.TemporaryDirectory() as scratch:
        path = pathlib.Path(scratch) / "results.jsonl"
        env = Env.from_files(directory, mode="eval", results_path=path)
        try:
            seen = run_episode(env, make_policy("solver_policy"))
        finally:
            env.close()
        logged = read_results(path)
    env = Env.from_files(directory, mode="train")
    try:
        train = run_episode(env, make_policy("solver_policy"))
    finally:
        env.close()
    return seen["reward"] == 0.0 and seen["info"] == {} and len(logged) == 1 and logged[0]["reward"] == train["reward"]


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _table(rows: dict[str, list[dict]]) -> list[str]:
    """Machine-independent metrics only: README copies these rows and `check` compares them byte for
    byte, so wall-clock time (different on every machine) lives in `_timing_table`."""
    lines = ["| Политика | Эпизодов | P | R | F1 | Возврат: ошибки | Возврат: чистые | Возврат: все | Shaped (эксп.): ошибки | Шаги |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in POLICIES:
        if name not in rows:
            continue
        s = summary(rows[name])
        lines.append(f"| {name} | {s['n']} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} | "
                     f"{_fmt(s['reward_errors'])} | {_fmt(s['reward_clean'])} | {_fmt(s['reward'])} | "
                     f"{_fmt(s['shaped_errors'])} | {_fmt(s['steps'], 1)} |")
    return lines


def _timing_table(rows: dict[str, list[dict]]) -> list[str]:
    """Mean wall-clock time per episode; depends on the machine, so it is not copied to the README."""
    lines = ["Время на эпизод (зависит от машины; в README не копируется):", "",
             "| Политика | Время, мс |", "|---|---:|"]
    for name in POLICIES:
        if name in rows:
            lines.append(f"| {name} | {_fmt(summary(rows[name])['ms'], 0)} |")
    return lines


def _split_counts(rows: dict[str, list[dict]]) -> tuple[int, int]:
    any_rows = next(iter(rows.values()))
    clean = sum(row["clean"] for row in any_rows)
    return len(any_rows) - clean, clean


def build_report(n: int, episodes_dir=None) -> str:
    from auditgym import SCHEMA_VERSION, __version__
    from auditgym.config import CONFIG_VERSION

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        base = pathlib.Path(episodes_dir) if episodes_dir else ROOT / "episodes"
        if not all((base / split).is_dir() for split in episodes_tool.SPLITS):
            base = pathlib.Path(scratch)
            episodes_tool.build(base, quiet=True)
        sets = {}
        for split in episodes_tool.SPLITS:
            directories = sorted(p for p in (base / split).iterdir() if p.is_dir())
            sets[split] = run_directories(POLICIES, directories)
        eval_ok = _eval_mode_agrees(sorted((base / "train").iterdir())[0])
    generation: dict = {}
    cases = generated_cases(n)
    sets["generated"] = run_generated(POLICIES, cases, generation)
    skipped_errors = len(episodes_tool.ERROR_POOL) - len(episodes_tool.supported_pool(episodes_tool.ERROR_POOL, True))
    skipped_clean = len(episodes_tool.CLEAN_POOL) - len(episodes_tool.supported_pool(episodes_tool.CLEAN_POOL, True))
    plan = episodes_tool.plan(quiet=True)
    fallbacks = sum(not entry["preferred"] for entry in plan)

    titles = {"train": "train (episodes/train)", "dev": "dev (episodes/dev)",
              "generated": f"генерированный набор ({n} эпизодов)"}
    out = ["# Результаты базовых политик", "",
           "Файл целиком пишет `python -m tools.report` (`make report`); руками числа не правим.",
           f"Генератор {__version__}, схема {SCHEMA_VERSION}, конфигурация {CONFIG_VERSION}, "
           f"Python {platform.python_version()}. Все политики идут через процесс среды в режиме train.", "",
           "## Наборы", ""]
    for name, rows in sets.items():
        with_errors, clean = _split_counts(rows)
        out.append(f"- {titles[name]}: {with_errors} с ошибками, {clean} чистых.")
    out += [f"- В плане train/dev {fallbacks} из {len(plan)} слотов взяли запасной вариант: "
            "генератор пока не умеет предпочтительный (`configs/episodes.json`).",
            f"- Из пулов генерированного набора пропущено как неподдерживаемые: {skipped_errors} конфигураций "
            f"с ошибками и {skipped_clean} чистых.", "",
            "Политики: `oracle` знает эталон (только оценщик); `solver_policy` читает все таблицы и запускает "
            "решатель; `reconcile` — простая сверка; `random_explorer` — случайные действия с seed 0; "
            "`silence` — пустой ответ; `flag_all*` — атака «пометить всё» в трёх вариантах; "
            "`heuristic` — демонстрационная эвристика без учёта отвлекающих случаев (не базовая политика и не часть "
            "баллов проверки): одинаковые суммы контрагента как дубли, любой неразнесённый остаток как разрыв, "
            "документ с датой вне периода и проводкой внутри как отсечка, расхождение с курсом напечатанной даты как "
            "курсовая ошибка; частичные оплаты, позиции, пояс компании и переплаты она не различает.", "",
            "Колонка «Shaped (эксп.): ошибки» — среднее экспериментальной награды с частичным баллом на книгах с "
            "ошибками (`auditgym/shaped.py`, формула в `docs/DESIGN.md`). **Она экспериментальная и не входит в "
            "награду кейса:** официальный возврат считается только по формуле CASE §05, колонки «Возврат» её не "
            "используют. В режиме eval её нет нигде.", ""]
    for name, rows in sets.items():
        out += [f"## {titles[name][0].upper() + titles[name][1:]}", ""] + _table(rows) + [""] + _timing_table(rows) + [""]
        truncated = {policy: summary(policy_rows)["truncated"] for policy, policy_rows in rows.items()}
        if any(truncated.values()):
            out += ["Обрезано по лимиту шагов: " + ", ".join(f"{k} {v}" for k, v in truncated.items() if v), ""]

    out += ["## Полнота по типам ошибок", "", "Найдено / всего по всем трём наборам.", "",
            "| Политика | " + " | ".join(ERROR_TYPES) + " |", "|---|" + "---:|" * len(ERROR_TYPES)]
    for name in TYPED:
        merged = [row for rows in sets.values() for row in rows[name]]
        cells = []
        for kind, (found, total) in type_recall(merged).items():
            cells.append(f"{found}/{total} ({_fmt(found / total, 2)})" if total else "нет в наборах")
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    from tools import levels as levels_tool

    level_stats = levels_tool.level_stats()
    out += ["", "## Уровни сложности", "",
            "Конфигурации — `configs/levels.json`, загрузка `auditgym.levels.load(имя)`; "
            f"{levels_tool.LEVEL_SEEDS} seed на уровень, политики идут через процесс среды в режиме train "
            "(`python -m tools.levels levels`). Страницы — число вызовов list_records для полного чтения всех "
            f"четырёх таблиц (среднее и диапазон); бюджет — {levels_tool.MAX_STEPS} шагов.", ""]
    out += levels_tool.levels_markdown(stats=level_stats) + [""]
    out += [f"Обрезано по лимиту шагов на всех уровнях: {sum(r['truncated'] for d in level_stats.values() for rows in d['policies'].values() for r in rows)}. "
            f"Потолок полного чтения по лимитам конфигурации — {levels_tool.MAX_PAGES} страниц, то есть не больше "
            f"{levels_tool.MAX_PAGES + 1} шагов с submit: ни один допустимый конфиг не доводит решатель до "
            "обрезки на 100 шагах. Давление бюджета есть только через штраф 0.002 за шаг (см. docs/DESIGN.md).", ""]
    out += ["Сколько страниц книги нужно решателю до окончательного ответа (проводки идут в порядке row_id, "
            "строки одного документа разбросаны по всем страницам; `python -m tools.levels prefix`):", "",
            levels_tool.prefix_text(), ""]

    seconds, attempts = generation["seconds"], generation["attempts"]
    retried = sum(a > 0 for a in attempts)
    out += ["", "## Генератор", "",
            f"- Эпизодов с повторной попыткой (attempt > 0): {retried} из {len(attempts)} "
            f"({_fmt(100 * retried / len(attempts), 1)} %), всего повторов {sum(attempts)}.",
            f"- Время генерации: среднее {_fmt(1000 * statistics.fmean(seconds), 1)} мс, "
            f"максимум {_fmt(1000 * max(seconds), 1)} мс.", "",
            "## Проверки порядка", ""]
    for name, rows in sets.items():
        problems = ordering_problems(rows)
        verdict = "выполняется" if not problems else "НАРУШЕН: " + "; ".join(problems)
        out.append(f"- {titles[name]}: oracle ≥ solver_policy > reconcile > flag_all*, "
                   f"flag_all* ≤ {BAD} на книгах с ошибками и < silence на всём наборе — {verdict}.")
    gaps = []
    for name, rows in sets.items():
        silence = summary(rows["silence"])["reward_errors"]
        best = max(summary(rows[n])["reward_errors"] for n in rows if n.startswith("flag_all"))
        gaps.append(f"{name} {best - silence:+.3f}")
    out += ["- Лучший вариант flag_all минус silence на книгах с ошибками: " + ", ".join(gaps)
            + ". Молчание там на нижней границе −1, поэтому любое случайное попадание атаки чуть выше него; "
            "на чистых книгах атака получает −1, молчание +1."]
    out += [f"- Режим eval: политика видит награду 0.0 и пустой info, журнал оценщика равен награде train — "
            f"{'да' if eval_ok else 'НЕТ'}.", "",
            f"Время прогона отчёта: {_fmt(time.perf_counter() - started, 1)} с.", ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="docs/RESULTS.md")
    parser.add_argument("--n", type=int, default=200, help="size of the generated set")
    parser.add_argument("--episodes", default=None, help="directory with train/ and dev/ (default: episodes)")
    args = parser.parse_args(argv)
    text = build_report(args.n, args.episodes)
    pathlib.Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
