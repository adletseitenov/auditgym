# AuditGym

Воспроизводимая RL-среда финансового аудита (кейс Genesis, FinTech): генератор синтетических компаний с известными ошибками четырёх типов (`duplicate_invoice`, `allocation_gap`, `fx_error`, `period_cutoff`) и двумя сценариями их взаимодействия, контракт `reset`/`step` с пятью действиями и 100 шагами, общая награда из кейса.
Оценщик работает в отдельном процессе, эталон живёт только в его памяти. Только стандартная библиотека Python 3.11, без сети, платных API и GPU.

## Проверка одной командой

```
docker build -t auditgym . && docker run --rm --network none auditgym
python3 -m auditgym check          # то же без Docker: Python ≥ 3.11, зависимостей нет
python3 -m auditgym check --full   # полный режим: все тесты и шесть длинных инструментов
```

Запускает юнит-тесты, линтер детерминизма и таблицу `tools/score.py` (наше зеркало критериев кейса §10, это не оценка жюри); хвост вывода: `Внутренние проверки (не оценка жюри): пройдено N/M (…%)`, `total N s`, `CHECK: PASSED`. Быстрый режим занимает около 4–7 минут в зависимости от машины, полный (`--full`) около 10–14 минут; код возврата 0 только при `CHECK: PASSED`. Измерения по машинам, состав режимов и CI — [docs/USAGE.md](docs/USAGE.md), «Проверка: подробности».

## Что сдано (CASE §09)

| Материал | Где |
|---|---|
| Код, лицензия, зависимости | `auditgym/`, `LICENSE` (MIT), `pyproject.toml` (`dependencies = []`) |
| README, записка о дизайне | этот файл, [docs/DESIGN.md](docs/DESIGN.md) |
| Matcher, награда, схемы | `auditgym/matcher.py`, `reward.py`, `findings.py`, `judge.py`; `schemas/` |
| Эпизоды 10 train + 10 dev | `episodes/` (пересборка: `make episodes`) |
| Тесты | `tests/` (`make test`) |
| Базовые политики | `policies/` (`make report`) |
| Траектории | `trajectories.jsonl` (`make trajectories`) |
| Демо | `python3 -m tools.demo` (`make demo`) |
| Адаптер жюри | `judge_adapter.py` (`python judge_adapter.py --demo`), [docs/ADAPTER.md](docs/ADAPTER.md) |

Полная таблица с командами пересборки — [docs/SUBMISSION.md](docs/SUBMISSION.md).

## Критерии (CASE §10) → где проверяется

| Критерий | Где проверяется |
|---|---|
| Контракт и изоляция (25) | `tests/env/test_evaluator.py`, `test_audit_env.py`, `test_process.py`; `python3 -m tools.leakscan` |
| Данные и эталон (25) | `tests/gen/`, `tests/solver/`; `python3 -m tools.fuzz --seeds 20`, `tools.tricks` |
| Надёжность награды (25) | `tests/judge/`; `python3 -m tools.attack`, `tools.mutants` |
| Воспроизводимость и обобщение (15) | `tests/test_rng.py`; `python3 -m tools.metamorph`, `tools/lint_determinism.py` |
| Инженерия и объяснение (10) | `tools/checks_pack.py` (одна команда, версии, измеренный разбор в `docs/DESIGN.md`, ограничения) |

Таблица «требование кейса §07 → код → тест» — [docs/SUBMISSION.md](docs/SUBMISSION.md).

## Быстрый старт для политики

Политика — объект с `reset(observation)` и `act(observation) -> action`. Среда запускает процесс-оценщик сама.

```python
from auditgym import Env
from policies.runner import make_policy      # любая политика из policies/

with Env(mode="train") as env:
    obs, info = env.reset(7, {"errors": {"duplicate_invoice": 1}})   # seed, config
    policy = make_policy("solver_policy")
    policy.reset(obs)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        done = terminated or truncated
print(reward, info)   # 0.964 {'f1': 1.0, 'inspection_steps': 18, 'precision': 1.0, 'recall': 1.0, 'reward': 0.964}
```

Эпизод из каталога: `Env.from_files("episodes/dev/ep_01")`; тот же оценщик по JSON-строкам — `python3 -m auditgym serve --mode train`; обёртка Gymnasium — `AuditGym-v0`. Действия: `list_records`, `get_record`, `get_document`, `calculator`, `submit`; конфигурация, уровни `easy`…`max`, ошибки — [docs/USAGE.md](docs/USAGE.md), полный договор — [docs/CONTRACTS.md](docs/CONTRACTS.md).

## Адаптер жюри (judge_adapter.py)

Тонкий адаптер лежит в корне репозитория (в контейнере `/app/judge_adapter.py`). Рабочая директория — корень репозитория (`cd /app`), зависимостей нет; жюри делает `sys.path.insert(0, "/app"); import judge_adapter`. Демонстрация: `python judge_adapter.py --demo` (четыре базовые политики только через адаптер, закрытый режим, по эпизодным каталогам; около секунды). Единая команда проверки та же: `python -m auditgym check`.

- `generate_episode(seed, config, output_dir)`: обёртка над нашим генератором (`auditgym export`); пишет `public/` (manifest, policy, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv) и `private/` (truth, provenance, `pre_*`). Одинаковые аргументы дают побайтно один каталог.
- `make_env()`: объект с `reset(seed, config)` и `step(action)`. Оценщик остаётся отдельным процессом (`auditgym.client.Env`). `config` = `{"mode": "closed"|"train", "episode_dir": ..., "team_config": {...}}`: с `episode_dir` грузится именно этот эпизод (повторный `reset` начинает его заново), `train` без каталога генерирует по `seed` и `team_config`. В `closed` награда для вызывающего всегда `0.0`, `info = {}`; истинная награда у оценщика (журнал).
- Действия `{"name": ..., "args": {...}}`: `list_records(table, cursor, limit)`, `get_record(table, id)`, `get_document(doc_id)`, `calculator(expression)`, `submit({"findings": <findings.json>})`; таблицы `ledger`, `invoices`, `payments`, `fx`. Наблюдения: первое `{"task", "manifest", "policy", "remaining_steps": 100}`, дальше `{"result", "error", "remaining_steps"}`; `submit` отвечает `{"accepted": true}` или ошибкой `invalid_findings`.
- `score_episode(episode_dir, findings, inspection_steps)`: `{"tp", "fp", "fn", "precision", "recall", "f1", "return"}`; неверная структура — `return = -1`, остальные поля `null`, код ошибки пишется в журнал оценщика (`$AUDITGYM_SCORE_LOG`, по умолчанию `runs/score_errors.jsonl`).

Соответствие наших четырёх типов ошибок типам кейса:

| Наш тип | Тип кейса | Что именно размечено |
|---|---|---|
| `duplicate_invoice` | дубль счёта | два проведённых счёта с одной подписью (контрагент, валюта, сумма, позиции с периодом услуги) |
| `allocation_gap` | пропущенное или частичное распределение платежа | часть платежа по ссылке на счёт осела на `unallocated` вместо закрытия дебиторки; сумма — эта строка в базовой валюте |
| `fx_error` | неверная дата или курс конвертации | проводка по курсу не той даты (`wrong_date`) или по иному значению (`wrong_rate`); эталон считается по `policy` |
| `period_cutoff` | ошибка отнесения к отчётному периоду | документ, дата которого (в часовом поясе компании) вне периода, но проведён; граница `[start, end)` |

Подробности, таблица «проверка → наш тест», допущения и известные расхождения — [docs/ADAPTER.md](docs/ADAPTER.md).

## Результаты базовых политик

Генерированный набор из 200 эпизодов (150 с ошибками, 50 чистых), все политики идут через процесс среды в режиме `train`. Таблицу целиком пишет `python3 -m tools.report --out docs/RESULTS.md` (`make report`); строки ниже скопированы оттуда, и `python3 -m auditgym check` сверяет их с `docs/RESULTS.md`.

<!-- results:begin -->
| Политика | Эпизодов | P | R | F1 | Возврат: ошибки | Возврат: чистые | Возврат: все | Shaped (эксп.): ошибки | Шаги |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | 200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.0 |
| solver_policy | 200 | 1.000 | 1.000 | 1.000 | 0.960 | 0.959 | 0.960 | 0.960 | 21.0 |
| reconcile | 200 | 0.141 | 0.511 | 0.219 | -0.589 | -1.000 | -0.692 | -0.584 | 19.5 |
| random_explorer | 200 | 0.000 | 0.000 | 0.000 | -1.000 | -0.089 | -0.772 | -0.999 | 11.0 |
| silence | 200 | 0.000 | 0.000 | 0.000 | -1.000 | 1.000 | -0.500 | -1.000 | 1.0 |
| flag_all | 200 | 0.017 | 0.148 | 0.030 | -0.953 | -1.000 | -0.964 | -0.941 | 19.5 |
| flag_all_cutoff | 200 | 0.015 | 0.132 | 0.027 | -0.959 | -1.000 | -0.969 | -0.959 | 19.5 |
| flag_all_duplicates | 200 | 0.007 | 0.066 | 0.013 | -0.981 | -1.000 | -0.986 | -0.979 | 19.5 |
| heuristic | 200 | 0.234 | 0.951 | 0.372 | -0.295 | -1.000 | -0.471 | -0.288 | 21.0 |
<!-- results:end -->

Возврат — `clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1)`. `heuristic` — демонстрационная политика, не базовая; пояснения — [docs/USAGE.md](docs/USAGE.md).

## Закрытая проверка

- Для закрытого прогона задайте секрет оценщика, лучше файлом `AUDITGYM_SECRET_FILE`: без него эпизод по малому seed восстанавливается перебором (`python3 -m tools.seed_recovery`).
- Один секрет, seed и конфигурация дают побайтно один эпизод; секрет не попадает в наблюдения, журналы и траектории.
- Граница — оценщик под отдельным пользователем или в отдельном контейнере; `Env()` под тем же пользователем границей не является.
- `Env(mode="eval", server_command=[...])` подключается к такому оценщику; `Env` секрет из окружения не наследует и в `eval` без явного секрета отказывает (`bad_secret_file`).
- Журнал оценщика подписан (HMAC-цепочка): `python3 -m auditgym verify-journal PATH --secret-file SECRET`. Строки `eval`, записанные без секрета оценщика (`"secret": "none"`) или без подписи, дают код 1: такой закрытый прогон не засчитывается.
- Эталон для oracle под секретом: `python3 -m auditgym export --secret-file ...`.

Рецепты (`sudo`, Docker), коды возврата `verify-journal`, экспорт — [docs/CLOSED-RUN.md](docs/CLOSED-RUN.md).

## Ограничения

Коротко; полный список — [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

- **Синтетические книги:** данные генерируются, только продажи и дебиторка; оцениваются четыре типа ошибок и два сценария.
- **Seed можно восстановить перебором**, если агент может импортировать пакет и seed мал; мера — секрет оценщика и длинные seed.
- **Политика в том же процессе, что и запуск, видит seed:** в закрытом прогоне политику запускают отдельным процессом.
- **`inspection_steps` не включает шаг `submit`:** если жюри считает и `submit`, награда oracle 0.998 вместо 1.0.
- **На книгах с ошибками молчание получает −1**, любая случайная угадка выше; это свойство формулы кейса.
- **Результаты `flag_all`:** точность 0.017 и возврат −0.953 на книгах с ошибками (генерированный набор, таблица выше); это измерение, а не доказательство, что награду нельзя обмануть.
- **Награда не доказывает**, что агент понимает учёт или обобщается на реальные книги; обучение модели на среде не измерялось.
- Диапазоны кейса: число ошибок (не больше 12 вместе с I1/I2) проверяется принудительно, размеры книги соблюдаются конфигурациями по умолчанию; проверено на macOS и Linux, Windows нет.

## Документация

| Файл | Назначение |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | проверка подробно, конфигурация, уровни, ошибки, JSON-протокол, Gymnasium, экспериментальная награда |
| [docs/ADAPTER.md](docs/ADAPTER.md) | адаптер жюри: интерфейс, отображение действий и ответов, проверки C01–C20, пары, расхождения |
| [docs/CLOSED-RUN.md](docs/CLOSED-RUN.md) | закрытая проверка: секрет, граница, рецепты, подписанный журнал, экспорт |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | ограничения полностью |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | что сдано и требования кейса → код → тест, полные таблицы |
| [docs/DESIGN.md](docs/DESIGN.md) | записка о дизайне, формулы, измеренный разбор сбоев |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | договор между частями: действия, наблюдения, форматы |
| [docs/RESULTS.md](docs/RESULTS.md) | полные результаты политик и уровней |
| [docs/DECISIONS.md](docs/DECISIONS.md), [docs/AUDIT.md](docs/AUDIT.md), [docs/JUDGE-NOTES.md](docs/JUDGE-NOTES.md) | решения, находки аудита, заметки для жюри |
| [docs/CASE.md](docs/CASE.md), [docs/SPEC.md](docs/SPEC.md), [docs/QUESTIONS.md](docs/QUESTIONS.md) | кейс, исходный план, вопросы организаторам |

## Лицензия и инструменты

Лицензия MIT (`LICENSE`). Среда и инструменты — Python 3.11 и только его стандартная библиотека; для сборки образа нужен Docker. Код, тесты и документы написаны командой вместе с агентами Claude Code (параллельные сессии по зонам генератора, среды, оценщика и упаковки, а также независимые агенты-аудиторы); решения по контракту и ревью принимали люди.
