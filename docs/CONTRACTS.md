# Контракты между зонами

Версия 1, 19.11.2026. Это общий договор четырёх зон: `gen`, `env`, `judge`, `qa`. Каждый пишет свою зону против этого файла и может не ждать остальных. **Менять только вместе:** правка здесь = сообщение в чат команды + повышение `SCHEMA_VERSION`.

Основание — `docs/CASE.md` (текст кейса) и `docs/SPEC.md` (ТЗ версии 2). Если этот файл расходится с кейсом, прав кейс.

## 1. Эпизод в памяти

Везде один и тот же JSON-совместимый словарь:

```python
episode = {
    "public": {
        "manifest": {...},      # §2.1
        "policy": {...},        # §2.2
        "ledger": [row, ...],   # §2.3
        "invoices": [inv, ...], # §2.4
        "payments": [pay, ...], # §2.5
        "fx": [rate, ...],      # §2.6
    },
    "private": {
        "truth": {...},         # §3.1
        "provenance": {...},    # §3.2
        "pre_ledger": [row, ...],  # книга до вставки ошибок, с финальными ID
    },
}
```

На диске: `<dir>/public/{manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv}` и `<dir>/private/{truth.json, provenance.json, pre_ledger.csv}`. Открытые и закрытые файлы всегда в разных каталогах. В CSV отсутствующая ссылка — пустое поле; через инструменты среды она отдаётся как JSON `null`.

**Правила значений:** деньги — `int` в минимальных единицах; курсы — десятичные строки, в коде `Decimal`; время — ISO-8601 со смещением (`2026-10-03T14:05:00+05:00`); даты — `YYYY-MM-DD`. `float` запрещён. Все списки записей отсортированы в порядке, который назначила поверхность (`surface`), и этот порядок — часть эпизода.

**ID непрозрачны:** строки из `Rng.token`, без префиксов по типу и без порядка. Тип записи определяется таблицей, а не видом ID. Пространства ID: `row_id` (книга), `doc_id` (счета и платежи, уникален в обеих таблицах), `invoice_id`, `payment_id`, `entity_id`; ID курса — составной ключ `"YYYY-MM-DD|SRC|TGT"`.

## 2. Открытые файлы

### 2.1 manifest.json
```json
{"schema_version": "1", "episode_id": "ep_7f3k2m9q", "base_currency": "KZT",
 "period_start": "2026-10-01T00:00:00+05:00", "period_end": "2026-11-01T00:00:00+05:00",
 "timezone": "+05:00",
 "counts": {"ledger": 612, "invoices": 48, "payments": 41, "fx": 62},
 "rounding": "half_even", "config_version": "1"}
```
`episode_id` — хэш, seed в нём нет. `counts` — фактические после вставки ошибок. Период — полуинтервал `[period_start, period_end)`.

### 2.2 policy.json
```json
{"ledger_currency": "base", "base_currency": "KZT", "timezone": "+05:00",
 "period_start": "...", "period_end": "...",
 "fx_rate_date": "document_date_in_company_timezone",
 "fx_formula": "round(amount_minor * 10^(exp_base - exp_src) * rate)",
 "rounding": "half_even",
 "currency_exponents": {"KZT": 2, "USD": 2, "EUR": 2},
 "accounts": {"cash": "cash", "receivable": "receivable", "revenue": "revenue",
              "unallocated": "unallocated", "fx_difference": "fx_difference", "expense": "expense"},
 "document_date": {"invoice": "issued_at", "payment": "timestamp"},
 "refund": "payment with negative amount_minor and one invoice ref",
 "amount_tolerance_minor": {"duplicate_invoice": 0, "allocation_gap": 0, "fx_error": 1, "period_cutoff": 0},
 "evidence_superset_allowed": true}
```

### 2.3 ledger.csv
Колонки строго в этом порядке: `row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id`.
У строки ровно одна из сторон больше нуля. `currency` всегда равна базовой. `account` — одно из значений `policy.accounts`. Группа проводок: по `payment_id`, если он не пуст; иначе по `invoice_id`; у фоновых (обе ссылки пусты) — пара с одинаковыми `timestamp` и `entity_id`.

### 2.4 invoices.jsonl
```json
{"doc_id": "d_…", "invoice_id": "i_…", "entity_id": "e_…",
 "issued_at": "2026-10-03T14:05:00+05:00", "due_at": "2026-11-02T14:05:00+05:00",
 "currency": "USD", "total_minor": 125000,
 "line_items": [{"description": "Subscription", "quantity": 1, "unit_price_minor": 125000,
                 "amount_minor": 125000, "service_start": "2026-10-01", "service_end": "2026-10-31"}]}
```
(префиксы `d_`, `i_` здесь только для читаемости примера; в данных их нет). `total_minor` равен сумме `amount_minor` позиций.

### 2.5 payments.jsonl
```json
{"doc_id": "…", "payment_id": "…", "timestamp": "2026-10-20T09:30:00+05:00",
 "currency": "USD", "amount_minor": 125000, "invoice_refs": ["…invoice_id…"]}
```
`invoice_refs` — список `invoice_id` или `null` (аванс). Отрицательная сумма — возврат. Платёж с несколькими ссылками гасит все свои счета полностью; частичный платёж ссылается ровно на один счёт. Валюта платежа равна валюте счёта.

### 2.6 fx.csv
Колонки: `date,source_currency,target_currency,rate`. Только направление «иностранная → базовая», на каждую дату периода с запасом в 5 дней по краям, без пропусков.

## 3. Закрытые файлы

### 3.1 truth.json
```json
{"schema_version": "1", "episode_id": "ep_7f3k2m9q",
 "issues": [{"issue_id": "t1", "type": "duplicate_invoice",
             "affected_ids": ["docA", "docB"], "expected_amount_minor": 125000, "currency": "USD",
             "evidence_sets": [["docA", "docB"]]}]}
```
Типы: `duplicate_invoice`, `allocation_gap`, `fx_error`, `period_cutoff`. `affected_ids` — множество `doc_id` (хранится отсортированным, сравнивается как множество). Каждый набор доказательств — не больше 10 ID, все существуют в открытых записях.

| Тип | `affected_ids` | Сумма и валюта | `evidence_sets` |
|---|---|---|---|
| `duplicate_invoice` | оба счёта | `total_minor`, валюта счёта | `[[docA, docB]]` |
| `allocation_gap` | платёж и счёт | `credit_minor` строки на `unallocated`, базовая валюта | `[[doc_платежа, doc_счёта, row_нераспределённых]]` |
| `fx_error` | документ | модуль разницы, базовая валюта | `[[doc, row_пересчёта]]` для каждой строки пересчёта |
| `period_cutoff` | документ | сумма документа, валюта документа | `[[doc, row_Дт], [doc, row_Кт]]` |

Точные предикаты — в `docs/SPEC.md`, раздел «Определения ошибок».

### 3.2 provenance.json
```json
{"seed": 91827, "config": {...}, "generator_version": "0.1.0", "attempt": 0,
 "steps": [{"operator": "duplicate_invoice", "params": {...}, "issue_id": "t1",
            "before": [...records...], "after": [...records...]}],
 "id_map": {"<внутренний ID>": "<открытый ID>"}}
```
Записи `before` и `after` — в открытых (финальных) ID.

## 4. Конфигурация

`reset(seed, config)`; `config` может быть `{}` или `null` — берутся значения по умолчанию.

```json
{"config_version": "1",
 "size": {"invoices": [40, 100], "background_pairs": [150, 300]},
 "currencies": ["USD"], "base_currency": "KZT", "timezone": "+05:00",
 "period": {"start": "2026-10-01", "months": 1},
 "errors": {"duplicate_invoice": 1, "allocation_gap": 1, "fx_error": 1, "period_cutoff": 1},
 "interaction": "none",
 "distractors": true,
 "surface_seed": null}
```
- `errors` — число ошибок каждого типа (0 и больше). `{}` или все нули — чистая компания.
- `interaction`: `"none"`, `"I1"` (частичный платёж в иностранной валюте у границы периода: `allocation_gap` + `fx_error`), `"I2"` (копия счёта, датированная вне периода: `duplicate_invoice` + `period_cutoff`). Сценарий добавляет свои две записи эталона сверх `errors`.
- `surface_seed: null` — берётся из `seed`. Другое значение даёт тот же эпизод с другими ID и порядком.
- Проверки: всего ошибок ≤ 30; ошибок ≤ документов / 8; проводок ≤ 2 000; страниц на полное чтение ≤ 60; валюты из списка `KZT, USD, EUR, RUB, CNY`; неизвестный ключ — ошибка. Нарушение → `ConfigError` с коротким кодом, эпизод не создаётся.

## 5. Протокол среды

### 5.1 Вызовы
```python
env = Env(mode="train")                      # клиент; сам запускает процесс-оценщик
observation, info = env.reset(seed, config)
observation, reward, terminated, truncated, info = env.step(action)
env = Env.from_files("<dir>")                # эпизод из каталога, reset() без seed
```
Процесс-оценщик: `python -m auditgym serve [--mode train|eval] [--results PATH]`, по одной JSON-строке на запрос и ответ:
```json
→ {"op": "reset", "seed": 7, "config": {}}
← {"observation": {...}, "info": {}}
→ {"op": "step", "action": {...}}
← {"observation": {...}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}
```
В stdout идёт только протокол; журналы — в stderr или в файл `--results`.

### 5.2 Действия
```json
{"action": "list_records", "table": "ledger", "cursor": null, "limit": 50}
{"action": "get_record", "table": "invoices", "id": "…"}
{"action": "get_document", "doc_id": "…"}
{"action": "calculator", "expression": "125000 * 465.37"}
{"action": "submit", "findings": {"episode_id": "…", "findings": [...]}}
```
Таблицы: `ledger`, `invoices`, `payments`, `fx`. Парсер также принимает имя действия в ключах `tool`, `name`, `type` и аргументы во вложенном `args`, `arguments` или `params`. `limit` больше 50 обрезается до 50, по умолчанию 50. `get_record` для `fx` принимает составной ключ.

### 5.3 Наблюдения
Всегда словарь с `type` и `steps_remaining`:
```json
{"type": "reset", "task": {...}, "manifest": {...}, "policy": {...}, "steps_remaining": 100}
{"type": "records", "table": "ledger", "records": [...], "next_cursor": "…" , "steps_remaining": 99}
{"type": "record", "table": "invoices", "record": {...}, "steps_remaining": 98}
{"type": "document", "table": "payments", "record": {...}, "steps_remaining": 97}
{"type": "calculation", "result": "58171250.00", "steps_remaining": 96}
{"type": "error", "error": {"code": "unknown_id", "message": "Unknown id."}, "steps_remaining": 95}
{"type": "final", "steps_remaining": 94}
```
`next_cursor` равен `null` на последней странице. `task` содержит: текст задачи, перечень типов ошибок, схему findings, лимиты (30, 10, 1 000, 100 шагов), допуски. Больше в наблюдениях нет ничего: ни времени, ни seed, ни конфигурации, ни счётчиков сверх manifest.

Коды ошибок (тексты фиксированные): `unknown_action`, `unknown_table`, `unknown_id`, `bad_cursor`, `bad_arguments`, `bad_expression`, `episode_over`, `not_reset`. Любая ошибка, кроме двух последних, расходует шаг.

### 5.4 Завершение и награда
- До терминального шага `reward = 0.0`, `info = {}`.
- `submit` → `terminated = true`. Сотый шаг без `submit` или лимит времени → `truncated = true`, ответ считается пустым, `inspection_steps = 100`.
- `train`: терминальный `reward` — формула кейса; `info = {"reward", "precision", "recall", "f1", "inspection_steps"}`. Никаких вердиктов по отдельным выводам.
- `eval`: политике возвращается `reward = 0.0`, `info = {}`; настоящая награда и разбор пишутся в файл `--results` (одна JSON-строка на эпизод).
- Неверная структура выводов → `terminated = true`, награда `-1.0`.
- `step` после конца → ошибка `episode_over`, `reward 0.0`, `terminated = true`, состояние не меняется. `step` до `reset` → `not_reset`.

## 6. Выводы агента

```json
{"episode_id": "ep_7f3k2m9q",
 "findings": [{"type": "duplicate_invoice", "affected_ids": ["docA", "docB"],
               "claimed_amount_minor": 125000, "currency": "USD",
               "evidence_ids": ["docA", "docB"], "explanation": "Same customer, items and total."}]}
```
**Неверная структура (награда −1):** не словарь; нет обязательного поля; `findings` не список; больше 30 выводов; больше 10 `evidence_ids`; `explanation` длиннее 1 000 символов; неизвестный `type`; чужой `episode_id`; `claimed_amount_minor` не `int` (в том числе `true`, `100.0`, `"100"`) или меньше нуля; ID не строка; всё сообщение больше 64 КБ. Лишние ключи игнорируются; повторы внутри `affected_ids` и `evidence_ids` схлопываются.

**FP, а не −1:** неизвестный ID, не та валюта, не та сумма, неполные доказательства, повтор уже засчитанного вывода.

## 7. Интерфейсы модулей

| Зона | Модуль | Функция | Что делает |
|---|---|---|---|
| gen | `auditgym/generate.py` | `generate(seed: int, config: dict \| None) -> dict` | возвращает эпизод §1 для любого seed и валидной конфигурации |
| gen | `auditgym/config.py` | `normalize_config(config) -> dict`, `ConfigError` | значения по умолчанию и проверки §4 |
| gen | `auditgym/storage.py` | `write_episode(episode, dir)`, `load_episode(dir) -> dict` | диск §1; `load(write(e)) == e` |
| judge | `auditgym/findings.py` | `parse_findings(payload, episode_id) -> list[dict]`, `StructureError` | проверка структуры и нормализация §6 |
| judge | `auditgym/matcher.py` | `match(truth: dict, findings: list, public: dict) -> dict` | возвращает `{"tp", "fp", "fn", "pairs": [[номер_вывода, issue_id]]}` |
| judge | `auditgym/reward.py` | `metrics(tp, fp, fn) -> dict`, `terminal_reward(f1, inspection_steps) -> float` | формулы кейса §05 |
| judge | `auditgym/judge.py` | `judge(episode: dict, payload, inspection_steps: int) -> dict` | всё вместе: `{"reward", "precision", "recall", "f1", "tp", "fp", "fn", "structure_error"}` |
| env | `auditgym/evaluator.py` | `Evaluator(mode, results_path, clock)` с `reset(seed, config)`, `load(dir)`, `step(action)` | состояние эпизода на стороне оценщика |
| env | `auditgym/serve.py`, `auditgym/client.py` | `Env` | процесс и клиент §5.1 |
| env | `auditgym/calc.py` | `calculate(expression: str) -> str`, `CalcError` | безопасная арифметика на `Decimal` |
| qa | `auditgym/solver.py` | `solve(public: dict) -> list[dict]` | выводы только по открытым записям; не импортирует `generate`, `inject`, `truth` |
| qa | `auditgym/verify.py` | `verify(episode: dict) -> list[str]` | список нарушений; пустой список — эпизод корректен |
| qa | `policies/*.py` | `class Policy: act(observation) -> action` | случайный, сверка, «пометить всё», oracle, решатель |

`matcher` получает `public`, чтобы проверить существование ID. Никто, кроме `evaluator` и `verify`, не читает `episode["private"]`.

## 8. Заглушки до готовности соседей

- `tests/fixtures/tiny/` — один маленький эпизод, написанный руками (12 строк книги, 3 счёта, 2 платежа, 1 дубль). Его кладёт зона `gen` в первый час; на нём `env`, `judge` и `qa` работают до появления генератора.
- Пока нет `judge`, `evaluator` считает награду функцией-заглушкой `judge()` с тем же возвращаемым словарём.
