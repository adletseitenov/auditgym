# Контракты между зонами

Версия 2, 19.11.2026 (после первого ревью; что изменилось — `docs/DECISIONS.md`, раздел A). Это общий договор четырёх зон: `gen`, `env`, `judge`, `qa`. Каждый пишет свою зону против этого файла и может не ждать остальных. **Менять только вместе:** правка здесь = сообщение в чат команды + повышение `SCHEMA_VERSION`.

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
        "pre_ledger": [row, ...],    # книга до вставки ошибок, с финальными ID
        "pre_invoices": [inv, ...],  # счета до вставки ошибок, с финальными ID
        "pre_payments": [pay, ...],  # платежи до вставки ошибок, с финальными ID
    },
}
```

На диске: `<dir>/public/{manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv}` и `<dir>/private/{truth.json, provenance.json, pre_ledger.csv, pre_invoices.jsonl, pre_payments.jsonl}`. Три файла `pre_*` вместе — полное состояние «до ошибки». Открытые и закрытые файлы всегда в разных каталогах. В CSV отсутствующая ссылка — пустое поле; через инструменты среды она отдаётся как JSON `null`.

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
 "evidence_superset_allowed": true, "period_basis": "document_date",
 "period_interval": "[start, end)",
 "payment_rule": "a payment with one reference moves the invoice's receivable by round_half_even(booked * paid_after / total) - ...",
 "duplicate_signature": "entity, currency, total, line items incl. service period; a duplicate_invoice is two invoices with this signature that are both posted in the ledger (an unposted invoice is never part of one); service_start/service_end only identify duplicates and never affect period_cutoff, which uses the document date alone (period_basis)"}
```
`policy.json` собирается по белому списку ключей (`auditgym.generate.POLICY_KEYS`); ничего из эталона и из конфигурации ошибок в него не попадает. Записи всех таблиц упорядочены по открытому ID (`ledger` — по `row_id`, документы — по `doc_id`, `fx` — по дате и валюте).

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
             "evidence_sets": [["docA", "docB", "rowA_debit", "rowB_debit"], ["docA", "docB", "rowA_debit", "rowB_credit"],
                               ["docA", "docB", "rowA_credit", "rowB_debit"], ["docA", "docB", "rowA_credit", "rowB_credit"]]}]}
```
Типы: `duplicate_invoice`, `allocation_gap`, `fx_error`, `period_cutoff`. `affected_ids` — множество `doc_id` (хранится отсортированным, сравнивается как множество). Каждый набор доказательств — не больше 10 ID, все существуют в открытых записях.

| Тип | `affected_ids` | Сумма и валюта | `evidence_sets` |
|---|---|---|---|
| `duplicate_invoice` | оба счёта | `total_minor`, валюта счёта | четыре набора `[docA, docB, строка A, строка B]`: по одной строке (дебет или кредит) из собственной группы проводок каждого счёта |
| `allocation_gap` | платёж и счёт | `credit_minor` строки на `unallocated`, базовая валюта | `[[doc_платежа, doc_счёта, row_нераспределённых, row_Дт_дебиторки_счёта, row_Кт_дебиторки_платежа]]` |
| `fx_error` | документ | модуль разницы, базовая валюта | счёт: `[[doc, row_Дт, ключ_курса], [doc, row_Кт, ключ_курса]]`; платёж: `[[doc, row_денег, ключ_курса]]`; ключ курса — `"дата|ИСТ|ЦЕЛЬ"` по дате документа в поясе компании |
| `period_cutoff` | документ | сумма документа, валюта документа | `[[doc, row_Дт], [doc, row_Кт]]` |

Точные предикаты — в `docs/SPEC.md`, раздел «Определения ошибок».

### 3.2 provenance.json
```json
{"seed": 91827, "config": {...}, "generator_version": "0.1.0", "attempt": 0,
 "retries": [],
 "steps": [{"operator": "duplicate_invoice", "params": {...}, "issue_id": "t1",
            "before": [...records...], "after": [...records...]}],
 "id_map": {"<внутренний ID>": "<открытый ID>"}}
```
`retries` — причины отброшенных попыток до удавшейся (`{"attempt": 0, "reason": "no target: I1"}`); пусто, если удалась первая. Попыток не больше 32 (`generate.MAX_ATTEMPTS`); если все отброшены (нет цели для оператора, `verify` отклонил эпизод, внутренний сбой), `generate` бросает `ConfigError("no_episode")` с текстом «no valid episode within 32 attempts»: без seed, причин и эталона, тихого результата нет (`tests/test_limits.py`). Записи `before` и `after` — в открытых (финальных) ID.

## 4. Конфигурация

`reset(seed, config)`; `config` может быть `{}` или `null` — берутся значения по умолчанию.

```json
{"config_version": "1",
 "size": {"invoices": [48, 100], "background_pairs": [150, 300]},
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
- Проверки: всего ошибок ≤ 30; не больше одной ошибки на 8 счетов (по нижней границе `size.invoices`); проводок ≤ 2 000; страниц на полное чтение ≤ 60; валюты из списка `KZT, USD, EUR, RUB, CNY`; неизвестный ключ — ошибка. Нарушение → `ConfigError` с коротким кодом, эпизод не создаётся.

## 5. Протокол среды

### 5.1 Вызовы
```python
env = Env(mode="train")                      # клиент; сам запускает процесс-оценщик
observation, info = env.reset(seed, config)
observation, reward, terminated, truncated, info = env.step(action)
env = Env.from_files("<dir>")                # эпизод из каталога, reset() без seed
observation, info = env.reset(seed=7, options={})   # options — синоним config (имя Gymnasium)
```
`reset()` без seed: у `Env.from_files` — тот же эпизод заново (в `eval` так же, как для эпизода по seed: перезапуск посреди эпизода разрешён, брошенный прогон с его шагами пишется в журнал строкой `abandoned`, новый начинается с полным бюджетом в 100 шагов и 60 с; перезапуск до первого шага или после конца эпизода брошенного прогона не оставляет); без файлов — эпизод по seed по умолчанию с конфигурацией по умолчанию: первый раз seed 0, дальше seed последнего удавшегося `reset` + 1 (0, 1, 2…, как у Gymnasium-среды, чей генератор идёт дальше); после самого большого seed последовательность по умолчанию начинается снова с 0. Seed — целое число или строка из одних ASCII-цифр (`"42"`, пробелы по краям допустимы, без знака, точки и других цифр), это то же число; seed длиннее 256 бит (`MAX_SEED_BITS`), любая другая строка, не целый seed и плохой config — `bad_config`. У `Env.from_files` config без seed (кроме пустого) — `bad_config`: у загруженного эпизода нет конфигурации. `config` и `options` вместе — `bad_config`. Отказ в `reset` или `load` не трогает идущий эпизод. Клиент наружу бросает только `EnvError` (в том числе на враждебный объект действия, чей `items()` или `__iter__` бросает исключение: такое действие отправляется как неизвестное) с коротким кодом (`bad_config`, `bad_episode`, `bad_request`, `internal_error`, `bad_mode`, `bad_results_path`, `bad_secret_file`, `bad_server_command`, `timeout`, `server_exited`, `bad_reply`, `closed`; `not_loaded` зарезервирован и не возникает: с C7 `reset()` без seed всегда начинает эпизод); ответа сервера ждёт не дольше `timeout` секунд (по умолчанию 120), затем останавливает его. Ответ на `reset` или `load`, который не является наблюдением `reset`, — `bad_reply` (клиент никогда не отдаёт ответ шага вместо сброса); `reset` длиннее одной строки протокола (1 МиБ) клиент не отправляет — `bad_config`.
Процесс-оценщик: `python -m auditgym serve [--mode train|eval] [--results PATH]`, по одной JSON-строке на запрос и ответ:
```json
→ {"op": "reset", "seed": 7, "config": {}}
← {"observation": {...}, "info": {}}
→ {"op": "step", "action": {...}}
← {"observation": {...}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}
```
В stdout идёт только протокол; диагностика — в stderr. В запросе `reset` ключ `options` — синоним `config`. Журнал оценщика (строка JSON на каждый законченный эпизод — `"event": "finished"`, с кодом ошибки структуры; в строке два счётчика: `inspection_steps` — действия до `submit` (он идёт в награду, сам `submit` не считается) и `steps_total` — все шаги, включая `submit`; в каждой строке есть `"secret": "set"` или `"none"` (был ли задан секрет генерации; значение не пишется), подпись покрывает поле, `verify-journal` печатает в сводке, сколько строк с каким флагом; и на каждый эпизод, брошенный через `reset`/`load` до конца после хотя бы одного шага, — `"event": "abandoned"` с потраченными шагами, без награды и с числом брошенных в этой сессии `abandoned_in_session`; схема `schemas/journal_row.schema.json`; при заданном секрете или ключе журнала у строки есть ещё `seq`, `prev`, `mac` — цепочка HMAC-SHA256, проверка `python -m auditgym verify-journal PATH [--secret-file F] [--expect-episodes N]`, `auditgym/journal.py`: код 0 — цепочка цела; 1 — плохая строка, либо при заданном ключе журнал пустой или без подписи, либо эпизодных строк (`finished`, `abandoned`) меньше N; 2 — проверять нечего: без ключа и без подписи, подпись без ключа, нечитаемый файл) пишется в режиме `eval`: в `--results PATH`, в унаследованный дескриптор `--results-fd N` (так делает `Env` на POSIX) или по умолчанию в `$AUDITGYM_RESULTS`, иначе `runs/results.jsonl` в рабочем каталоге; каталог создаётся, файл — с правами 0600. Целые числа в запросе любой длины. Строка, которую нельзя прочитать, во время эпизода — шаг: строка длиннее 1 МиБ (без `"op"` или с `"op": "step"`) и строка с `"op": "step"`, которая не разбирается (обрезана, слишком глубокая), — шаг с ошибкой `bad_arguments`, а если в ней названо действие `submit` (где угодно в строке, и внутри действия-строки JSON) — `submit` с неверной структурой (награда −1). Длинная строка с другим `"op"` (`reset` с огромным config) — `bad_request`, шаг не тратится; вне эпизода — тоже `bad_request`, после конца эпизода — тот же ответ `episode_over`, что у шага. Действие-строка JSON, которая называет `submit`, но не разбирается, — `submit` с нечитаемыми аргументами (неверная структура).

Эпизод, который идёт, когда вход кончился (EOF), пришёл `close` или сломался вывод, пишется в журнал, если в нём был хотя бы один шаг или вышли его 60 с: `"event": "finished"`, усечение (пустой ответ, награда как у молчания, 100 шагов осмотра; если вышли и 60 с — шаги, сделанные на самом деле), `"ended_by": "close"`. Предел 60 с проверяется и на `reset`/`load`/`close`, а не только на следующем шаге: эпизод, у которого время вышло, пишется как обычное усечение по времени. Эпизод без единого шага (политика видела только наблюдение `reset`) строки не оставляет, и перезапуск загруженного эпизода до первого шага — не новая попытка; такой эпизод, закрытый без шагов после перезапуска, не превращается в строку с наградой −1. Поле `episode_run` — какая это по счёту строка того же `episode_id` в этой сессии оценщика (1, 2, …): повтор того же seed виден сразу. Файл журнала — только обычный файл, не через симлинк (FIFO и симлинк — отказ, а не зависание или запись в чужой файл), права 0600; дескриптор, который клиент передаёт серверу, всегда ≥ 3, сервер отказывает при `--results-fd` меньше 3.

Секрет оценщика (`docs/AUDIT.md` L1): `serve --secret-file PATH`, `$AUDITGYM_SECRET_FILE`, `$AUDITGYM_SECRET` или `Env(secret_file=...)` (`auditgym/secret.py`) передают его в `generate(seed, config, secret=...)`; непригодный файл секрета — ошибка запуска (код 2). Без секрета эпизоды те же, что у открытых seed. `Env` передаёт секрет серверу только явно (`secret_file`): `AUDITGYM_SECRET` и `AUDITGYM_SECRET_FILE` из окружения процесса, создающего `Env`, в окружение сервера не попадают никогда, иначе политика подняла бы свой `Env(mode="train")` с тем же секретом и перебором seed по `episode_id` восстановила бы скрытый эпизод (`Env(mode="eval")` с секретом только в окружении и без явного `secret_file=`/`server_command=` — `EnvError("bad_secret_file")`: отказ, а не тихий прогон без секрета; в `train` — `RuntimeWarning`). Из окружения секрет читает только `python -m auditgym serve`, запущенный харнессом напрямую.

Ограничение (L2) и требования к закрытому прогону: всё, что процесс политики может прочитать или записать от имени своего пользователя (объект `Env`, журнал, в том числе через `/proc/<pid>/fd`, файл секрета, память сервера), политика может прочитать или записать. Поэтому закрытый прогон требует **секрета** (без него журнал не подписан, а эпизод по seed воспроизводится открытым кодом) и **отдельного пользователя или контейнера**: оценщик, его журнал и файл секрета принадлежат другому пользователю или живут в другом контейнере (`Env(server_command=[...])`), политика получает только два канала. Подпись строк (цепочка HMAC-SHA256, `verify-journal`) ловит изменённую, вставленную, переставленную или дописанную без ключа строку, но не строки, отрезанные с конца файла; поэтому харнесс сверяет, что на каждый его `reset` в журнале ровно одна строка с этим `episode_id` и `episode_run` = 1.

### 5.2 Действия
```json
{"action": "list_records", "table": "ledger", "cursor": null, "limit": 50}
{"action": "get_record", "table": "invoices", "id": "…"}
{"action": "get_document", "doc_id": "…"}
{"action": "calculator", "expression": "125000 * 465.37"}
{"action": "submit", "findings": {"episode_id": "…", "findings": [...]}}
```
Таблицы: `ledger`, `invoices`, `payments`, `fx`. Парсер также принимает имя действия в ключах `tool`, `name`, `type` и аргументы во вложенном `args`, `arguments`, `params`, `input` или `parameters` (вложенные важнее плоских). `limit` от 1 до 50, по умолчанию 50; больше 50 обрезается до 50, и ответ `records` тогда несёт `"limit_clamped": true` (поля нет, если обрезки не было); текст задачи говорит об этом. `get_record` для `fx` принимает составной ключ.

Форма запроса нестрогая, смысл строгий (`docs/AUDIT.md` C1–C3, C9):
- действие — словарь или строка с JSON-объектом; имя действия и таблицы — без учёта регистра и пробелов по краям; у таблицы допускаются расширение файла и единственное число (`Ledger`, `ledger.csv`, `invoice`);
- синонимы аргументов: `get_record` — `id`, `record_id`, `row_id`, `doc_id`, `invoice_id`, `payment_id` (берётся первый непустой в этом порядке); `get_document` — `doc_id`, `id`, `document_id`; общий `id` ищется сначала по `doc_id`, затем по `invoice_id`/`payment_id`;
- курсор `null`, `""`, `0`, `"0"` — первая страница; `limit` — целое, строка из цифр или целое число с плавающей точкой (`"50"`, `50.0`);
- `submit` принимает findings.json объектом, строкой JSON, голым списком выводов (подставляется `episode_id` идущего эпизода), плоско в действии (`episode_id` и `findings` рядом; явный `"episode_id": null` передаётся как есть и даёт неверную структуру) или под ключом `"findings.json"` (`findings` и `findings.json` с разными значениями — неверная структура). Голый список (в том числе пустой `[]`) — это выводы текущего эпизода; объект без `episode_id` голым списком не считается и не чинится: это сообщение без обязательного поля, неверная структура, награда −1 (`submit_payload` в `auditgym/actions.py`). Все правила структуры §6 применяются к результату. `submit`, чьи аргументы нельзя прочитать (`"args": 5`, строка не JSON), — неверная структура.
Неизвестные действие, таблица и ID остаются ошибками.

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
`next_cursor` равен `null` на последней странице. `task` содержит: текст задачи (в нём же: `inspection_steps` не считает `submit`, граница `limit`, формы `submit` — объект `{episode_id, findings}` и голый список как findings текущего эпизода), перечень типов ошибок, схему findings, лимиты (30, 10, 1 000, 100 шагов), допуски. Больше в наблюдениях нет ничего: ни времени, ни seed, ни конфигурации, ни счётчиков сверх manifest.

Коды ошибок (тексты фиксированные): `unknown_action`, `unknown_table`, `unknown_id`, `bad_cursor`, `bad_arguments`, `bad_expression`, `episode_over`, `not_reset`. Любая ошибка, кроме двух последних, расходует шаг.

### 5.4 Завершение и награда
- До терминального шага `reward = 0.0`, `info = {}`.
- `submit` → `terminated = true`. Сотый шаг без `submit` → `truncated = true`, ответ пустой, `inspection_steps = 100`, `steps_remaining = 0`. Лимит времени (60 с) → `truncated = true`, ответ пустой, `steps_remaining = 0`, а `inspection_steps` — действия осмотра, сделанные на самом деле (шаг, на котором увидели конец времени, засчитан в `steps_total`, но не выполнялся и в `inspection_steps` не входит); награда по той же формуле (на чистых книгах `1 − 0.002·inspection_steps`).
- `train`: терминальный `reward` — формула кейса; `info = {"reward", "precision", "recall", "f1", "inspection_steps"}`. Никаких вердиктов по отдельным выводам.
- `eval`: политике возвращается `reward = 0.0`, `info = {}`; настоящая награда и разбор пишутся в журнал оценщика (§5.1; одна JSON-строка на эпизод). `reset` посреди эпизода разрешён и изолирует новый эпизод, но брошенный эпизод остаётся в журнале строкой `abandoned` с его шагами: перезапуск не прячет потраченный бюджет.
- Неверная структура выводов → `terminated = true`, награда `-1.0`.
- `step` после конца → ошибка `episode_over`, `reward 0.0`, состояние не меняется и шаг не тратится; флаги и `steps_remaining` — как в финальном ответе: после `submit` `terminated = true`, `truncated = false`, после усечения (по шагам или по времени) всякий следующий шаг отвечает `terminated = false`, `truncated = true`, `steps_remaining = 0`. `step` до `reset` → ошибка `not_reset` (протокольная: `reward 0.0`, `terminated = false`, `truncated = false`, шаг не тратится).

## 6. Выводы агента

```json
{"episode_id": "ep_7f3k2m9q",
 "findings": [{"type": "duplicate_invoice", "affected_ids": ["docA", "docB"],
               "claimed_amount_minor": 125000, "currency": "USD",
               "evidence_ids": ["docA", "docB"], "explanation": "Same customer, items and total."}]}
```
**Неверная структура (награда −1):** не словарь; нет обязательного поля (в том числе объект `{"findings": [...]}` без `episode_id`; голый список `[...]` — другое дело, §5.2); `findings` не список; больше 30 выводов; больше 10 `evidence_ids`; `explanation` не строка или длиннее 1 000 символов; неизвестный `type`; чужой `episode_id`; `claimed_amount_minor` не целое (в том числе `true`, `100.5`, `"100"`, NaN) или меньше нуля; ID не строка; всё сообщение больше 256 КиБ (`auditgym/limits.py`). Обязательных полей вывода пять: `explanation` можно не указывать (по умолчанию пустая строка); целочисленный `float` (`125000.0`) читается как `125000`; `type` и `currency` сравниваются без учёта регистра ASCII и пробелов по краям (AUDIT R2, R5). Повтор ID внутри `affected_ids` одного вывода — тоже неверная структура (один объект назван дважды). Лишние ключи игнорируются; повторы внутри `evidence_ids` схлопываются (oracle может склеить пересекающиеся допустимые наборы), лимит 10 считается по уникальным ID.

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
