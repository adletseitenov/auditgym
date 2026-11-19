# Схемы

JSON Schema (draft 2020-12), по одному файлу на форму данных. Версия схем — `SCHEMA_VERSION = "1"` (`auditgym/__init__.py`); она записана в `manifest.json` и `truth.json` каждого эпизода, версия конфигурации (`config_version`) — в `manifest.json` и в `provenance.json`. Менять схему можно только вместе с `docs/CONTRACTS.md` и с повышением версии.

Файлы используют только ключевые слова `type` (имя или список), `required`, `properties`, `additionalProperties`, `enum`, `items`, `minimum`, `maximum`, `pattern`, `minItems`, `maxItems`. Ссылок `$ref` и `oneOf` нет, поэтому составные формы разложены на файлы: наблюдение проверяется файлом `observation-<type>`, а вложенные `manifest`, `policy` и записи таблиц — своими схемами по полю `table`. Проверяет это `tests/test_schemas.py` валидатором `tools/schema_lite.py` (стандартная библиотека; незнакомое ключевое слово в схеме — ошибка, так что файлы не выйдут за подмножество).

| Файл | Что описывает | Где живёт |
|---|---|---|
| `config.schema.json` | `config` для `reset` (все ключи необязательны; нормализованный конфиг из `provenance.json` проходит ту же схему) | вход среды |
| `manifest.schema.json`, `policy.schema.json` | `public/manifest.json`, `public/policy.json` | открытые файлы |
| `ledger_row`, `invoice`, `payment`, `fx_row` | строка `ledger.csv`, строка `invoices.jsonl`, `payments.jsonl`, `fx.csv` | открытые файлы и записи в наблюдениях |
| `truth.schema.json` | `private/truth.json` | закрытый файл |
| `action-list_records`, `action-get_record`, `action-get_document`, `action-calculator`, `action-submit` | пять действий в канонической форме (парсер принимает и синонимы, `docs/CONTRACTS.md` §5.2) | действие агента |
| `findings.schema.json` | сообщение `findings` (то, что лежит в `submit`) | действие `submit` |
| `observation-reset`, `-records`, `-record`, `-document`, `-calculation`, `-error`, `-final` | наблюдения по полю `type`; лишних ключей нет | ответ среды |
| `info-empty`, `info-final` | `info` до конца эпизода и в `eval` (пустой) и пять чисел в конце `train` | ответ среды |
| `journal_row.schema.json` | строка журнала оценщика (`runs/results.jsonl`) | закрытый журнал |
| `trajectory_row.schema.json` | строка `trajectories.jsonl` | сдача |

Чего схемы не выражают и что проверяет код: ограничения между ключами конфигурации (плотность ошибок, число проводок, число страниц — `auditgym/config.py`), равенства книг (`auditgym/verify.py`), размер сообщения 256 КиБ и число уникальных ID в доказательствах (`auditgym/findings.py`).
