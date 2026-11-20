# Что сдано и где проверяется

Полные таблицы, вынесенные из `README.md` без изменений. Компактный вид — в `README.md`.

## Что сдано (CASE §09)

| Материал | Где | Как пересобрать или проверить |
|---|---|---|
| Исходный код, лицензия, зависимости | `auditgym/` (генератор: `world.py`, `books.py`, `inject.py`, `surface.py`, `generate.py`; среда: `evaluator.py`, `serve.py`, `client.py`, `actions.py`), `LICENSE` (MIT), `pyproject.toml` (`dependencies = []`) | `python3 -m auditgym check` |
| README | этот файл | `python3 -m auditgym check` (строка «Инженерия и объяснение» читает README) |
| Matcher и награда, схемы, политики | `auditgym/matcher.py`, `reward.py`, `findings.py`, `judge.py`; `schemas/` (25 файлов JSON Schema); политика эпизода `public/policy.json` (`auditgym/generate.py`, `build_policy`) | `python3 -m tools.attack` (таблица атакующих ответов); `python3 -m unittest tests.test_schemas` |
| Эпизоды | `episodes/train` (10) и `episodes/dev` (10), по `public/` и `private/` | `python3 -m tools.episodes --out episodes` (пересборка побайтно; `make episodes`) |
| Тесты | `tests/` (среда, сопоставление и награда, генератор, решатель, схемы) | `python3 -m unittest discover -s tests` (`make test`) |
| Базовые политики | `policies/`: `random_explorer`, `reconcile`, `flag_all*`, `oracle`, а также `silence` и `solver_policy`; демонстрационная (не базовая) `heuristic` | `python3 -m tools.report --out docs/RESULTS.md` (`make report`) |
| Траектории | `trajectories.jsonl` (поля `episode_id, policy, split, step, action, observation, reward, terminated, truncated`, `split` — `train` или `dev`; в последней строке эпизода ещё `metrics` с P, R, F1; все четыре базовые политики (`random_explorer`, `reconcile`, `flag_all`, `solver_policy`) на всех 20 книгах train и dev, около 12,5 МБ; решатель читает только наблюдения, в них нет эталона). Траектории oracle лежат отдельно, в `runs/trajectories.oracle.jsonl` (не сдаются: в действиях эталон) | `python3 -m tools.trajectories --out trajectories.jsonl --oracle-out runs/trajectories.oracle.jsonl` (`make trajectories`; все политики на всех 20 книгах — поведение по умолчанию, флаг `--all-episodes` оставлен для совместимости) |
| Записка о дизайне | `docs/DESIGN.md` | — |
| Демо | `python3 -m tools.demo` (`make demo`): один эпизод по шагам, разбор оценщика, атаки на тот же эпизод | `python3 -m tools.demo` |

## Требования кейса → код → проверка

| Требование кейса (§07) | Где в коде | Какой тест или проверка |
|---|---|---|
| Четыре типа ошибок | `auditgym/inject.py` (операторы), `auditgym/config.py` (`errors`) | `tests/gen/test_inject.py`, `python3 -m tools.fuzz --seeds 20` |
| Два сценария взаимодействия | `auditgym/inject.py` (`I1`, `I2`) | `tests/gen/test_inject.py` (`InteractionTest`), решётка `tools/fuzz.py` |
| Чистые компании и отвлекающие случаи, ловушки против трюков | `auditgym/world.py` (`distractors`, ловушки A и B), `config.errors = {}` | `tests/gen/test_generate.py` (`CleanCompanyTest`), `tests/gen/test_decoys.py`, `tests/solver/test_solver.py`, `python3 -m tools.tricks`; строка score «clean companies: solver finds nothing» |
| Контракт `reset`/`step`, пять действий, 100 шагов | `auditgym/evaluator.py`, `actions.py`, `serve.py`, `client.py` | `tests/env/test_evaluator.py`, `tests/env/test_audit_env.py` (тест на каждое предложение кейса §03–§04); 15 строк «Контракт и изоляция» |
| Общий matcher и терминальная награда, атакующие ответы | `auditgym/matcher.py`, `reward.py`, `findings.py`, `judge.py` | `tests/judge/` (`test_matcher`, `test_reward`, `test_attacks`, `test_mutations`), `python3 -m tools.attack` |
| Траектории без закрытого эталона | `tools/trajectories.py` | `tests/env/test_trajectories.py`, `tests/test_schemas.py` (`TrajectoriesTest`), `python3 -m tools.leakscan` |
| Изоляция: эталон не в процессе политики | `auditgym/serve.py`, `client.py` (оценщик — отдельный процесс) | `tests/env/test_process.py`, `python3 -m tools.leakscan` |
| Воспроизводимость, смена ID и порядка | `auditgym/rng.py`, `surface.py`, `tools/transform.py` | `tests/test_rng.py`, `python3 -m tools.metamorph`, линтер `tools/lint_determinism.py` |
| Проверка без сети, платных API и GPU; одна команда | `Dockerfile`, `auditgym/__main__.py`, `tools/check_modes.py` | `docker run --rm --network none auditgym`; строка score «one command runs everything offline» |
