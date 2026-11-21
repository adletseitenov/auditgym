# Адаптер жюри: judge_adapter.py

Основание: определения проверок FinTech, опубликованные 20.11 («Единый способ подключения», Блоки 1–4). Адаптер тонкий: он переводит оболочку действий и ответы, ничего не выбирает за политику и не меняет задачу. Весь код — `judge_adapter.py` (корень репозитория), прогон базовых политик — `tools/adapter_run.py`, парные проверки и матрица D01–D20 — `tools/adapter_pairs.py`, тесты — `tests/adapter/`.

## Запуск

- Рабочая директория: корень репозитория (в контейнере `/app`, `COPY . /app`, `.dockerignore` файл не исключает). Зависимостей нет, Python ≥ 3.11.
- Подключение: `sys.path.insert(0, "/app"); import judge_adapter` — точки входа `generate_episode`, `make_env`, `score_episode`.
- Демонстрация: `python judge_adapter.py --demo [--seeds N]`: четыре базовые политики (`random_explorer` с фиксированным seed 7, `reconcile`, `flag_all`, `oracle`) на четырёх сгенерированных эпизодах, только через `make_env().reset/step`; печатает, что политика видела награду 0 и что награда журнала равна `score_episode`.
- Проверка: `python -m auditgym check` (тесты адаптера входят в быстрый режим; ожидание реального лимита 60 с — только в `--full`).

## Интерфейс

`generate_episode(seed, config, output_dir)` — `auditgym.export.build_episode` + `storage.write_episode`: тот же генератор и формат, что читает `Env.from_files`. Секрет оценщика из окружения (`AUDITGYM_SECRET_FILE`/`AUDITGYM_SECRET`) подмешивается так же, как в `export`; без него эпизод воспроизводится по seed и конфигурации.

`make_env(journal_path=None, server_command=None)` → `AdapterEnv`:

| `reset(seed, config)` | что происходит |
|---|---|
| `{"mode": "closed", "episode_dir": D, "team_config": ...}` | оценщик `eval` (отдельный процесс, `Env.from_files(D, mode="eval")`, `--allow-load`) грузит именно D; seed и `team_config` игнорируются, ID не меняются; повторный `reset` на тот же D перезапускает эпизод с полным бюджетом (брошенный прогон попадает в журнал как `abandoned`) |
| `{"mode": "train", "episode_dir": D}` | то же в `train`: терминальная награда в `reward` и `info` |
| `{"mode": "train", "team_config": C}` | `Env(mode="train").reset(seed, C)`: наш генератор |
| `{"mode": "closed"}` без каталога | генерация в оценщике `eval` по `seed` и `team_config` (без секрета; секрет оценщика задаёт `server_command`, см. `docs/CLOSED-RUN.md`) |

Неверный `config`, неизвестный режим или нечитаемый каталог — `AdapterError(code)` на `reset` (код фиксированный, без пути). Секрет из окружения адаптер оценщику не передаёт.

Закрытый режим: `reward` — `0.0`, `info` — `{}` на каждом шаге, включая завершение. Истинная награда, TP/FP/FN и `inspection_steps` — в журнале оценщика (`journal_path`, по умолчанию `$AUDITGYM_RESULTS` или `runs/results.jsonl`); для стороны жюри есть `env.last_result()` (последняя строка `finished` текущего эпизода) и `env.inspection_steps`. Политика в закрытом прогоне получает только возвращаемое `step`.

`score_episode(episode_dir, findings, inspection_steps)` — `auditgym.judge.judge` над файлами эпизода; ответ ровно из семи ключей; невалидная структура — `return = -1`, остальное `null`, строка `{"event": "score_error", "episode_id", "structure_error": <код>}` в `$AUDITGYM_SCORE_LOG` (по умолчанию `runs/score_errors.jsonl`; отдельный файл, чтобы не рвать подписанную цепочку журнала). `findings = None` — пустой ответ усечённого эпизода.

## Оболочка действий → внутренний формат

| Оболочка | Внутреннее действие | Результат (`result`) |
|---|---|---|
| `list_records {table, cursor, limit}` | то же; первый cursor `null`, страница до 50 | `{"records": [...], "next_cursor": null или строка}` |
| `get_record {table, id}` | то же | объект записи (для `fx` ключ `date\|src\|dst`; `date:src:dst` принимается и переводится) |
| `get_document {doc_id}` | то же | объект счёта или платежа |
| `calculator {expression}` | то же, грамматика только арифметика (`auditgym/calc.py`) | десятичная строка: `0.1+0.2` → `"0.3"` |
| `submit {findings}` | `submit`, `findings` = объект `{episode_id, findings: [...]}` | `{"accepted": true}`; неверная структура — `error.code = "invalid_findings"`, `result = null`, эпизод завершён, награда −1 только у оценщика |

Имя действия сравнивается точно (регистр и пробелы). Неизвестное имя, не-объект, `args` не объект — ограниченная ошибка (`unknown_action`, `bad_arguments`) и потраченный шаг. Таблицы совпадают с каноническими (`ledger`, `invoices`, `payments`, `fx`), переводить нечего.

Первое наблюдение — ровно `{"task", "manifest", "policy", "remaining_steps": 100}`; `task` — строка: текст задачи плюс схема findings, лимиты и правила ответа по типам (то, что наша среда кладёт в `task`). Следующие — ровно `{"result", "error", "remaining_steps"}`, ошибка `{"code", "message"}` из фиксированного списка, без путей, seed, ID запроса и закрытых данных. Принятие `submit` определяется тем же разбором структуры, что у оценщика; о правильности ответ ничего не говорит (тест `test_a_wrong_but_well_formed_answer_is_accepted_...`).

## Бюджет и время

- 100 действий, `submit` входит в 100, но не в `inspection_steps`; все предыдущие шаги, включая ошибочные, входят (C14–C16, журнал: `inspection_steps`, `steps_total`).
- 100 действий без `submit`: `truncated = true`, пустой ответ (`fn` = число ошибок эталона), `inspection_steps = 100` (C17). После конца любое действие даёт `episode_over`, журнал не меняется (C15, C17).
- 60 с на эпизод: часы оценщика идут от `reset`/`load` и проверяются на каждом шаге, `reset` и закрытии; шаг после 60 с даёт `truncated = true` и пустой ответ, учитываются только реально сделанные действия (C18). Медленная политика не обходит лимит: время считается по стенным часам процесса-оценщика.

## Проверки жюри → наши тесты (92 из 92 через адаптер)

Каждая строка — тест, который идёт через `generate_episode`, `make_env` и `score_episode` (фикстуры и каталоги эпизодов на диске, `reset` с `episode_dir`). Таблица сверяется с кодом тестом `tests/adapter/test_table.py`: ровно эти 92 ID, каждое имя теста существует. B3-NN — тесты 1–20 Блока 3.

| ID | что проверяется | тест (`tests/adapter/`) | статус |
|---|---|---|---|
| C01 | reset на C: пара observation, info, JSON | `test_contract.C01to04.test_C01_reset_returns_observation_and_info_json` | pass |
| C02 | первое observation: task, manifest, policy, бюджет, без seed/эталона | `test_contract.C01to04.test_C02_first_observation_has_task_manifest_policy_budget_and_no_secret` | pass |
| C03 | manifest, числа сверены с данными | `test_contract.C01to04.test_C03_manifest_matches_the_data` | pass |
| C04 | list_records: пятёрка step, булевы флаги | `test_contract.C01to04.test_C04_list_records_returns_the_five_tuple` | pass |
| C05 | L: страницы по 50, без пропусков и повторов | `test_contract.C05to08.test_C05_pagination_of_L_has_no_gaps_and_no_repeats` | pass |
| C06 | один курсор дважды | `test_contract.C05to08.test_C06_same_cursor_twice_gives_the_same_page` | pass |
| C07 | get_record R1 | `test_contract.C05to08.test_C07_get_record_returns_the_right_record_only` | pass |
| C08 | get_document D1 | `test_contract.C05to08.test_C08_get_document_returns_the_right_document` | pass |
| C09 | calculator `0.1+0.2` = `"0.3"` | `test_contract.C09to14.test_C09_calculator_is_decimal_arithmetic` | pass |
| C10 | calculator вне арифметики | `test_contract.C09to14.test_C10_calculator_refuses_anything_but_arithmetic` | pass |
| C11 | get_record U | `test_contract.C09to14.test_C11_get_record_of_U_is_a_bounded_error` | pass |
| C12 | get_document U | `test_contract.C09to14.test_C12_get_document_of_U_is_a_bounded_error` | pass |
| C13 | неизвестное имя действия | `test_contract.C09to14.test_C13_unknown_action_name_is_a_bounded_error_and_a_step` | pass |
| C14 | ошибка не даёт бесплатного шага | `test_contract.C09to14.test_C14_errors_do_not_give_free_actions_and_do_not_spoil_access` | pass |
| C15 | submit завершает, дальше ничего не меняется | `test_contract.C15to19.test_C15_submit_ends_the_episode_and_later_actions_change_nothing` | pass |
| C16 | submit как действие 100 | `test_contract.C15to19.test_C16_submit_as_action_100_is_accepted` | pass |
| C17 | 100 действий без submit: truncated | `test_contract.C15to19.test_C17_one_hundred_inspections_truncate_with_empty_findings` | pass |
| C18 | лимит времени 60 с | `test_contract.C15to19.test_C18_time_limit_truncates_with_empty_findings` | pass (лимит 1 с; настоящие 60 с: `test_contract.RealTimeLimitTest`, `--full`) |
| C19 | закрытый режим: награда 0, истина у оценщика | `test_contract.C15to19.test_C19_closed_mode_gives_the_policy_nothing_of_the_reward` | pass |
| C20 | нет утечек, нет shell/SQL/FS/сети | `test_contract.C20.test_C20_nothing_leaks_and_the_tools_open_no_door` | pass (код и изоляцию жюри смотрит отдельно) |
| D01 | clean, plain reconciliation (no distractors) | `test_scenarios.ScenarioTest.test_D01` | pass |
| D02 | clean, a correctly split payment | `test_scenarios.ScenarioTest.test_D02` | pass |
| D03 | clean, a correct refund | `test_scenarios.ScenarioTest.test_D03` | pass |
| D04 | clean, legitimately repeated amounts | `test_scenarios.ScenarioTest.test_D04` | pass |
| D05 | clean, a period/offset boundary by policy | `test_scenarios.ScenarioTest.test_D05` | pass |
| D06 | duplicate invoice | `test_scenarios.ScenarioTest.test_D06` | pass |
| D07 | duplicate invoice, EUR | `test_scenarios.ScenarioTest.test_D07` | pass |
| D08 | duplicate invoice, two currencies, another offset | `test_scenarios.ScenarioTest.test_D08` | pass |
| D09 | allocation gap, payment allocation entirely missing | `test_scenarios.ScenarioTest.test_D09` | pass |
| D10 | allocation gap, partial allocation, EUR | `test_scenarios.ScenarioTest.test_D10` | pass |
| D11 | allocation gap, entirely missing, CNY, another offset | `test_scenarios.ScenarioTest.test_D11` | pass |
| D12 | fx error, wrong rate date | `test_scenarios.ScenarioTest.test_D12` | pass |
| D13 | fx error, wrong rate value | `test_scenarios.ScenarioTest.test_D13` | pass |
| D14 | period cutoff | `test_scenarios.ScenarioTest.test_D14` | pass |
| D15 | period cutoff, far-west offset | `test_scenarios.ScenarioTest.test_D15` | pass |
| D16 | interaction I1 | `test_scenarios.ScenarioTest.test_D16` | pass |
| D17 | interaction I1, CNY, another offset | `test_scenarios.ScenarioTest.test_D17` | pass |
| D18 | interaction I1 with other errors, two currencies | `test_scenarios.ScenarioTest.test_D18` | pass |
| D19 | interaction I2, EUR | `test_scenarios.ScenarioTest.test_D19` | pass |
| D20 | interaction I2, USD and RUB, another offset | `test_scenarios.ScenarioTest.test_D20` | pass |
| B3-01 | один правильный F: награда 0.98 | `test_block3.Block3Test.test_01_one_correct_finding` | pass |
| B3-02 | пустые findings: -1 | `test_block3.Block3Test.test_02_empty_findings` | pass |
| B3-03 | пустой эталон, пустые findings: F1=1, 0.98 | `test_block3.Block3Test.test_03_empty_truth_empty_findings` | pass |
| B3-04 | пустой эталон, ложный вывод: -1 | `test_block3.Block3Test.test_04_empty_truth_one_false_finding` | pass |
| B3-05 | два одинаковых F: 47/150 | `test_block3.Block3Test.test_05_two_identical_findings` | pass |
| B3-06 | неправильный тип | `test_block3.Block3Test.test_06_wrong_type` | pass |
| B3-07 | другая существующая запись | `test_block3.Block3Test.test_07_other_existing_record` | pass |
| B3-08 | сумма 10001 при τ=0 | `test_block3.Block3Test.test_08_amount_off_by_one_at_tolerance_zero` | pass |
| B3-09 | другая валюта | `test_block3.Block3Test.test_09_other_currency` | pass |
| B3-10 | только D1 вместо {D1, D2} | `test_block3.Block3Test.test_10_only_d1` | pass |
| B3-11 | несуществующий evidence_id вместо D2 | `test_block3.Block3Test.test_11_missing_evidence_id_instead_of_d2` | pass |
| B3-12 | массовая пометка 20 выводов: -871/1050 | `test_block3.Block3Test.test_12_mass_flagging_five_records_four_types` | pass |
| B3-13 | две ошибки, найдена одна: 47/150 | `test_block3.Block3Test.test_13_two_errors_one_found` | pass |
| B3-14 | две ошибки, один верный и один ложный: -0.02 | `test_block3.Block3Test.test_14_two_errors_one_found_one_false` | pass |
| B3-15 | inspection_steps=0: награда 1 | `test_block3.Block3Test.test_15_zero_inspection_steps` | pass |
| B3-16 | inspection_steps=50: 0.90 | `test_block3.Block3Test.test_16_fifty_inspection_steps` | pass |
| B3-17 | inspection_steps=100: 0.80 | `test_block3.Block3Test.test_17_hundred_inspection_steps` | pass (формула; эпизод без submit на 100-м шаге не получить) |
| B3-18 | 31 вывод: структура, -1 | `test_block3.Block3Test.test_18_thirty_one_findings` | pass |
| B3-19 | 11 различных evidence_ids: структура, -1 | `test_block3.Block3Test.test_19_eleven_different_evidence_ids` | pass |
| B3-20 | explanation 1001 символ: структура, -1 | `test_block3.Block3Test.test_20_explanation_of_1001_ascii_characters` | pass |
| M01 | affected_ids=[R2, R1, R1] — то же множество | `test_block3.OpenExamplesTest.test_M01_affected_ids_repeat_is_the_same_set` | pass |
| M02 | лишняя существующая запись: FP=1, FN=1 | `test_block3.OpenExamplesTest.test_M02_extra_existing_unaffected_record` | pass |
| M03 | evidence [D2, D1, D3]: TP | `test_block3.OpenExamplesTest.test_M03_superset_with_existing_available_evidence` | pass |
| M04 | evidence [D1, D2, U]: FP=1, FN=1 | `test_block3.OpenExamplesTest.test_M04_unavailable_id_beside_the_minimal_set` | pass |
| M05 | 11 элементов (10 повторов D1): структура, -1 | `test_block3.OpenExamplesTest.test_M05_raw_limit_before_repeats_collapse` | pass |
| M06 | валюта другим регистром или с пробелом | `test_block3.OpenExamplesTest.test_M06_currency_is_exact` | pass |
| M07 | сумма 10001 при τ=0 | `test_block3.OpenExamplesTest.test_M07_amount_one_off_with_tau_zero` | pass |
| M08 | τ=1: 10001 проходит, 10002 нет | `test_block3.OpenExamplesTest.test_M08_tolerance_one_is_inclusive` | pass |
| M09 | сумма строкой или true: структура, -1 | `test_block3.OpenExamplesTest.test_M09_amount_string_or_bool` | pass |
| M10 | максимальное паросочетание: TP=2 | `test_block3.OpenExamplesTest.test_M10_maximum_matching_beats_greedy` | pass (какой вывод взял какую ошибку: `tests/judge`, `matcher.match`; здесь TP/FP/FN и награда в обоих порядках) |
| M11 | оба вывода подходят обеим: TP=2, порядок не важен | `test_block3.OpenExamplesTest.test_M11_both_fit_both_and_order_does_not_matter` | pass (какой вывод взял какую ошибку: `tests/judge`, `matcher.match`; здесь TP/FP/FN и награда в обоих порядках) |
| M12 | два одинаковых правильных вывода: TP=1, FP=1 | `test_block3.OpenExamplesTest.test_M12_two_identical_correct_findings` | pass |
| P01 | D01: перестановка | `test_pairs.PairsTest.test_P01_D01` | pass |
| P02 | D02: перестановка | `test_pairs.PairsTest.test_P02_D02` | pass |
| P03 | D03: перестановка | `test_pairs.PairsTest.test_P03_D03` | pass |
| P04 | D04: перестановка | `test_pairs.PairsTest.test_P04_D04` | pass |
| P05 | D05: перестановка | `test_pairs.PairsTest.test_P05_D05` | pass |
| P06 | D06: замена ID | `test_pairs.PairsTest.test_P06_D06` | pass |
| P07 | D07: замена ID | `test_pairs.PairsTest.test_P07_D07` | pass |
| P08 | D08: замена ID | `test_pairs.PairsTest.test_P08_D08` | pass |
| P09 | D09: замена ID | `test_pairs.PairsTest.test_P09_D09` | pass |
| P10 | D10: замена ID | `test_pairs.PairsTest.test_P10_D10` | pass |
| P11 | D11: перестановка и замена ID | `test_pairs.PairsTest.test_P11_D11` | pass |
| P12 | D12: перестановка и замена ID | `test_pairs.PairsTest.test_P12_D12` | pass |
| P13 | D13: перестановка и замена ID | `test_pairs.PairsTest.test_P13_D13` | pass |
| P14 | D14: перестановка и замена ID | `test_pairs.PairsTest.test_P14_D14` | pass |
| P15 | D15: перестановка и замена ID | `test_pairs.PairsTest.test_P15_D15` | pass |
| P16 | D16: перестановка и замена ID | `test_pairs.PairsTest.test_P16_D16` | pass |
| P17 | D17: перестановка и замена ID | `test_pairs.PairsTest.test_P17_D17` | pass |
| P18 | D18: перестановка и замена ID | `test_pairs.PairsTest.test_P18_D18` | pass |
| P19 | D19: перестановка и замена ID | `test_pairs.PairsTest.test_P19_D19` | pass |
| P20 | D20: перестановка и замена ID | `test_pairs.PairsTest.test_P20_D20` | pass |

Как устроены группы.

- **Блок 1** (C01–C20): фикстуры C, E, L, U — `tests/adapter/helpers.py`, все делает `generate_episode`.
- **Блок 2** (D01–D20, матрица `tools/adapter_pairs.MATRIX`): эпизод ищется перебором seed по предикату над данными (расщеплённый платёж, возврат, повторяющиеся суммы, граница пояса, `wrong_date`/`wrong_rate`, вариант `missing`/`partial` пропуска распределения, состав сценариев взаимодействия). D09 и D11 — `allocation_gap_mode: missing`, D10 — `partial`; D12 — неверная дата курса, D13 — неверный курс; D16–D18 — взаимодействие I1, D19–D20 — I2, у каждого свои валюты, пояс и набор ошибок. Для каждого D: `generate_episode` дважды даёт одни и те же байты; `auditgym.verify` пуст (книги согласованы до внесения ошибки, инъекция заявленная); суммы дебета и кредита равны до и после; truth непуст кроме D01–D05; независимый пересчёт по открытым строкам и снимку `pre_*` (дубль: два проведённых счёта с одной сигнатурой; разрыв: непривязанный кредит равен сумме truth, при `missing` закрытия дебиторки нет совсем, при `partial` закрытие плюс разрыв равны прежнему закрытию); все `affected_ids` и все ID всех наборов доказательств достаются через `make_env` в режиме `train` (`get_record`/`get_document`) и равны записям в файлах; форматы §03 (время со смещением, деньги целыми, FX строкой-десятичной, явные `null`) проверены на том, что отдаёт `list_records`; oracle получает return 1.0 через `score_episode` и через `step`. Это проверка нашими средствами: независимой сверки жюри она не заменяет (`verify` наш).
- **Блок 3 и M01–M12** (`tests/adapter/test_block3.py`, юнит-слой — `tests/judge/test_official_block3.py`): открытые данные — настоящий сгенерированный эпизод; truth страницы (A, R1/R2, D1/D2/D3, 10000 KZT, τ=0, для M08 τ=1 в policy копии) записан в КОПИИ каталога из реальных ID записей, U выбран среди отсутствующих. Каждый случай идёт через `score_episode(episode_dir, findings, inspection_steps)` и, кроме формульных тестов с 100 шагами, ещё и в закрытом режиме `make_env` (`reset`, `inspection_steps` шагов, `submit`): строка журнала оценщика равна по TP/FP/FN, награде и числу шагов, политика видит награду 0. Тесты 3, 4 и `RealEpisodeTest` работают на нетронутых эпизодах, где truth — настоящая инъекция (`verify` пуст). Допуск 1e-9 абсолютный. Тесты 15–17 передают `inspection_steps` напрямую (15 и 16 ещё и в эпизодном прогоне). Для M10/M11 граф рёбер задан наборами доказательств; кто какую ошибку взял, видит только юнит-слой.
- **Блок 4** (P01–P20, `python -m tools.adapter_pairs`, все 20 пар, около 8 с): P01–P05 перестановка (D01–D05), P06–P10 замена ID (D06–D10), P11–P20 обе (D11–D20). Копию делает `tools/transform.py` (фиксированные параметры, исходный каталог не меняется, проверяется хэшем). Для каждой пары: повтор на исходном эпизоде (тот же env дважды и свежий env) совпадает по JSON-значениям в записях, переходах и наградах; oracle с переназначенными ID при том же `inspection_steps` (5) даёт те же TP/FP/FN/P/R/F1/return по `score_episode` и по журналу; выводы после обратного отображения равны исходным; все таблицы через `list_records`, отображённые назад, равны исходным; контроль: исходные ID на переименованной копии не засчитываются (tp = 0).

## Известные расхождения и допущения

1. **Журнал `eval` и `verify-journal`.** `Env.from_files(mode="eval")` передаёт `--allow-load`; строки загруженных эпизодов помечены `source=load` (с `source_sha256` каталога под MAC), и `verify-journal` выводит их справочно и возвращает 0 (DECISIONS R): в официальной проверке каждый закрытый эпизод подаётся через `episode_dir`. Награду жюри берёт из `score_episode` или из строк журнала; эти значения тест сверяет.
2. **Результат 100-го действия без submit.** Сервер при усечении отвечает финальным наблюдением без результата последнего действия (`error.code = "truncated"`). Политика, дошедшая до 100 действий без submit, всё равно получает `truncated = true`.
3. **Время.** Усечение по времени выполняется при следующем запросе (шаг, `reset`, закрытие), а не асинхронно; при нём оценщик считает `inspection_steps` по реально выполненным действиям, при усечении по 100 шагам — 100. Счётчик `env.inspection_steps` адаптера при усечении по времени может быть на 1 больше; авторитетна строка журнала. Лимит в тесте C18 ускорен (1 с); настоящие 60 с проверены в `--full`.
4. **Независимая настраиваемость чистых вариантов (D02–D05).** Расщеплённый платёж, возврат, повторяющиеся суммы и граничные даты не имеют отдельных переключателей: все они включаются одним `distractors` (по умолчанию `true`), а конкретный вариант находится перебором seed по предикату над данными. D01 — `distractors: false`.
5. **`allocation_gap`: пропуск и частичный случай.** `allocation_gap_mode` = `partial` (по умолчанию: на `unallocated` уходит часть платежа), `missing` (весь платёж остаётся нераспределённым, закрытия дебиторки нет) или `mixed` (DECISIONS S). D09–D11 содержат оба случая (D09, D11 — `missing`, D10 — `partial`). Платёж без ссылки на счёт (аванс) — по-прежнему законный случай, не ошибка. I1 остаётся частичным.
6. **Ключи FX.** Наш ключ `date|src|dst`; `get_record` для `fx` принимает и `date:src:dst`. В `affected_ids` и `evidence_ids` ключ FX остаётся нашим (oracle берёт их из `truth`). При замене ID ключи FX не меняются.
7. **Преобразование пар** делает `tools/transform.py`; кроме порядка строк он перемешивает `line_items` и `invoice_refs` внутри документа (строже правил жюри; среда этого различия не замечает, это проверено парами).
8. **Закрытый режим без `episode_dir`** генерирует эпизод без секрета (воспроизводим по seed); для скрытых эпизодов жюри использует `generate_episode` и `episode_dir`.
9. **Принятие submit** определяется на стороне адаптера тем же разбором структуры, что у оценщика (`auditgym.findings.parse_findings`); правила структуры ведёт участник зоны matcher.
10. Политики `policies/` исполняются через `PolicyBridge` (`tools/adapter_run.py`): он переводит их плоские действия в оболочку и ответы адаптера в прежний вид. Классы политик не переписывались; в демо политика и адаптер в одном процессе, изоляция — процесс оценщика; в закрытом прогоне жюри политика запускается отдельным процессом.
