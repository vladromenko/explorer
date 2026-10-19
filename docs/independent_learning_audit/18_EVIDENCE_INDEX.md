# 18. Индекс доказательств

Каждая строка — ограниченный проверяемый вывод. VERIFIED не переносится с наличия кода на успех задачи. Для исходников точные функции и диапазоны строк с Git-ссылками находятся в `evidence/SOURCE_FUNCTION_INDEX.md`; содержимое и SHA — в `evidence/source_manifest.json`. Для run/checkpoint/scene используются собственные IDs первичных JSON, а не пересказ сообщения.

| ID | Статус | Вывод | Первичное доказательство | Разделы |
| --- | --- | --- | --- | --- |
| E01 | VERIFIED | История доступна с27.09; более ранняя часть не восстановлена | evidence/conversation_index.csv; evidence/git_history.tsv | 19;02 |
| E02 | VERIFIED | Раннее ready относилось к компонентам, полного mobile executor не было | 19 D04 + source/docs/LEARNING-IMPLEMENTATION.ru.md | 19;02;09 |
| E03 | VERIFIED | 15–17.10 parameter search временно не обучал ACT | source/simulator/skill_training.py;19 D15/D18 | 04;19 |
| E04 | VERIFIED | Реальные gradient runs и числовые checkpoints существуют | experiments.csv;checkpoints.csv;evidence/historical/checkpoint-007000-proof.json | 05;14;17 |
| E05 | VERIFIED | Текущий набор739 records/629959 raw rows | dataset_statistics.json;evidence/recording_statistics.json;scripts/analyze_dataset.py | 06 |
| E06 | VERIFIED | Command estimate nextstate равен прошлой action; это не measured joints | dataset_statistics.json;source/src/mobile_policy_contract.py; пользовательское уточнение | 03;06;07 |
| E07 | VERIFIED | Плечо123° вне полного диапазона учебных наблюдений | evidence/runtime_distribution_followup.json;examples/refusal-695/actions.jsonl | 05;06 |
| E08 | VERIFIED | Clock40.75s:728retreat/6unhook/1place; runtime остаётсяpick | evidence/runtime_distribution_followup.json | 05;06;07 |
| E09 | VERIFIED | Fresh gripper target75.8205 и отказ при48° точно воспроизводятся | examples/refusal-695/refusal_forensics.json;source/simulator/room_policy.py::issued_command | 05 |
| E10 | UNKNOWN | Единственная причинная доля clock/state/vision/teacher для скачка | examples/refusal-695/refusal_forensics.json counterfactuals and limitations | 05;11;12 |
| E11 | VERIFIED | Четыре cross-split группы точных RGB в ограниченной выборке | evidence/visual_and_state_followup.json | 06 |
| E12 | UNKNOWN | Полная semantic scene leakage и противоречивые targets всех кадров | visual sample scope2217 only; recording statistics | 06;15 |
| E13 | VERIFIED | Teacher до3.6°/кадр против runtime2°; rawguard20° | dataset_statistics.json;source/src/mobile_policy_contract.py;source/simulator/room_policy.py | 07;05 |
| E14 | VERIFIED | Timing repair10/10s не устранил последующий fresh refusal | evidence/historical/cadence-physics-regression.json;08 failures | 07;08 |
| E15 | VERIFIED | Локальные5/10 не равны полной миссии | system_snapshot.json recovery history; selected full trial/result.json | 01;17 |
| E16 | VERIFIED | Сохранены667 checkpointpaths/649 files/644 hashes | checkpoints.csv;evidence/checkpoint_inventory_summary.json | 14 |
| E17 | VERIFIED | Пользовательская пауза действует;29000 полный,29078 job | system_snapshot.json | 17 |
| E18 | NOT_IMPLEMENTED | Универсальное моторное обучение произвольной задаче из языка | source/simulator/language_learning.py;source/simulator/room_learning.py;09 | 09;04 |
| E19 | UNKNOWN | Реальные torque/lag/material/current latency и надёжность transfer | 03 таблица;15 | 03;15 |
| E20 | STRONGLY_SUPPORTED | Поздний контракт/сквозная проверка/потеря neuralconsumer объясняют существенную потерю времени | 26 decision chains19, run histories02, failure database08 | 11;19 |

Хеш каждого включённого файла приведён в audit_manifest.json/SHA256SUMS. Исходные абсолютные пути вне архива — указатели на доступные владельцу данные; переносимый небольшой пример отказа уже включён. Прежние сообщения могут содержать ошибочные выводы: наличие цитаты доказывает сказанное, но не истинность технического утверждения.
