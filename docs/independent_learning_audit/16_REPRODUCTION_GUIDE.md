# 16. Воспроизведение без изменения обучения

## Границы

Текущий learner по просьбе пользователя paused. Этот пакет не содержит команд автоматического resume, прошивки или приводов. Старые документы и очищенная переписка могут содержать исторические команды записи/движения: **они включены как доказательства, а не инструкции к запуску**.

Основной код закреплён в 1da8ecb; документы bded49e. Точный новый audit commit — в Git-истории/итоговом ответе. Все checkpoints остаются на внешнем диске. Размеры/SHA — checkpoints.csv. Сам ZIP не обязан содержать веса, поэтому на другом компьютере сначала нужен явно выбранный файл с соответствующим хешем.

## Проверить переносимость пакета

Из распакованного каталога:

```sh
python3 scripts/verify_packet.py .
python3 -m unittest discover -s tests -v
```

Это проверка файлов, checksums и сохранённых чисел, без neural training/physics/ROS. Она не доказывает успешную уборку.

## Повторить конкретный inference

На исходном Mac, с подключённым диском и правильной средой, выбрать НОВОЙ output-каталог, чтобы не переписать доказательства:

```sh
env PYTHONPATH=.:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /Volumes/ExplorerSim/env/bin/python \
  docs/independent_learning_audit/scripts/replay_refusal.py \
  --repo /Users/vladromenko/Desktop/Projects/Explorer-Learning \
  --diagnostic /Volumes/ExplorerSim/ExplorerSim/diagnostics/live-room-395c6a0365d9 \
  --output /tmp/rosmaster-refusal-reproduction-new
```

Скрипт загружает сохранённый checkpoint на CPU, eval/no_grad, записывает normalized inputs и полный chunk; проверяет trace RGB hash и сравнивает первый denormalized target с 695. Двигатели и environment.step не вызываются. Clock/RGB counterfactuals — inference, не физические эксперименты. Нельзя по ним одним объявить clock shortcut окончательной причиной.

## Повторить инвентаризацию

`scripts/collect_inventory.py` читает manifests/jobs/processes и при явном `--hash-checkpoints` хеширует веса потоково. Output обязан быть вне experiment root. Полный проход нескольких десятков GiB может занимать минуты и конкурировать за I/O; если обучение когда-нибудь возобновлено, не повторять его без нужды.

`scripts/analyze_dataset.py` читает текущие recording sources по точному samples hash, проверяет весь samples stream и 3RGB samples на запись, строит ограниченную визуальную статистику. Он не является полным image semantic audit. Изменение current dataset не предусмотрено.

## Проверка исторических физических результатов

Не запускать старые mission scripts. Прочитать selected report JSON, trace и source commit; сравнить признаки stage/teacher/student/full/local, protocol, seed, weightsSHA. Для настоящего повторного физического эксперимента нужен новый отдельный каталог, известный saved state/reset, resource budget и неизменённые safety gates. Такой эксперимент аудитом не выполнялся.

## Недоступность аппаратуры

Read-only SSH19.10 к существующим aliases завершился DNS failure. Не заменять это обходом safety, сканированием чужих устройств или запуском новых UART владельцев. Перед реальными тестами нужен отдельный non-actuating inventory и разрешение на конкретные движения.

## Повторить полный подсчёт state/clock

На исходном Mac: `python3 scripts/recheck_runtime_distribution.py --packet . --output /tmp/rosmaster-range-new.json` из распакованного пакета. Скрипт проверяет SHA всех исходных streams и считает state/фазы заново. Он требует новый выходной файл вне дерева экспериментов. Clock берётся **точно как в export_dataset: actor_elapsed_s, иначе index/20**, а не из абсолютного simulation_time. Первичная версия вспомогательного recheck ошибочно использовала simulation_time и дала731 вместо728 retreat строк; сверка поймала это расхождение. После исправления нужно получить728/6/1. В первичном forensic follow-up использовался правильный учебный clock; старый результат не подгонялся.
