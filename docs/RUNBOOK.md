# Запуск и проверка Explorer Learning

Актуальный код находится в `/Users/vladromenko/Desktop/Projects/Explorer-Learning`. Данные, прошлые неудачи, датасеты и веса остаются на ExplorerSim. Движения ROSMASTER этим руководством не разрешаются.

## Открытие и достоверный статус

Откройте существующее приложение **Explorer Simulator** из меню. Оно открывает `http://127.0.0.1:8766/#learning-dashboard`, не возобновляя пользовательскую паузу и не создавая новую очередь. Порт8767 — прежний интерфейс; не используйте его для новой очереди.

В dashboard различаются состояние программы и `training_evidence`: подготовка/первый пакет/реальные обновления/неподтверждённая свежесть. Время heartbeat владельца не является временем последнего обновления весов. ETA относится только к оставшимся обновлениям текущего learner; оценка, новые данные и следующие раунды туда не входят.

Архивные счётчики считаются отдельно и имеют timestamp; «Считаются…» не означает ноль попыток. Открытие страницы не создаёт лишнюю физическую сцену. Ручная сцена запускается явным reset, статус обучения при этом независим. STOP доступен даже при ошибке сцены.

## Read-only команды

В терминале, из корня проекта:

```sh
export PYTHONPATH=.:src
export EXPLORER_SIM_MESH_DIR=/Volumes/ExplorerSim/assets/m3_reference/meshes
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/Volumes/ExplorerSim/env/bin/python bin/queue-room-cleanup.py --status
/Volumes/ExplorerSim/env/bin/python bin/diagnose-live-room.py --status
```

Источники истины:

- `/Volumes/ExplorerSim/ExplorerSim/neural-recovery/current.json` и `924d1ee5ab01/state.json`;
- `/Volumes/ExplorerSim/ExplorerSim/runs/d4f59a63085b/job.json`, `progress.json`, `train.log`, `model/checkpoints/`;
- `/Volumes/ExplorerSim/ExplorerSim/room-learning/current.json` и каталог сохранённого ID;
- `/Volumes/ExplorerSim/ExplorerSim/diagnostics/early-room-current.json`, затем указанные `state.json`, `trial*/progress.json`, `result.json`;
- `/Volumes/ExplorerSim/ExplorerSim/diagnostics/checkpoint-monitor-d4f59a63085b/state.json`.

Новый неизменяемый аудит, без запуска обучения:

```sh
/Volumes/ExplorerSim/env/bin/python bin/audit-learning-run.py \
  --training d4f59a63085b \
  --output /Volumes/ExplorerSim/ExplorerSim/diagnostics/audit-20261019/data-runtime-NEXT.json
```

Не перезаписывайте NEXT существующим именем. Аудит точных SHA/scene splits не заменяет поиск почти одинаковых траекторий и физическую оценку.

## Сохранение, диск и возобновление

«Пауза» должна завершить работников и сохранить контрольное состояние. Не отключайте диск по одному тексту кнопки: дождитесь подтверждения отсутствия активных процессов и штатного извлечения. Ручная пауза не снимается автоматикой.

`bin/resume-paused-learning.py` по умолчанию только проверяет/показывает план. `--apply` — явное возобновление после разрешения пользователя; оно проверяет отсутствие владельцев, ресурсы, чистый Git, сохраняет прежние состояния и устанавливает новую immutable source. Это не команда, которую надо повторять при работающем learner.

Полное resume checkpoint требует optimizer/RNG/config и прежний датасет. Если градиенты ещё не начинались и checkpoint нет, допускается проверенное повторное использование готового экспорта с исходными warm-start весами; это отдельно отмечается в состоянии. Экспорт и неудачи не удаляются. Запрещены два writer в один run.

## Ранняя оценка

Работает один наблюдатель текущего learner. Не запускайте второй. Его исходники и исходники learner заморожены; Git-изменения в рабочем проекте не меняют активные процессы.

Для инженера: `bin/diagnose-live-room.py --retry-repaired --model 34de45050a00` допускается только после содержательного исправления, чистого нового коммита, завершения прежнего диагностика, проверки процесса и ресурсов. Это повтор той же сцены после правки, не новый независимый успех. `--model` требует готовый bundle и правильную lineage. Наблюдатель использует новые стабильные числовые сохранения, не потенциально записываемый `last`.

Для анализа записанного кадра без движения есть `bin/probe-saved-policy.py --model <bundle> --frame <recorded.npz> --output <new.json>`. Он проверяет ресурсный резерв; не требует робота, не выдаёт допуск и не меняет learner. Исторический clock counterfactual не является новой попыткой задачи.

## Проверка исправлений и установка

```sh
/Volumes/ExplorerSim/env/bin/python -m pytest -q \
  tests/test_training_activity.py tests/test_checkpoint_monitor.py \
  tests/test_learning_archive_status.py tests/test_live_room_diagnostic.py \
  tests/test_simulator_semantic_cache.py tests/test_simulator_server.py \
  tests/test_neural_recovery.py tests/test_room_learning.py tests/test_room_transfer.py
```

Проверки кода не считаются физическим успехом. При работающем физическом диагностике тяжёлые benchmarks запускаются только с запасом памяти, чтобы не вытеснять его. Новое/изменяемое поведение не добавляет оператор `continue`.

После тестов и Git commit `bin/install-neural-launcher.py` обновляет только launcher, не обучение. `bin/queue-room-cleanup.py --upgrade-waiting` допускает только нетронутый ожидающий полный цикл с тем же ID, резервной копией и без дочерних работников. Активный frozen source и checkpoint не редактируются.

Проверенные изменения отправляются в `origin codex/explorer-onboard-20261010` и `learning HEAD:main`.

## Перенос

Только после принятого полного сценария программа создаёт `explorer_room_cleanup_v1`, перепроверяет содержимое/хеши архива и сохраняет архив всего цикла. Импорт Jetson неактивен; флаг physical_ready остаётся false.

До движения нужны read-only инвентаризация и тест загрузки/инференса на целевом Jetson, совместимость всех входов, p50/p95 задержки, свежесть кадров, STOP/watchdog/ручной перехват и отдельное разрешение на подготовленный физический тест. Сейчас SSH имена explorer.local/homebrain.local недоступны. На этом этапе не отправлялись команды приводам и не менялась прошивка.
