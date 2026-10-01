# Единый автономный контур Explorer

## Запуск

Explorer хранит цель, план, эпизоды, вопросы и исходы на Jetson. Закрытие браузера не отменяет задачу. Перезапуск процесса не повторяет незавершённое движение: задание возвращается к наблюдению и строит новый план. Ручной перехват или STOP отменяет текущую доставку.

Во вкладке **Автономность** задайте конечный результат, число попыток и время, один раз разрешите нужную деятельность и запустите задачу. Контакт с предметом и подготовка сцены разрешаются отдельно только для лёгкого выбранного предмета.

CLI использует тот же backend:

```bash
explorer autonomy status
explorer autonomy grant
explorer autonomy start --object носок --destination "корзина для белья" "Найди и положи предмет"
explorer autonomy cancel JOB_ID
```

В Telegram используется тот же явный контракт: `/do носок -> корзина для белья`.

`grant` разрешает базу и руку на час. Разрешение, аппаратная готовность и качество навыка проверяются независимо.

## Исполнение

Сохраняемый цикл: `OBSERVE → PLAN → VALIDATE → EXECUTE → VERIFY → RECORD → RESET → NEXT`.

Планировщик использует условия `true/false/unknown` и поиск по стоимости. При `unknown` он вставляет наблюдение и перепланирование. Шаги связаны с версией сцены. Физический исполнитель остаётся один: существующие `DeliveryTask`, `DeliveryRobot`, `Missions` и `TrajectoryExecution`. Автономный слой не публикует моторные команды.

Размещение в командном режиме проверяет отделение предмета от отводимой руки, устойчивость на опоре, целевую область и непрерывность трека. Команда открытия записывается как `command_estimate`, но не выдаётся за измерение сервопривода.

Recovery выбирается по подтверждённой причине: новый ракурс при потере трека, релокализация при потере pose, другой подход при недостижимости, regrasp при пустом захвате или проскальзывании. `unknown` сначала вызывает наблюдение. Время и повторы ограничены.

## Обучение

- `CandidateScorer` — обновляемая logistic-модель выбора кандидата. Checkpoint продвигается только после сравнения с неизменным геометрическим baseline.
- `OutcomeModel` учится по наблюдаемому RGB-D, lidar/odometry или проверенному удержанию. Команда, сравненная с командой, не считается физическим наблюдением.
- `InterventionActorLearner` — ограниченный actor/critic для коротких коррекций. Replay различает предложенное и фактически исполненное действие, intervention, reward, `terminated` и `truncated`. Это инженерный baseline, не полное воспроизведение HIL-SERL.
- LeRobot ACT сохраняет существующий адаптер 6/9 DoF, экспорт, episode-level split, обучение и offline validation.
- SmolVLA имеет адаптер Explorer для изображения, инструкции, командного состояния и 9 DoF действия. Физический backend заблокирован до собственного checkpoint и проверки памяти/задержки.

Событийная сегментация выделяет остановку базы, подведение, закрытие, подтверждённый подъём, перенос и размещение. Кандидат навыка входит в реестр после нескольких эпизодов и подтверждения спорной границы. Train/validation разделяются по исходному episode ID; validation не используется для обучения.

Все видео, команды, наблюдения и manifest находятся в `/home/vlad/Explorer/data/episodes`. Автоматического удаления и облачной выгрузки нет.

## Фактические границы

На текущем robotio суставное положение расчётное. Guarded closure использует зрение и малые команды только после принятия параметров; силу в ньютонах она не измеряет. Подготовка сцены допускает только лёгкие разрешённые предметы и проверяет побочные перемещения. Пока физический reset-профиль не принят, серия остановится после первой попытки с конкретным запросом, а не изобразит self-reset.

| Требование | Код | Проверка | Статус |
|---|---|---|---|
| Command-mode place | `grasp_verification.py`, `delivery_vision.py` | command-estimate success и негативы track/zone/support/follow | программно проверено; нужен физический эпизод |
| Контракты | `autonomy_contracts.py` | validation tests | программно проверено |
| Поиск плана | `skill_planner.py` | state/permissions меняют план | программно проверено |
| Постоянная цель/серия | `autonomy_runtime.py`, `episode_store.py` | restart, idempotency, episode tests | реализовано; физический запуск зависит от readiness |
| Recovery | `skill_planner.py`, `DeliveryTask.regrasp` | route tests | программно проверено |
| Подготовка сцены | `scene_preparation.py` | direction/collateral negatives | программно; физический contact profile не принят |
| Guarded closure | `semantic_world.py`, `delivery_robot.py` | stale/slip/deformation branches | подключено за acceptance flag |
| Scorer/outcome/RL | `learning_stack.py` | реальные parameter updates | программно; нужны физические labels |
| ACT | `learning-run.py`, `policy_execution.py` | held-out episodes | путь установлен; физическая policy не принята |
| SmolVLA | `model_backends.py` | schema/chunk invalidation | адаптер готов; веса/8 GB acceptance отсутствуют |
| Skill discovery/replay | `skill_discovery.py` | segmentation/split tests | программно; нужны эпизоды |
| Память | `semantic_world.py` | lifecycle/query | работает; 3D ограничена принятой camera→map геометрией |
| Ресурсы | `resource_scheduler.py` | restart/priority | программно проверено |
| Автономный день | `autonomous_day.py` | priority tests | селектор готов; физическая сессия не проведена |
| Goal-conditioned plan | `skill_planner.py`, `autonomy_contracts.py` | разные predicates/permissions/scene version дают другой план | программно проверено; физический исполнитель остаётся принятым `DeliveryTask` |
| Recovery по причине | `skill_planner.py`, `delivery_task.py` | track/localization/reach/slip выбирают разные ветви | программно проверено; без бесконечных повторов |
| Active perception | `change_view`, `reobserve`, `mobile_alignment.py` | stale track и новая версия сцены требуют нового наблюдения | существующий физический путь переиспользован; learned reactive policy не заявляется |
| Anti-forgetting | `skill_discovery.py::replay_anchors`, `PolicyRegistry` | episode-level split, task-balanced replay, rollback | механизм готов; сравнение A/B ждёт собственные эпизоды |
| Outcome adaptation | `learning_stack.py::OutcomeModel` | принимает только независимые RGB-D/lidar/hold transitions | параметры действительно обучаются; на роботе ещё нет достаточного набора переходов |
| Точечная помощь | `autonomy_runtime.py::HelpRequest` | отсутствующие object/destination дают один локальный вопрос | работает; аппаратная зависимость показывается статусом, а не просьбой о новой калибровке |
| Ручной арбитраж | существующие `manual_teleop.py`, `gamepad_panel.py`; `stop_all()` | multikey, neutral, stale input, focus loss, ownership и STOP входят в общую регрессию | сохранено; автономная задача отменяется при ручном перехвате/STOP |
| Автопродвижение/откат | `CandidateScorer`, `PolicyRegistry` | held-out episode split, сравнение с geometry baseline, rollback checkpoint | программно проверено; без физических данных активен baseline |
| Данные на Jetson | `episode_store.py` | transactional manifest, restart→interrupted, quota status | `/home/vlad/Explorer/data/episodes`; удаления и облачной выгрузки нет |

## Источники механик

- [SayCan](https://arxiv.org/abs/2204.01691): язык предлагает структуру, исполнимость определяют навыки и состояние.
- [DynaMem](https://arxiv.org/abs/2411.04999): динамическая память появившихся, переместившихся и пропавших объектов.
- [Visual Pushing and Grasping](https://arxiv.org/abs/1803.09956): подготовка получает кредит через результат захвата.
- [LOTUS](https://arxiv.org/abs/2311.02058): библиотека повторяющихся навыков; здесь первый вариант использует событийную сегментацию.
- [HIL-SERL](https://github.com/rail-berkeley/hil-serl): actor/learner и вмешательства; Explorer пока использует ограниченный совместимый baseline.
- [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla): image/state/instruction → action chunks и fine-tuning на своих данных.

## Структура и восстановление

- Автономность: `src/autonomy_runtime.py`, `src/skill_planner.py`, `src/autonomy_contracts.py`.
- Обучение: `src/learning_stack.py`, `src/model_backends.py`, `src/skill_discovery.py`, `src/lerobot_bridge.py`.
- Мир: `src/semantic_world.py`, `src/scene_preparation.py`, `src/grasp_verification.py`.
- Панель/API: `src/web.py`, `src/index.html`, `bin/autonomy-command.py`.
- До реформы: tag `pre-unified-reform-20261001`.
- Фактически развёрнутый Jetson до реформы: branch/tag `jetson-pre-unified-reform-20261001`.
