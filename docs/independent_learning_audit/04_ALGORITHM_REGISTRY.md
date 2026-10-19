# 04. Реестр алгоритмов и статусы применения

## 6. Реестр применённых алгоритмов: что именно обучалось

| Механизм | Как используется и зачем | Результат / граница |
| --- | --- | --- |
| Контекстный bandit + logistic scorer | 18 признаков контекста/кандидата, sigmoid, градиентное обучение с регуляризацией; epsilon exploration; ранжирование допустимых IK-вариантов | Ранний локальный бортовой механизм. Не visual end-to-end управление и не доказанный полный навык |
| OutcomeModel / ridge | Линейная регуляризованная модель независимо наблюдаемых физических приращений | Реализована, но достаточная физическая валидация полной миссии отсутствует |
| Ограниченный actor/critic baseline | Обновляет critic по reward/следующему состоянию и actor с ограниченным выходом | Исходник и тесты есть; не полноценный HIL-SERL/SAC и не текущий ACT learner |
| Символьный поиск плана | Предусловия/эффекты навыков, стоимость, доступность/разрешения, восстановление по типу причины | Координация написанных навыков; не обучение новой моторики из языка |
| Сегментация и replay anchors | Событийное выделение навыков и сохранение старых данных | Механизм организации опыта, не доказательство отсутствия забывания |
| Ruckig + IK/MoveIt | Плавные команды реальной руки, достижимость/траектории | Исполнение/планирование, не нейросетевое обучение |
| MuJoCo teacher | Геометрический генератор команд в физической модели, случайные варианты сцены | Производит данные. Его доступ к симуляционной геометрии допустим только для генерации/оценки |
| Behaviour cloning / ACT | Предсказывает chunk команд по текущему RGB, состоянию и clock; градиентная минимизация ошибки экспертных действий | Реально обучаемая моторная политика; текущий основной neural backend |
| Выбор checkpoint по физическим rollout | Проверка нескольких сохранённых весов и reference на тех же новых сценах | Защищает от выбора только по offline loss; сам является подбором, не финальной независимой оценкой |
| Фазовые ridge-регрессоры | По state и грубо уменьшенному RGB обучались отдельные линейные residual-модели по фазам | Экспериментальный baseline `phase_learning.py`; не включён в управление и не переносимая политика |
| Cross-entropy search v2/v3 | Популяция ограниченных параметров, выбор лучших по парным физическим попыткам, обновление распределения | Подбор параметров геометрического исполнителя. **Не SGD-обучение ACT** |
| Bounded program repair v4 | Классификация причины после остановки, комбинации разрешённых операторов, парный контроль и подтверждение | Не произвольное самопрограммирование; неизвестные дефекты требуют новой реализации |
| Dataset aggregation с коррекциями | Ученик посещает состояние, эксперт исправляет; student-префикс маскируется из целей; новые исправления добавляются к истории | DAgger-подобная ограниченная схема, не полная реализация всех вариантов DAgger/HIL-SERL |
| GroundingDINO + RGB-D | Семантическая рамка по тексту, глубина/геометрия, повторные свежие наблюдения | Поиск предмета; детектор не обучался заново каждой попыткой носка |
| Frontier / информация-стоимость / A* | Исследование наблюдаемого свободного пространства, путь с footprint и живыми проверками | Текущая навигация написана алгоритмически; ACT не обучена самостоятельной whole-room навигации |
| IK / nonlinear least squares / CAD-sweep | Конечные решения достижимости по своей кинематике и проверка полного пути; solver не является нейросетью | Позволяют планировать безопасные подходы, но не подтверждают фактический захват |
| Геометрический multi-view / hold / place verifier | Достижимый обзор бортика, независимые ракурсы, наблюдаемое следование ткани/нахождение внутри | Наблюдаемые gates, не доступ actor к истинным контактам |
| Независимый физический оценщик | После действий читает настоящий контакт/подъём/положение/стабильность симуляции | Метрики и отбор данных; не вход управления |

Исходники: [ранние модели][stack], [фазовые residuals][phase], [поиск параметров][search], [ограниченные ремонты][repair], [ACT/export][learning], [нейросетевое восстановление][neural], [комнатный цикл][room-learning].

### 6.1. Что именно делает cross-entropy search

Для каждого раунда создаются четыре кандидата: incumbent и три ограниченных вариации. Одна вариация использует более широкое исследование диапазона, чтобы распределение не схлопывалось в одинаковые попытки. Кандидаты проверяются попарно на двух одинаковых сценах. Сравнение учитывает слабую сцену, этап и сохранение уже проверенного pick; при застое меняется подпространство параметров.

При program repair численные параметры фиксированы, меняется набор операторов. Это даёт возможность приписать разницу конкретному изменению программы, а не одновременно новой сцене, новым числам и новой структуре. Дополнительное подтверждение проводится на новых учебных сценах. Никакой гарантии нахождения решения за 1000 проб этот метод не давал.

### 6.2. Что делает настоящая ACT сейчас

Одна партия содержит 32 наблюдения и соответствующие будущие целевые команды. Изображение обрабатывается visual backbone, state/clock объединяются с признаками; Transformer выдаёт последовательность будущих действий. Ошибка L1 по нормализованным экспертным действиям проходит назад через сеть; optimizer изменяет веса. Так многократно проходятся выбранные обучающие кадры. Контрольные сохранения пишутся каждые 1000 обновлений.

Это imitation learning, а не RL, в котором студент сам получает reward и непосредственно оптимизирует его градиентом. Неудачи влияют через диагноз, выбор новых сцен, экспертные коррекции и следующий датасет. Ошибочная траектория не должна становиться эталоном лишь потому, что была записана.


[act-external]: https://github.com/tonyzhaozh/act
[archive]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_archive.py
[audit-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING_AUDIT.md
[audit08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/CODE-AUDIT-20261008.ru.md
[bridge]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/lerobot_bridge.py
[c-0293b06]: https://github.com/vladromenko/explorer/commit/0293b068acf58542b8ec61605c75ab7feed8e802
[c-09cb728]: https://github.com/vladromenko/explorer/commit/09cb7282c2a9a52977329c6cff78470e9ebe1a3d
[c-0b231e4]: https://github.com/vladromenko/explorer/commit/0b231e48e2e7af1256d2a60a6cc67e6f1287556f
[c-0bd7728]: https://github.com/vladromenko/explorer/commit/0bd772860832ef54d8a65aed0d8abaa8f1bce0d7
[c-146a463]: https://github.com/vladromenko/explorer/commit/146a4630ef407efef09b5f6ec3944909b206634e
[c-1d6ddf5]: https://github.com/vladromenko/explorer/commit/1d6ddf538369e6c54b1fedb22f6e57ef2ec119d2
[c-1da8ecb]: https://github.com/vladromenko/explorer/commit/1da8ecb26efaa798d4293f126d7d123aff7985e5
[c-2161826]: https://github.com/vladromenko/explorer/commit/2161826271fb90efd9bd6e9ae7cc818590d09e51
[c-2d3aeae]: https://github.com/vladromenko/explorer/commit/2d3aeae87d190f83449981fbdad5cf5e55673f36
[c-2e484e3]: https://github.com/vladromenko/explorer/commit/2e484e39f98078d99b5e579c043bc1f6e5072ee6
[c-30271ba]: https://github.com/vladromenko/explorer/commit/30271bae7e0c53a3a06462fa55a7da962dace506
[c-3568b30]: https://github.com/vladromenko/explorer/commit/3568b300116926717f03b8b13cf01c3984702093
[c-37ae853]: https://github.com/vladromenko/explorer/commit/37ae853480c6dc88c7dbbebdfd348e38fa0876f2
[c-3fb6905]: https://github.com/vladromenko/explorer/commit/3fb69053bafc5185b70dcbef75988bf79d043bb2
[c-4146b1c]: https://github.com/vladromenko/explorer/commit/4146b1c6ba509b31431fcbde5daa291d184b869e
[c-42374fe]: https://github.com/vladromenko/explorer/commit/42374fe013127d9a3d82eb0a81011f51c29e05c8
[c-43304eb]: https://github.com/vladromenko/explorer/commit/43304eb10133002fe2fe11395d9f2d67317d7c16
[c-4386cc0]: https://github.com/vladromenko/explorer/commit/4386cc0cdf162de3dbaab6d3568335301146003d
[c-5b27e21]: https://github.com/vladromenko/explorer/commit/5b27e215b94ff0d483806391559f4dc83a8b7576
[c-5bb6914]: https://github.com/vladromenko/explorer/commit/5bb69148a733272d53f3897acbcab65a6a335f4b
[c-5db446d]: https://github.com/vladromenko/explorer/commit/5db446d32752698256ddb7e4a5c36cc258967605
[c-65c5738]: https://github.com/vladromenko/explorer/commit/65c57387e988998cc07349d5fc89e624dcb9d122
[c-67908f9]: https://github.com/vladromenko/explorer/commit/67908f956eea5b5f72db5c906bf989d8810c2504
[c-6d6394a]: https://github.com/vladromenko/explorer/commit/6d6394a4ba3a2e92381b2f45fbd60f4867e572b5
[c-726f8cb]: https://github.com/vladromenko/explorer/commit/726f8cbd72f7d4723d8a0e8500099efbae3dfb67
[c-853c4c0]: https://github.com/vladromenko/explorer/commit/853c4c0d9c418fac3a4b9f1318c9b036e3c11b33
[c-8c6ce11]: https://github.com/vladromenko/explorer/commit/8c6ce11c8f27c91e60aaa8c9623a55040a019706
[c-929497a]: https://github.com/vladromenko/explorer/commit/929497a7f85366983d2bfbdf4dea9910e727ced2
[c-934c479]: https://github.com/vladromenko/explorer/commit/934c47986e11621fe0e0a074e51866e2bc81b92b
[c-a5d9ea7]: https://github.com/vladromenko/explorer/commit/a5d9ea7c3f40324fe4c37a54bd3425b46bf34f65
[c-b4186e1]: https://github.com/vladromenko/explorer/commit/b4186e1749ad2ac12d845fc51e63adf74828fad8
[c-b6373e3]: https://github.com/vladromenko/explorer/commit/b6373e35bd827ba42f50d5d1b2e6954e575ed61e
[c-b828129]: https://github.com/vladromenko/explorer/commit/b8281297e4eb3f846bba22193710b04cacc6775a
[c-b9cd5e3]: https://github.com/vladromenko/explorer/commit/b9cd5e31f29d86935c60e24ff577878eaa785801
[c-bf36e4e]: https://github.com/vladromenko/explorer/commit/bf36e4ea370d44eb1757c093247bb1e10e167ff6
[c-c553017]: https://github.com/vladromenko/explorer/commit/c55301788dca39584bcf7d68f761580de2a6c7ac
[c-c746696]: https://github.com/vladromenko/explorer/commit/c7466967cbe00f26ff7ee867cd0ff8542d8b0050
[c-cfbd8eb]: https://github.com/vladromenko/explorer/commit/cfbd8eb0d2f6f0b4defa055aac9ea972380f6c84
[c-cfe4aa7]: https://github.com/vladromenko/explorer/commit/cfe4aa7044e75798384d8fd6269fcab84761820d
[c-d1c4d11]: https://github.com/vladromenko/explorer/commit/d1c4d1155fad093f5ca04c92343dd1c0d44a6fb5
[c-e9ad08b]: https://github.com/vladromenko/explorer/commit/e9ad08be2d3694369610d88676024015b21ec6d8
[c-f2ac13d]: https://github.com/vladromenko/explorer/commit/f2ac13dff1b378fc05600bd1443fb537441d289e
[c-fd62906]: https://github.com/vladromenko/explorer/commit/fd629068482570e18bc5d2d087133dc647f8a957
[cad-external]: https://github.com/dimwael/yahboom_m3pro_description/tree/bdac682e57a8eeaf7b18eece44bfbfc54de98e8a
[contract]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_policy_contract.py
[coordinator]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/room_cleanup.py
[dagger-external]: https://arxiv.org/abs/1011.0686
[diagnostic]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/live_room_diagnostic.py
[diffusion-external]: https://github.com/real-stanford/diffusion_policy
[early-learning]: https://github.com/vladromenko/explorer/blob/2d3aeae87d190f83449981fbdad5cf5e55673f36/docs/LEARNING-IMPLEMENTATION.ru.md
[entry]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/train_entry.py
[environment]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/environment.py
[eval-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EVALUATION_REPORT.md
[explorer-model]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/explorer_model.py
[guide]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/TRAINING-GUIDE.ru.md
[implementation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/LEARNING-IMPLEMENTATION.ru.md
[intervention]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_intervention.py
[learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/learning.py
[lerobot-external]: https://huggingface.co/docs/lerobot/en/act
[lineage]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/episode_lineage.py
[local-grasp]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/local_grasp_experiment.py
[mjx-external]: https://mujoco.readthedocs.io/en/stable/mjx.html
[model-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/MODEL_AND_TRAINING.md
[neural]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/neural_recovery.py
[observer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/checkpoint_monitor.py
[onboard]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/EXPLORER-ONBOARD-RELEASE-20261010.ru.md
[origin]: https://github.com/vladromenko/explorer
[phase]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/phase_learning.py
[r-adaptive17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/adaptive-learning-20261017.json
[r-audit12]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-audit-e7905c76974c-20261012.json
[r-auto16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-20261016.json
[r-causal17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/causal-repair-20261017.json
[r-correction16]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/automatic-learning-correction-20261016.json
[r-corrective15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-corrective-cycle-20261015.json
[r-hierarchy15]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/hierarchical-rework-20261015.json
[r-jetson14]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-training-d6ebe13cae23-jetson-20261014.json
[r-macos]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-macos-20261011.json
[r-monitor17]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/learning-causal-monitor-20261017.json
[r-neural18]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/neural-recovery-20261018.json
[r-overnight]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/reports/simulator-overnight-check-20261012.json
[record]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/mobile_demonstrations.py
[repair]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/failure_repair.py
[repo]: https://github.com/vladromenko/explorer-learning
[room-doc]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/ROOM-CLEANUP.ru.md
[room-learning]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_learning.py
[room-policy]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_policy.py
[room-port]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_cleanup_port.py
[room08]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/audits/20261008-room-resume.md
[runbook]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/RUNBOOK.md
[search]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/skill_training.py
[simopt-external]: https://arxiv.org/abs/1810.05687
[smolvla-external]: https://huggingface.co/docs/lerobot/en/smolvla
[stack]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/src/learning_stack.py
[transfer]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_transfer.py
[unified]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/UNIFIED-AUTONOMY.ru.md
[validation]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/simulator/room_validation.py
[vision]: https://github.com/vladromenko/explorer-learning/blob/1da8ecb26efaa798d4293f126d7d123aff7985e5/docs/OBJECT-VISION.ru.md


## Подробные карточки:17 полей каждого подхода

Машиночитаемый реестр — `evidence/algorithm_registry.json`. Ниже поля1–17 раскрываются для каждого семейства. Файлы закреплены ревизией1da8ecb; точные функции/строки — в `evidence/SOURCE_FUNCTION_INDEX.md`. Обоснование выбора — восстановление по коду и сообщениям19, а не утверждение о скрытом процессе рассуждения. UNKNOWN не заменяется фиктивным числом GPU-часов. Наличие кода не доказывает физическое освоение навыка.

### Contextual bandit / CandidateScorer

1. **Название:** Contextual bandit / CandidateScorer
2. **Идея:** Ранжирование допустимых кандидатов по sigmoid вероятности исхода
3. **Причина выбора:** Дешёвый прозрачный локальный selector поверх безопасной геометрии; не визуальная моторная сеть
4. **Предполагавшаяся цель:** Ранжирование допустимых кандидатов по sigmoid вероятности исхода
5. **Данные:** Контекст/кандидат/фактический исход
6. **Реализация:** src/learning_stack.py::CandidateScorer
7. **Исходники:** src/learning_stack.py::CandidateScorer
8. **Параметры:** 18features; epochs300; lr.04; L2.002; epsilon exploration
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Код и компонентные проверки есть; полная физическая learning curve UNKNOWN
11. **Результат:** Код и компонентные проверки есть; полная физическая learning curve UNKNOWN
12. **Проблемы:** Не обучает непрерывное управление по RGB и не доказал полный сценарий
13. **Почему нет полного решения:** Не обучает непрерывное управление по RGB и не доказал полный сценарий
14. **Что сохранить:** Версионирование и paired promotion
15. **Что заменить/проверить:** Не использовать как доказательство neural motor learning
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### OutcomeModel / ridge

1. **Название:** OutcomeModel / ridge
2. **Идея:** Регрессия наблюдаемого изменения состояния по state/action
3. **Причина выбора:** Интерпретируемая оценка последствий для планирования/отбора
4. **Предполагавшаяся цель:** Регрессия наблюдаемого изменения состояния по state/action
5. **Данные:** Переходы state/action/nextstate, совместимые единицы
6. **Реализация:** src/learning_stack.py::OutcomeModel
7. **Исходники:** src/learning_stack.py::OutcomeModel
8. **Параметры:** 6state+6action по умолчанию; ridge least-squares
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Реализация подтверждена; полный физический прогон модели UNKNOWN
11. **Результат:** Реализация подтверждена; полный физический прогон модели UNKNOWN
12. **Проблемы:** Линейный переход не моделирует надёжно ткань/контакт/скрытое состояние
13. **Почему нет полного решения:** Линейный переход не моделирует надёжно ткань/контакт/скрытое состояние
14. **Что сохранить:** Контракты переходов и диагностические residuals
15. **Что заменить/проверить:** Сравнить с простым baseline до использования в planning
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### InterventionActorLearner / actor-critic

1. **Название:** InterventionActorLearner / actor-critic
2. **Идея:** Ограниченный линейный actor и critic с TD target
3. **Причина выбора:** Прозрачный ранний baseline с interventions, не реализация SAC/HIL-SERL
4. **Предполагавшаяся цель:** Ограниченный линейный actor и critic с TD target
5. **Данные:** Observation/action/reward/nextstate/terminated
6. **Реализация:** src/learning_stack.py::InterventionActorLearner
7. **Исходники:** src/learning_stack.py::InterventionActorLearner
8. **Параметры:** 12obs/3action; epochs80; rate.01; gamma.97
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Код и tests; серийный автономный успех полной уборки UNKNOWN
11. **Результат:** Код и tests; серийный автономный успех полной уборки UNKNOWN
12. **Проблемы:** Ограниченная модель и непроверенный MDP всей задачи
13. **Почему нет полного решения:** Ограниченная модель и непроверенный MDP всей задачи
14. **Что сохранить:** Разделение proposed/executed и данные interventions
15. **Что заменить/проверить:** Не выдавать за современный проверенный visual RL
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Behaviour Cloning / ACT

1. **Название:** Behaviour Cloning / ACT
2. **Идея:** L1 imitation будущих абсолютных action chunks
3. **Причина выбора:** Уже интегрированный LeRobot и компактный MPS/Jetson кандидат; выбора по конкурсу не было
4. **Предполагавшаяся цель:** L1 imitation будущих абсолютных action chunks
5. **Данные:** RGB320×240;9Dcommand state;14Dclock; экспертные targets
6. **Реализация:** simulator/learning.py;simulator/train_entry.py
7. **Исходники:** simulator/learning.py;simulator/train_entry.py
8. **Параметры:** 11.905M;ResNet18;dim128;heads4;enc2/dec1;chunk20;batch32;lr5e-5;VAEoff
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** 119047updates3/10vs3/10;133577updates selected local5/10vs2/10;d4 run29078 paused
11. **Результат:** 119047updates3/10vs3/10;133577updates selected local5/10vs2/10;d4 run29078 paused
12. **Проблемы:** Full-room failure695; state/phase shift; teacher/runtime limits differ
13. **Почему нет полного решения:** Full-room failure695; state/phase shift; teacher/runtime limits differ
14. **Что сохранить:** Checkpoint/resume/processors and validated records
15. **Что заменить/проверить:** Architecture choice only after tiny overfit/recovery/paired BC-DP tests
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### LeRobot

1. **Название:** LeRobot
2. **Идея:** Dataset/processors/policies/trainer framework, не отдельный алгоритм
3. **Причина выбора:** Повторное использование готовых training и checkpoint механизмов
4. **Предполагавшаяся цель:** Dataset/processors/policies/trainer framework, не отдельный алгоритм
5. **Данные:** Структурированные frames, normalizers и точный action schema
6. **Реализация:** src/lerobot_bridge.py;simulator/learning.py
7. **Исходники:** src/lerobot_bridge.py;simulator/learning.py
8. **Параметры:** Installed0.6.1; точные зависимости вevidence/dependencies.json
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Реальные MPS updates и optimizer resume подтверждены
11. **Результат:** Реальные MPS updates и optimizer resume подтверждены
12. **Проблемы:** Framework не обеспечивает корректность собственного teacher/MDP/runtime; horizon reader лишне декодировал RGB
13. **Почему нет полного решения:** Framework не обеспечивает корректность собственного teacher/MDP/runtime; horizon reader лишне декодировал RGB
14. **Что сохранить:** Provenance, saved processors, optimizer/RNG
15. **Что заменить/проверить:** Собственный adapter проверять отдельно от framework smoke
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Phase regression

1. **Название:** Phase regression
2. **Идея:** Отдельные ridge residuals по фазам и уменьшенному RGB
3. **Причина выбора:** Дешёвый эксперимент во время перехода к навыкам
4. **Предполагавшаяся цель:** Отдельные ridge residuals по фазам и уменьшенному RGB
5. **Данные:** Размеченные фазы/state/RGB/target residual
6. **Реализация:** simulator/phase_learning.py::fit_phase_residuals
7. **Исходники:** simulator/phase_learning.py::fit_phase_residuals
8. **Параметры:** regularization1 по умолчанию; фазовая группировка
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Подготовка/fit реализованы; в действующем управлении НЕ используются
11. **Результат:** Подготовка/fit реализованы; в действующем управлении НЕ используются
12. **Проблемы:** Нет физической acceptance; подготовка данных не является освоенным навыком
13. **Почему нет полного решения:** Нет физической acceptance; подготовка данных не является освоенным навыком
14. **Что сохранить:** Схема фазового анализа и диагностика
15. **Что заменить/проверить:** Не включать без отдельной проверки closed-loop
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Cross-entropy search

1. **Название:** Cross-entropy search
2. **Идея:** Поиск численных параметров готового исполнителя
3. **Причина выбора:** Короткая адаптация геометрии вместо новых долгих нейросетевых запусков
4. **Предполагавшаяся цель:** Поиск численных параметров готового исполнителя
5. **Данные:** Парные физические результаты одинаковых сцен
6. **Реализация:** simulator/skill_training.py::propose/update_search
7. **Исходники:** simulator/skill_training.py::propose/update_search
8. **Параметры:** 4кандидата;2pairedсцены; elite update; ограниченные диапазоны
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Историческая39-attempt серия:0 полных успехов; ограничения поиска сохранялись
11. **Результат:** Историческая39-attempt серия:0 полных успехов; ограничения поиска сохранялись
12. **Проблемы:** Небольшое покрытие сцен и operator design; ACT weights не обновлялись
13. **Почему нет полного решения:** Небольшое покрытие сцен и operator design; ACT weights не обновлялись
14. **Что сохранить:** Парное сравнение кандидатов и защиту старых успехов
15. **Что заменить/проверить:** Не использовать вместо требуемого dataset-to-policy pipeline
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Bounded program repair

1. **Название:** Bounded program repair
2. **Идея:** Диагноз выбирает набор разрешённых операторов исполнения
3. **Причина выбора:** Автоматизировать известные виды восстановления с проверкой регрессий
4. **Предполагавшаяся цель:** Диагноз выбирает набор разрешённых операторов исполнения
5. **Данные:** Завершённая попытка, наблюдаемые и отдельные физические outcomes
6. **Реализация:** simulator/executor_recovery.py::propose/comparison
7. **Исходники:** simulator/executor_recovery.py::propose/comparison
8. **Параметры:** Конечный operator bank; численные параметры фиксируются при сравнении
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Реализация подтверждена; ранние3 repair checks/0 accepted improvements вUI
11. **Результат:** Реализация подтверждена; ранние3 repair checks/0 accepted improvements вUI
12. **Проблемы:** Не умеет произвольно исправлять новый код или фундаментально менять модель
13. **Почему нет полного решения:** Не умеет произвольно исправлять новый код или фундаментально менять модель
14. **Что сохранить:** Typed reasons, paired compare, rollback
15. **Что заменить/проверить:** Снять обещание универсального самоисправления; добавить явные UNKNOWN причины
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Dataset aggregation / DAgger-like corrections

1. **Название:** Dataset aggregation / DAgger-like corrections
2. **Идея:** Эксперт продолжает из посещённого студентом состояния; префикс не expert target
3. **Причина выбора:** Устранить covariate shift чистых успешных показов
4. **Предполагавшаяся цель:** Эксперт продолжает из посещённого студентом состояния; префикс не expert target
5. **Данные:** Student rollout prefix, допустимое экспертное исправление, success evaluation
6. **Реализация:** simulator/neural_recovery.py;simulator/room_intervention.py
7. **Исходники:** simulator/neural_recovery.py;simulator/room_intervention.py
8. **Параметры:** Локальная correction weight4; общий минимум8passes; полный collector имеет собственные квоты
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** 19/24 локальных исправлений успешны; последующий selection5/10vs2/10
11. **Результат:** 19/24 локальных исправлений успешны; последующий selection5/10vs2/10
12. **Проблемы:** Локальная серия не доказывает coverage full-room handoff; full corrections квота не принята
13. **Почему нет полного решения:** Локальная серия не доказывает coverage full-room handoff; full corrections квота не принята
14. **Что сохранить:** Маски targets и запись provenance
15. **Что заменить/проверить:** Проверить действительные expert targets из трудных student states
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Geometric teacher

1. **Название:** Geometric teacher
2. **Идея:** IK/геометрические фазы создают физически исполняемые показы
3. **Причина выбора:** Заменить ручные показы синтетическими с автоматическим outcome
4. **Предполагавшаяся цель:** IK/геометрические фазы создают физически исполняемые показы
5. **Данные:** Сцена, модель, привилегированная геометрия только генератору
6. **Реализация:** simulator/environment.py::make_episode_trial;simulator/room_learning.py::demonstrate
7. **Исходники:** simulator/environment.py::make_episode_trial;simulator/room_learning.py::demonstrate
8. **Параметры:** Локальные фазы;20Hz; диапазоны материалов/поз вdataset_statistics.json
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Исторически426/700local successes; pairedteacher7/10vs5/10 не ACTscore
11. **Результат:** Исторически426/700local successes; pairedteacher7/10vs5/10 не ACTscore
12. **Проблемы:** Узкие позы, временной шаблон, отличия issued limits и наблюдений runtime
13. **Почему нет полного решения:** Узкие позы, временной шаблон, отличия issued limits и наблюдений runtime
14. **Что сохранить:** Генератор, физическую оценку и все исходные данные
15. **Что заменить/проверить:** Teacher/runtime parity до нового большого dataset
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### IK / CAD sweep

1. **Название:** IK / CAD sweep
2. **Идея:** Конечная нелинейная кинематическая оптимизация и проверка пути
3. **Причина выбора:** Проверять достижимость и геометрические ограничения до движения
4. **Предполагавшаяся цель:** Конечная нелинейная кинематическая оптимизация и проверка пути
5. **Данные:** Наблюдаемая геометрия, FK модели, bounds
6. **Реализация:** simulator/skill_port.py;simulator/explorer_model.py
7. **Исходники:** simulator/skill_port.py;simulator/explorer_model.py
8. **Параметры:** Bounds/accuracy/clearance не ослаблены; см. pinned source
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** a5d9ea7 устранил воспроизведённый false AABB overlap; raised obstacle control остаётся запрещён
11. **Результат:** a5d9ea7 устранил воспроизведённый false AABB overlap; raised obstacle control остаётся запрещён
12. **Проблемы:** FK на commanded state и оценочная геометрия не доказывают реальные контакты
13. **Почему нет полного решения:** FK на commanded state и оценочная геометрия не доказывают реальные контакты
14. **Что сохранить:** Геометрические guards и regression fixtures
15. **Что заменить/проверить:** Калибровать модель, не снимать guard из-за failed run
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### MoveIt

1. **Название:** MoveIt
2. **Идея:** Бортовое планирование руки по URDF/SRDF и ограничениям
3. **Причина выбора:** Повторное использование стандартного motion planning
4. **Предполагавшаяся цель:** Бортовое планирование руки по URDF/SRDF и ограничениям
5. **Данные:** Robot model, planning scene, pose goals; текущий state только estimate
6. **Реализация:** src/arm_planner.py;src/arm_planner_worker.py
7. **Исходники:** src/arm_planner.py;src/arm_planner_worker.py
8. **Параметры:** Исторические конфигурацииURDF/SRDF; текущий live setup UNKNOWN
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Сентябрьские ручные MoveIt/cancel/return проверки описаны; автономная уборка не принята
11. **Результат:** Сентябрьские ручные MoveIt/cancel/return проверки описаны; автономная уборка не принята
12. **Проблемы:** Planner не учит motor policy и не добавляет отсутствующий encoder
13. **Почему нет полного решения:** Planner не учит motor policy и не добавляет отсутствующий encoder
14. **Что сохранить:** Сцену ограничений и клиентский интерфейс
15. **Что заменить/проверить:** Live latency/state validity перед real deployment
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Ruckig / trajectory smoothing

1. **Название:** Ruckig / trajectory smoothing
2. **Идея:** Ограничение скорости/ускорения/рывка команд
3. **Причина выбора:** Согласованная плавная рука вместо резких endpoint скачков
4. **Предполагавшаяся цель:** Ограничение скорости/ускорения/рывка команд
5. **Данные:** Командные положения и границы
6. **Реализация:** src/factory_trajectory.py;config/arm-motion.json
7. **Исходники:** src/factory_trajectory.py;config/arm-motion.json
8. **Параметры:** Точные лимиты вconfig; исторический host period40ms
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Наблюдаемые плавные движения/захват были; не neural learning
11. **Результат:** Наблюдаемые плавные движения/захват были; не neural learning
12. **Проблемы:** Плавная команда не гарантирует tracking, удержание или освоение задачи
13. **Почему нет полного решения:** Плавная команда не гарантирует tracking, удержание или освоение задачи
14. **Что сохранить:** Bounded streaming/cancel
15. **Что заменить/проверить:** Проверить исполнение показов тем же runtime
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Symbolic planning

1. **Название:** Symbolic planning
2. **Идея:** Поиск по предусловиям/эффектам доступных навыков
3. **Причина выбора:** Составлять исполнимые последовательности и восстанавливаться по типу отказа
4. **Предполагавшаяся цель:** Поиск по предусловиям/эффектам доступных навыков
5. **Данные:** Каталог навыков, world facts, permissions
6. **Реализация:** src/skill_planner.py;src/room_cleanup.py
7. **Исходники:** src/skill_planner.py;src/room_cleanup.py
8. **Параметры:** Конечный каталог и стоимости; произвольный новый motor primitive не создаётся
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Координация реализована; универсальное обучение из текста NOT_IMPLEMENTED
11. **Результат:** Координация реализована; универсальное обучение из текста NOT_IMPLEMENTED
12. **Проблемы:** План не компенсирует неработающий pick или недостоверное восприятие
13. **Почему нет полного решения:** План не компенсирует неработающий pick или недостоверное восприятие
14. **Что сохранить:** Typed contracts и разрешения
15. **Что заменить/проверить:** Разделить language grounding и физически принятые skills
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Navigation / frontier / A*

1. **Название:** Navigation / frontier / A*
2. **Идея:** Исследование наблюдаемых фронтиров и безопасный путь
3. **Причина выбора:** Находить объекты за пределами локального handoff
4. **Предполагавшаяся цель:** Исследование наблюдаемых фронтиров и безопасный путь
5. **Данные:** Свежие scans, observed map/pose, footprint
6. **Реализация:** simulator/room_cleanup_port.py;src/visual_navigation.py
7. **Исходники:** simulator/room_cleanup_port.py;src/visual_navigation.py
8. **Параметры:** Information/cost ranking; clearance и проверка пути; full source содержит реальные лимиты
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** После934c479/c746696 найдены оба объекта на20-м обзоре scene1703000014;0collisions
11. **Результат:** После934c479/c746696 найдены оба объекта на20-м обзоре scene1703000014;0collisions
12. **Проблемы:** Один повторно используемый seed не независимая room qualification; не learned ACT navigation
13. **Почему нет полного решения:** Один повторно используемый seed не независимая room qualification; не learned ACT navigation
14. **Что сохранить:** Observed map и collision guards
15. **Что заменить/проверить:** Широкий независимый mission protocol и реальные odom errors
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Object detection / RGB-D

1. **Название:** Object detection / RGB-D
2. **Идея:** GroundingDINO text detection + depth geometry
3. **Причина выбора:** Семантический поиск носка/корзины без скрытых labels
4. **Предполагавшаяся цель:** GroundingDINO text detection + depth geometry
5. **Данные:** Свежие RGB-D/cameraInfo и текстовый query
6. **Реализация:** simulator/semantic_vision.py;src/room_cleanup.py
7. **Исходники:** simulator/semantic_vision.py;src/room_cleanup.py
8. **Параметры:** Pretrained detector; модели и thresholds изconfig/grounding-model.json
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** В диагностике оба объекта подтверждены; отдельные false/unreachable surveys чинились
11. **Результат:** В диагностике оба объекта подтверждены; отдельные false/unreachable surveys чинились
12. **Проблемы:** Детектор не дообучается автоматически каждой попыткой; current real sensor domain UNKNOWN
13. **Почему нет полного решения:** Детектор не дообучается автоматически каждой попыткой; current real sensor domain UNKNOWN
14. **Что сохранить:** Private diagnostic service и freshness gates
15. **Что заменить/проверить:** Проверить real images без приводов и input ablations
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Observed hold/place multi-view verification

1. **Название:** Observed hold/place multi-view verification
2. **Идея:** Независимые ракурсы и наблюдаемая геометрия подтверждают удержание/размещение
3. **Причина выбора:** Избежать actor truth leakage и ложного success
4. **Предполагавшаяся цель:** Независимые ракурсы и наблюдаемая геометрия подтверждают удержание/размещение
5. **Данные:** Два свежих вида, края корзины, движение предмета
6. **Реализация:** simulator/skill_port.py;simulator/surface_destination.py
7. **Исходники:** simulator/skill_port.py;simulator/surface_destination.py
8. **Параметры:** Высота/FoV/freshness/accuracy thresholds сохранены вsource
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Rim aiming/range repairs прошли компоненты; последний full student не дошёл до размещения
11. **Результат:** Rim aiming/range repairs прошли компоненты; последний full student не дошёл до размещения
12. **Проблемы:** Verification может блокироваться обзором; true physical success оценивается отдельно
13. **Почему нет полного решения:** Verification может блокироваться обзором; true physical success оценивается отдельно
14. **Что сохранить:** Мультирегиональные проверки и saved images
15. **Что заменить/проверить:** Оценить false positives/negatives на отдельной размеченной выборке
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Domain randomization

1. **Название:** Domain randomization
2. **Идея:** Варьирование поз, света, цветов, материалов в заданных границах
3. **Причина выбора:** Добавить разнообразие синтетических сцен
4. **Предполагавшаяся цель:** Варьирование поз, света, цветов, материалов в заданных границах
5. **Данные:** Сценовые параметры/seeds и физически валидные модели
6. **Реализация:** simulator/domain_randomization.py;simulator/environment.py
7. **Исходники:** simulator/domain_randomization.py;simulator/environment.py
8. **Параметры:** Полные измеренные диапазоны06; basket heights.08–.30m
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** 735seeds/739records; разные цвета/массы/Young; raw ranges проверены
11. **Результат:** 735seeds/739records; разные цвета/массы/Young; raw ranges проверены
12. **Проблемы:** Разные seeds не доказывают разные наблюдаемые траектории; real parameters не идентифицированы
13. **Почему нет полного решения:** Разные seeds не доказывают разные наблюдаемые траектории; real parameters не идентифицированы
14. **Что сохранить:** Детерминированные seeds, manifest и диапазоны
15. **Что заменить/проверить:** Обосновать диапазоны измерениями и student-state coverage
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment

### Checkpoint selection

1. **Название:** Checkpoint selection
2. **Идея:** Физическое сравнение saved candidates и reference на одинаковых сценах
3. **Причина выбора:** Не полагаться на offline MAE и не терять лучший baseline
4. **Предполагавшаяся цель:** Физическое сравнение saved candidates и reference на одинаковых сценах
5. **Данные:** Замороженные веса, отдельные development seeds, полный runtime
6. **Реализация:** simulator/neural_recovery.py;simulator/room_learning.py
7. **Исходники:** simulator/neural_recovery.py;simulator/room_learning.py
8. **Параметры:** Local10paired; full development10; final protocol2×50 отдельно
9. **Вычисления:** ACT: steps per run in experiments.csv; other families: exact total compute UNKNOWN, not inferred from tests
10. **Эксперименты:** Local candidate5/10vsreference2/10; другая серия3/10vs3/10 не improvement
11. **Результат:** Local candidate5/10vsreference2/10; другая серия3/10vs3/10 не improvement
12. **Проблемы:** Selection сцены нельзя выдавать за независимую final qualification; room приемки нет
13. **Почему нет полного решения:** Selection сцены нельзя выдавать за независимую final qualification; room приемки нет
14. **Что сохранить:** Reference-on-tie и protocol separation
15. **Что заменить/проверить:** Сравнивать по полным миссиям; не переиспользовать final seeds
16. **Доказанное:** Code/parameters and only explicitly reported outcomes; source function lines in SOURCE_FUNCTION_INDEX.md
17. **Гипотезы/пробелы:** Unique causal contribution and superiority over alternatives UNKNOWN unless isolated by paired experiment
