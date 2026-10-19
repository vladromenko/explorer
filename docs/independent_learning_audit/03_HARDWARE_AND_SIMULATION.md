# 03. Аппаратура, симуляция и достоверность параметров

## Что известно о действующей конфигурации

**VERIFIED по уточнению пользователя и контракту factory-кода: реальной обратной связи по суставам нет.** Положение руки оценивается по отправленным целевым углам. Сентябрьские ответы экспериментального контроллера не доказывают наличия измерений в нынешней системе. Моё промежуточное обратное утверждение при аудите было ошибочным и отозвано; разбор — D02 и D26 в19.

Текущий сетевой опрос Jetson недоступен: существующие aliases не разрешались через DNS. Поэтому исторические частоты ниже не являются сегодняшним измерением. Никакие приводы или прошивки во время аудита не включались.

| Узел | Назначение, связь и интерфейс | Частота / задержка | Обратная связь, команды, ограничения | Применение |
| --- | --- | --- | --- | --- |
| ROSMASTER M3 Pro, mecanum | Штатный MCU → USB → Jetson; micro-ROS velocity/odom | Исторически08.10:61 odom сообщений/8с, максимальный интервал114мс; текущая задержка UNKNOWN | Команды vx/vy/wz. Историческая колёсная одометрия не равна точной локализации; учитывать проскальзывание и footprint | В MuJoCo контактная модель колёс; в полном сценарии база управляется внешним навигатором, выход ACT для базы обнуляется |
| Манипулятор и захват | В управляющем контракте6 servo channels:5 осей и захват; описание владельца — шестикоординатная рука | Исторический поток команд40мс; физическое время отклика UNKNOWN | **Command estimate, не измеренные углы.** Нет подтверждённых измерений усилия/раскрытия | MuJoCo torque-limited PD;6 углов входят в9D action вместе со скоростями базы |
| MCU | Позднее задание перечисляет STM32F407; исходный заказ и исторический CubeProgrammer — H7/0x450 | Нынешняя идентичность UNKNOWN | Разные исторические прошивки не считать одним действующим интерфейсом | MCU побитно не эмулируется; для выяснения нужен отдельный read-only inventory |
| Два Tmini-plus | UART/MCU → ROS scan0/scan1 |08.10: каждый39 сообщений/8с; максимальные интервалы155/152мс | Измерения дальности; слепые зоны и калибровка остаются существенными | Симуляционные лучи/сканы; шум и задержки экземпляра не идентифицированы |
| DaBai RGB-D | USB → Jetson; cameraInfo и зарегистрированная глубина |11.10 после исправления около15fps; отдельно зафиксировано около108мс отставания SDK depth | Свежие RGB-D и экструзика нужны для геометрии; аппаратная синхронизация SDK не была доступна | Actor получает wrist RGB320×240; reference camera640×480; реальная дисторсия/шум воспроизведены не полностью |
| Дополнительная USB-камера | Исторически Logitech / внешний наблюдаемый вид; точная нынешняя модель UNKNOWN | Текущие fps и задержка UNKNOWN | Внешний вид инженера нельзя автоматически считать входом actor | Возможна диагностика; перенос на policy требует собственного контракта |
| Jetson Orin Nano8GB | Основной бортовой Linux/ROS2/CUDA компьютер по истории развёртывания |14.10 старый ACT: один forward0.967с; это не полноценный latency benchmark текущей версии | Нынешний интерфейс недоступен по DNS; память/питание ограничены | Inference после отдельной проверки; готовность нынешнего пакета не установлена |
| MacBook Pro M4 Pro24GB | Обучение, симуляция и внешний диск | Реальные MPS updates измерены; детали05/07 и исторические JSON | Общая память CPU/GPU, CPU physics и I/O; пользовательская пауза сохранена | Реальное ACT-обучение; MuJoCo physics не превращается автоматически в GPU simulation |
| Дополнительный Orin NX16GB | Указан владельцем; исторический alias HomeBrain | Текущая доступность/нагрузка UNKNOWN | Не считать автоматически подключённым к ROSMASTER | Возможное дополнительное вычисление или inference требует измерения |
| Raspberry Pi5, Hailo-10H | Дополнительные устройства владельца | UNKNOWN | Нет проверенного текущего интерфейса или поддержанного ACT compiler | Возможная обработка восприятия — HYPOTHESIS; это не универсальный PyTorch accelerator |

## Виртуальная модель против реального экземпляра

VERIFIED относится к наличию параметров в коде, а не к их измеренной точности на роботе. Неизвестные значения не заменяются благоприятными предположениями.

| Параметр | Симуляция | Реальный экземпляр / источник | Достоверность и проверка переноса |
| --- | --- | --- | --- |
| CAD/FK |20STL и URDF/SRDF,22SHA; сторонняя ревизия bdac682 | Есть отдельные статические позы и hand-eye, но не полный обмер | Байты VERIFIED; реальная геометрия требует измерения осей, offsets и крепления камеры |
| Суставные единицы | Градусы servo endpoint; FK по mapping команд | Командные углы, не энкодеры | Проверить люфт и tracking визуальными измерениями; не выдавать команды за измерения |
| Приводы | kp25,kv0.65; torque limits[1,1.2,1.2,.9,.7,.35]Nm | UNKNOWN | Оценки. Системная идентификация с движениями потребует отдельного разрешения |
| Контакты | Трение пола.8; пример ткани1.1; pads радиус4мм/длина28мм | UNKNOWN | Измерение материалов и sensitivity sweep |
| Ткань | flex shell7×8; масса.022кг; Young10000Pa; толщина.0007м | Конкретный носок динамически не идентифицирован | Взвешивание, геометрия, статическая податливость, затем тесты чувствительности |
| Захват | Команды30→170°, linkage0→-1.1rad | Историческое удержание около160°; сила/раскрытие UNKNOWN | Отдельная проверка mapping и податливости, не слепой перенос170° |
| Камера | fx480.7079,fy480.7113,cx322.29,cy245.64 для640×480; near.015/far8 | Исторический cameraInfo; нынешние параметры UNKNOWN | Сверить intrinsics, extrinsics и timestamps на реальных кадрах |
| Оптика и задержки | Идеальный pinhole; коэффициенты дисторсии сохранены как provenance | Исторически дисторсия и SDK lag существовали | Render/delay randomization следует выбирать по измерению |
| Время | physics timestep.002с; action period.05с; settling учитывается отдельно | Задержка MCU/USB/servo UNKNOWN | Сопоставить фактические команды, движение и end-to-end timestamps |
| Локализация | Наблюдаемая карта/поза; точность реального odom не идентифицирована | Повторная локализация не принята | Не подменять реальный датчик симуляционной истинной позой |

Точные параметры и функции: `source/config/arm_model.json`, `source/simulator/model_config.json`, `source/simulator/explorer_model.py`, `source/simulator/environment.py`; строки — `evidence/SOURCE_FUNCTION_INDEX.md`; происхождение — `evidence/historical/geometry-provenance.json`. Часть аппаратных сведений восстанавливается только из исторических документов и переписки, что явно ограничивает их силу.

## 11. Откуда виртуальная модель ROSMASTER и насколько ей можно доверять

Базовая CAD/URDF-геометрия происходит из стороннего [yahboom_m3pro_description][cad-external], закреплённого на `bdac682e57a8eeaf7b18eece44bfbfc54de98e8a`. Проверены SHA 20 STL и URDF/SRDF — всего 22 геометрических файла против `arm_model.json`. Это не автоматически точная заводская цифровая копия данного экземпляра.

MuJoCo 3.14.0 рассчитывает контакты, приводы, mecanum и деформируемый материал flex. Носок не должен приклеиваться скрытым weld/attach; база не телепортируется принудительной установкой позы. Геометрический teacher и независимый оценщик могут знать симуляционную истину, actor — нет. [Физическая среда][environment], [модель][explorer-model].

Приближены или требуют измерения на экземпляре: массы/инерции, трение, жёсткость контактов и ткани, torque/PD, задержки/люфты приводов, реальная форма мягкого носка, оптика/шум/несинхронность камеры. Исходные оценки PD включают `kp=25`, `kv=0.65`; наличие чисел не превращает их в идентифицированные реальные параметры.

Mecanum имеет четыре колеса, ролики 45°, модель диаметра 80 мм. Компонентный тест подтвердил правильные знаки шести направлений, но не достаточную точность полной навигации.

Историческая стационарная hand-eye оценка имела максимум 5.62 мм / 1.68° на описанных кадрах. Она не доказывает текущую динамическую калибровку. В истории физический захват носка получался около команды 160°, а симуляционный reference использовал около 170°; перенос mapping захвата должен быть отдельно принят.

Изменение цвета/позиции/размера, материала и yaw в заданном диапазоне — domain randomization. Это не доказательство покрытия любого складчатого носка, любого освещения, квартиры, ковра или корзины. Требуется sim-to-real проверка и идентификация расхождений, а не только больше synthetic кадров.


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
