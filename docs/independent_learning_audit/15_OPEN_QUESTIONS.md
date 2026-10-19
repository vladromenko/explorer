# 15. Пробелы, которые нельзя закрыть убедительным пересказом

| Вопрос | Статус | Почему нет ответа | Как получить |
| --- | --- | --- | --- |
| Единственная причина fresh gripper jump | UNKNOWN | Записан прогноз, но нет expert counterfactual/полного state той точки | План B, C, F, G, K; current replay только различает некоторые пути |
| Физическая pose/cloth trajectory на всех 695 действиях | UNKNOWN | Старый trace не сохранял полный qpos/qvel/flex state каждый tick | Новый отдельный конечный logged rollout, не реконструкция выдуманного old state |
| Реальный expert target для кадра 695 | UNKNOWN | Не записан; teacher из другого состояния неэквивалентен | Воспроизводимое сохранение состояния и отдельный expert query |
| Полная semantic/RGB утечка всех исторических images | UNKNOWN | Проведены inventory и bounded sample, не semantic annotation миллионов кадров | Стратифицированный полный image/scene clustering, заранее лимитировать I/O |
| Точные суммарные GPU-hours за месяц | UNKNOWN | Нет единого usage ledger; timestamps включают ожидание/паузы | Не восстанавливать как created→finished; ввести stage accounting |
| Всё содержимое удалённых 8 октября diagnostic records | UNKNOWN | Есть указание об удалении по просьбе пользователя, состава нет | Только сохранившийся backup, если найден; не обещать восстановление |
| Качество full-room expert corrections | UNKNOWN: полная физическая проверка не выполнена | Новый collector принят программно и в 20-step preflight, не в серии полных recoveries | Gate≥8 полных,≥4student-state recoveries с реальными expert targets |
| Готовность cookie/tray/прочих навыков | NOT_IMPLEMENTED до полного подтверждённого цикла | Есть templates/preparation, текущая квалификация только sock | Отдельная задача/teacher/actor/holdout |
| Точная нынешняя модель MCU | UNKNOWN на текущем read-only доступе | Пользовательская спецификация F407, historical CubeProgrammer сообщал 0x450/H7; текущий DNS недоступен | Read-only identity/photo/model после отдельного подключения; не прошивка |
| Joint feedback текущего робота | VERIFIED отсутствует в действующей конфигурации | Уточнено пользователем; command cache не measured state | Не переобъявлять по старым raw logs; обоснованный estimator без выдуманных encoders |
| Реальные torque/friction/cloth/backlash/delay | UNKNOWN | Simulator estimates; нет полного system ID | Sensitivity и отдельные безопасные измерения |
| Реальные latency/frequency нового пакета на Jetson | UNKNOWN |19.10 `.local` DNS failure;14.10 другая модель и один inference | Non-actuating model/interface timing с теми же inputs |
| Поддержка MPS всех альтернативных репозиториев | UNKNOWN пока не tested | PyTorch/JAX совместимость framework не гарантирует весь stack | Small install/probe в отдельной среде, не ломать active env |
| Насколько VLA pretraining переносится на 6servo+base | UNKNOWN | Другая embodiment/action space; адаптера и данных недостаточно | Контракт/adapter и bounded same-scene comparison |

Отсутствие доказательства не заменяется благоприятной оценкой. Заявления «все прошлые ошибки учтены», «вероятность плохого обучения минимальна», «после загрузки сразу заработает» по доступным данным не обоснованы.

Предел истории: восстановлен основной доступный диалог с27сентября; дополнительные связанные обсуждения просмотрены отдельно и помечены как вторичные. Не все старые вложения и tool stdout доступны в переносимом пакете. Полный исходный state эпизода695, не записанный во время выполнения, в архиве отсутствует. Манифест исторического файла не означает, что файл всё ещё существует; для доступных весов выполнена отдельная потоковая SHA-проверка.
