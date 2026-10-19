# 11. Ранжированные причины, а не объяснение «нужно больше эпох»

Статус относится к фактам и предполагаемому причинному вкладу отдельно. Доказанный дефект не означает доказанную единственную причину всех неудач. Источники: [18_EVIDENCE_INDEX](18_EVIDENCE_INDEX.md), [история](02_FULL_HISTORY.md), [отказ ACT](05_ACT_FORENSICS.md), [данные](06_DATASET_AUDIT.md), [переписка](19_CONVERSATION_AND_DECISION_AUDIT.md).

```text
Нет устойчивого полного навыка
├─ Неустойчивая постановка/организация
│  ├─ не принят минимальный end-to-end contract до длинных запусков
│  ├─ parameter search временно заменил neural pipeline
│  └─ ранние проверки полной миссии добавлены поздно
├─ Данные и teacher
│  ├─ похожие траектории/idle/неполная provenance
│  ├─ мало проверенных correction targets из student states
│  └─ teacher/action/time runtime могут различаться
├─ Моторная политика
│  ├─ clock/phase shortcuts и частичная наблюдаемость — гипотезы
│  ├─ fresh unsafe predictions — факт
│  └─ статистический выбор checkpoint не заменяет closed-loop acceptance
├─ Исполнитель и восприятие
│  ├─ недостижимые обзоры/ложная геометрия/неполная свежесть
│  └─ cadence и action limiting
├─ Инфраструктура
│  ├─ browser stop, renderer, cold loader, memory accounting
│  └─ съёмный диск и неполные сохранения
└─ Перенос
   ├─ commanded вместо measured в текущем factory интерфейсе
   └─ неидентифицированные dynamics/camera/material
```

| № / приоритет | Причина / уверенность | Факты за | Факты против / пробел | Последствие | Минимальная проверка / стоимость | Исправление при подтверждении |
| --- | --- | --- | --- | --- | --- | --- |
| R01 / P0 | Нет стабилизированного сквозного контракта до train; VERIFIED как история, STRONGLY_SUPPORTED как вклад | Cadence22/10 с, phase handoff, повторные raw guards после длинного train | Некоторые локальные эпизоды успешны | Неприменимые в runtime targets | Контрактный replay32 примеров, <10 мин CPU | Единый исполнимый action/time/observation contract |
| R02 / P0 | Подмена neural learning поиском настроек; VERIFIED | Признание 17.10, исходник skill_training, ACT в тот период не дообучалась | Parameter search мог чинить полезный executor | Деньги/время не оптимизировали требуемые веса | Traceability по run+user requirement, <1 ч анализа | Раздельные продуктовые стадии с принятым learner consumer |
| R03 / P0 | Поздняя полная обратная связь; VERIFIED | Room diagnostics только 19.10, блокеры до ACT | Полный запрос уточнялся; локальные тесты были | Суточный train не выявляет невозможный обзор | Один конечный teacher/runtime mission smoke до train | Fail-fast contracts + stable frozen evaluation |
| R04 / P0 | Несогласованная динамика и ограничение команд; STRONGLY_SUPPORTED | Teacher delta до 3.6° против 2°, оценочная физика | Само по себе не объясняет +27.82° | Накопленное отставание от демонстрации | Aligned teacher replay с log qpos, 10–30 мин | Обучать на реально исполнимом контракте, не ослаблять safety |
| R05 / P0 | Недостаточное покрытие recovery; STRONGLY_SUPPORTED | Чистые teacher старты преобладают, новый full collector ещё не принят | 19/24 локальных corrections были успешны | Студент попадает в неизвестные состояния | Матрица 3×5 малых perturbations, 15–60 мин sim | Гарантированные expert targets из посещённых состояний |
| R06 / P1 | Clock shortcut/неправильный phase reset; HYPOTHESIS | 14 clock features, handcrafted nearest phase time | Нет полного clock-ablation с task outcomes | Смена времени заменяет реакцию на предмет | Read-only counterfactual + 10 paired коротких rollouts | Наблюдаемая фаза/goal/память или clock-free вариант, если лучше |
| R07 / P1 | Частичная наблюдаемость wrist RGB; STRONGLY_SUPPORTED как риск | На кадре 695 предмет визуально не подтверждается; measured state нет | В начале предмет виден; сеть может использовать историю команд | Нельзя различить промах/успех по данному кадру | Visibility coverage + frozen-frame label + input ablation | Активный обзор/второй сенсор/краткая история/геометрия по конкурсу |
| R08 / P1 | Низкое разнообразие/idle; VERIFIED наличие, HYPOTHESIS вклад | 1595 близких пар, сходные старты, много idle | Ни near commands, ни thumbnail не доказывают утечку | Validation loss переоценивает перенос | Сценовые кластеры + balanced active frames, 10–60 мин | Dedup-aware split и разнообразие причинных состояний |
| R09 / P1 | Неоднозначные expert targets для похожих входов; HYPOTHESIS | Несколько teacher families, VAEoff | Точных идентичных signatures не найдено | L1 компромисс может быть неисполним | Joint RGB/state/clock bins + target variance, <1 ч | Унификация teacher/goal conditioning или multimodal actor |
| R10 / P1 | Сеть/гиперпараметры не прошли минимальный конкурс; VERIFIED отсутствие конкурса | Выбор ACT по интеграции, нет pairedBC/DP/RL | ACT локально улучшилась 2→5/10 | Неизвестна эффективность архитектуры | Малый одинаковый dataset/seeds, bounded budgets | Выбрать по данным, не по популярности |
| R11 / P1 | Слабая/несопоставимая оценка; VERIFIED история | Offline MAE выбирал веса, разные 10-сцены, teacher metrics в сообщениях | Позднее paired gates исправлены | Ложное ощущение прогресса | Повторить расчёт метрик из первичных JSON, <30 мин | Разделить selection/development/final/real; хранить protocol |
| R12 / P1 | Инфраструктурные простои; VERIFIED | Browser46 с, renderer167attempts>2 ч, loader/cold vision | После ремонтов реальные updates идут | Меньше полезных экспериментов | Status vs log/process/lease, <5 мин | Stage-aware watchdog и честный UI |
| R13 / P1 | Съёмный диск/неполные checkpoint; VERIFIED | 18000 incomplete, resume17000; pause29000complete | Данные сохранены; не полная потеря | Повтор хвоста и долгий cold load | Проверка полного optimizer state + hashes, минуты | Atomic complete marker и safe disconnect; отдельная будущая правка |
| R14 / P0 для real | В текущей системе нет реальной обратной связи суставов; VERIFIED по уточнению пользователя и контракту | State строится по отправленным командам | Ошибка tracking не измерена; исторические raw logs не описывают нынешний интерфейс | Командная оценка не компенсирует люфт/нагрузку | Read-only проверка текущего интерфейса, затем отдельные разрешённые измерения | State estimator по доступным наблюдениям; не предполагать существующий encoder |
| R15 / P0 для real | Неидентифицированная физика/камера/ткань; VERIFIED пробел | kp/kv/torque/friction/material оценочны, pinhole идеален | CAD и некоторые static hand-eye проверки есть | Sim success может не переноситься | Sensitivity sweep + реальные измерения без/с отдельно разрешёнными движениями | System identification и обоснованная randomization |
| R16 / P1 | Ожидание локальной recovery блокирует full collector; VERIFIED | room state waiting_for_local_learning | Single-writer предотвращает реальные конфликты | Поздний task-level feedback | Read-only stable checkpoint observer уже есть; сравнить его latency | Разделить resource leases по writer/read-only и stage gates |

## До следующего длительного обучения

Обязательно проверить R01, R03, R04, R05 и целостность данных/normalizers. Для текущего fresh refusal провести различающие R06–R09. Не требуется заранее выбирать «отказаться от ACT». Требуется определить, решается ли маленький корректный набор и малые recovery perturbations при том же safety/runtime.

Для переноса дополнительно обязательны R14–R15. Проверка Mac не решает их. Новая архитектура не должна зависеть от недоступного реальному роботу hidden state на actor input.
