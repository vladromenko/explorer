# Explorer — ROSMASTER M3 Pro / Jetson Orin Nano / ROS 2

Единый автономный контур, фактические границы, команды и карта кода: [docs/UNIFIED-AUTONOMY.ru.md](docs/UNIFIED-AUTONOMY.ru.md). Автономная задача запускается во вкладке **Автономность** или командой `explorer autonomy`; ручное управление и прежние данные сохранены.

Explorer — мобильный манипулятор с mecanum-шасси, шестиканальной рукой, двумя лидарами, RGB-D камерой, IMU, геймпадом, микрофоном и динамиком. Репозиторий содержит собственный код Jetson и STM32, web/телефонный интерфейс, навигацию, восприятие, обучение и документацию.

Код на роботе: `/home/vlad/Explorer`. Учебная копия на Mac: `/Users/vladromenko/Documents/Codex/Explorer`.

## Быстрый вход

- Основная панель: `http://explorer.local:8080/`
- Телефон: `http://explorer.local:8080/mobile`
- Ссылка с локальным ключом: `bin/explorer url`
- Telegram: `/status`, `/camera`, `/mobile`, `/experiments`, `/learn`, `/skills`, `/lights`, `/stop`

## Текущий статус

Физически приняты в указанном объёме: голономные направления шасси, остановка после окончания действительного потока команд, геометрия двух лидаров для карты, конечные команды/траектории руки и stationary hand-eye для суставов 1–4. Заводской `robotio` не возвращает измеренное положение шести сервоприводов; углы руки обозначаются как command estimate.

Повторная локализация, раскрытие и удержание захвата, физическая проверка обученной политики и полный pick-and-deliver открываются автоматически после упражнений в разделе **«Путь к автономности»**. Наличие кода не подменяет физическую приёмку.

## Порядок использования

1. Включите робот и web-панель. Проверьте питание, STOP и камеру.
2. Для ручного управления выберите клавиатуру или геймпад в web-панели и разрешите управление. Шасси, выбранный режим руки XYZ/суставы и захват работают одновременно; актуальная раскладка всегда показана под камерой.
3. Для карты откройте «Камера и поездки», запустите SLAM, объедьте комнату, сохраните карту и места.
4. Выполните упражнение повторной локализации в «Пути к автономности».
5. Для обучения откройте вкладку «Обучение», создайте навык и запишите полный показ обычным управлением. Отметьте успех, неудачу или неизвестный результат; этапы переключать не нужно.
6. Пригодный успешный показ автоматически ставит LeRobot ACT в очередь на Explorer. Кандидат проходит независимую offline-проверку, затем требует отдельной наблюдаемой физической попытки.
7. Автономные функции открываются после машинно читаемых порогов и отзываются при недействительных доказательствах.

Ручные шаги, MoveIt и автономные движения руки используют один согласованный плавный профиль: нулевая скорость и ускорение на старте/финише, ограничения скорости/ускорения/рывка и опережающие перекрывающиеся команды для целочисленного интерфейса `robotio`. Параметры находятся в `config/arm-motion.json`.

Мини-экран показывает Explorer-кота, приблизительный процент по напряжению, само напряжение и шкалу батареи. Передняя RGB-панель настраивается из основной и телефонной панелей: `Авто`, `Выкл.`, белые фары, рабочие цвета и заводские эффекты. В `Авто` свет выключен в покое и становится синим при движении, задании или работе руки.

## Команды

```bash
bin/explorer start
bin/explorer stop
bin/explorer status
bin/explorer readiness
bin/explorer logs control
bin/explorer url
bin/explorer mobile-url
bin/explorer delivery status
```

## Установка и восстановление

```bash
git clone git@github.com:vladromenko/explorer.git /home/vlad/Explorer
cd /home/vlad/Explorer
bin/install-all.sh --assets /путь/к/Explorer-assets
```

Подробности: [INSTALL-AND-RECOVERY.ru.md](docs/INSTALL-AND-RECOVERY.ru.md).

## Документация

- [Учебник: архитектура, концепции, технологии и карта кода](docs/ROBOT-TEXTBOOK.ru.md)
- [Понятное управление](docs/operator-guide.ru.md)
- [Эксперименты, живые лидары и сохранение карты](docs/lab-quickstart.ru.md)
- [Полная раскладка клавиатуры и джойстика](docs/MANUAL-CONTROLS.ru.md)
- [Обучение захвату и перевозке](docs/TRAINING-GUIDE.ru.md)
- [Технический свод реализации обучения](docs/LEARNING-IMPLEMENTATION.ru.md)
- [Телефон, LAN, Tailscale и Telegram](docs/MOBILE-REMOTE.ru.md)
- [Автоматические допуски](docs/AUTONOMY-GRADUATION.ru.md)
- [Исследовательские идеи: реализовано и не реализовано](docs/research_coverage.md)
- [Сквозная доставка](docs/DELIVERY-IMPLEMENTATION.ru.md)
- [Аудит кода и подтверждённые ограничения — 08.10.2026](docs/CODE-AUDIT-20261008.ru.md)

## Структура

- `src/` — ROS 2 узлы, алгоритмы, web и исполнители;
- `bin/` — CLI, установка, обучение и диагностика;
- `config/` — конфигурации и физические допуски;
- `systemd/` — пользовательские службы;
- `firmware/` — полный проект STM32;
- `tests/` — программные тесты;
- `docs/` — учебник и инструкции.

`data/`, `models/`, большая часть `vendor/`, секреты, VPN, фотографии и видео не публикуются в Git. Их восстановление предусмотрено установочными скриптами.

### Manual operator UI

Open `http://explorer.local:8080/` for the seven-section desktop interface or `/mobile` for the touch interface. The desktop panel supports keyboard and the Jetson-connected gamepad for both holonomic chassis and six-joint arm control. Diagnostics and allowlisted service logs are available in the Machine status tab. See `docs/operator-guide.ru.md` for the exact controls.
