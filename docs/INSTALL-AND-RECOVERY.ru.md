# Установка и восстановление Explorer

## Что хранится в Git

Репозиторий содержит весь собственный код Explorer, конфигурации ROS 2, systemd-службы, тесты, документацию и исходники STM32. В Git нет кадров камеры, демонстраций, карт квартиры, журналов, паролей, VPN-ключей и больших сторонних моделей. Команда `git ls-files` не содержит фотографий или видео.

Большие бинарные зависимости отделены, потому что занимают несколько гигабайт и зависят от JetPack. Для них предусмотрены `bin/export-assets.sh` и `bin/install-assets.py`; известные модели сверяются с `config/components.json`.

## Требования

- Jetson Orin Nano 8 GB с совместимым JetPack, Ubuntu 24.04 и CUDA/TensorRT;
- ROS 2 Jazzy в `/opt/ros/jazzy`;
- пользователь `vlad`, checkout в `/home/vlad/Explorer`;
- контроллер CP210x, Orbbec DaBai DCW2, два лидара, геймпад и USB-аудио;
- локальный экспорт `models/` и `vendor/` со старой машины или подготовленного носителя.

Установщик не перепрошивает Jetson или STM32.

## Резервная копия перед заменой железа

```bash
cd /home/vlad/Explorer
bin/export-assets.sh /путь/к/Explorer-assets
tar -C /home/vlad/Explorer -czf /путь/к/Explorer-private-state.tgz \
  config/access_token config/telegram.json config/vpn data/maps data/mobile-demonstrations data/demonstrations
```

Второй архив содержит секреты и данные квартиры. Храните его приватно. Telegram-токен из `~/.config/explorer/secrets/telegram-token` сохраните отдельно с правами `0600`.

## Чистая установка

```bash
sudo mkdir -p /home/vlad
sudo chown vlad:vlad /home/vlad
git clone git@github.com:vladromenko/explorer.git /home/vlad/Explorer
cd /home/vlad/Explorer
bin/install-all.sh --assets /путь/к/Explorer-assets
```

Скрипт устанавливает Nav2/SLAM/EKF, создаёт runtime, learning и voice Python-окружения, устанавливает MoveIt в приватный prefix, восстанавливает большие assets, создаёт `/dev/explorer_mcu`, ставит systemd-службы и запускает тесты.

Если серийный номер нового USB-UART отличается:

```bash
EXPLORER_MCU_SERIAL=НОВЫЙ_СЕРИЙНЫЙ_НОМЕР bin/install-hardware.sh
```

После первого добавления в группу `dialout` войдите в Linux заново.

## Секреты и удалённый доступ

Создание web-ключа:

```bash
python3 - <<'PY'
from pathlib import Path
import secrets
p=Path('/home/vlad/Explorer/config/access_token')
p.write_text(secrets.token_urlsafe(24)+'\n');p.chmod(0o600)
PY
```

Telegram настраивается `bin/configure-telegram.py`; токен хранится только в `~/.config/explorer/secrets/telegram-token`. VPN располагается в `config/vpn/amnezia.conf`, Tailscale авторизуется командой `bin/remote-access login`. Эти файлы игнорируются Git.

## Восстановление состояния

Остановите службы, распакуйте приватный архив в `/home/vlad/Explorer`, затем:

```bash
cd /home/vlad/Explorer
python3 bin/install-services.py
bin/explorer start
bin/explorer status
bin/explorer readiness
```

Калибровки относятся к конкретной механике. После замены камеры, руки, лидаров, колёс или контроллера нельзя переносить соответствующий физический допуск без повторной проверки. «Путь к автономности» покажет нужные упражнения.

## Проверка

```bash
source /home/vlad/Explorer/bin/env.sh
cd /home/vlad/Explorer
python3 -m pytest -q tests
git status --short
git ls-files | grep -Ei '\.(jpg|jpeg|png|gif|webp|mp4|mov)$' || true
```
