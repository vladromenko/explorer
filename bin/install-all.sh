#!/bin/bash
set -euo pipefail
root=/home/vlad/Explorer
if [ "$(pwd -P)" != "$root" ]; then echo "Репозиторий должен находиться в $root" >&2; exit 2; fi
assets=''
learning=1
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) assets="${2:?}";shift 2;;
    --without-learning) learning=0;shift;;
    *) echo "Использование: bin/install-all.sh [--assets ПАПКА] [--without-learning]" >&2;exit 2;;
  esac
done
if [ -z "$assets" ]; then
  echo 'Для полной установки нужен экспорт models/vendor: --assets ПАПКА' >&2; exit 2
fi
mkdir -p "$root/data" "$root/logs" "$root/models" "$root/vendor"

sudo "$root/bin/install-system.sh"
sudo apt-get install -y --no-install-recommends python3-venv python3-opencv python3-numpy python3-yaml \
  python3-scipy python3-serial python3-pytest pulseaudio-utils rsync curl git
python3 -m venv --system-site-packages "$root/.venv"
"$root/.venv/bin/pip" install --upgrade pip
"$root/.venv/bin/pip" install -r "$root/config/runtime-requirements.txt"
python3 "$root/bin/install-moveit-user.py"
if [ "$learning" = 1 ]; then "$root/bin/install-learning.sh"; fi
python3 -m venv "$root/.venv-voice"
"$root/.venv-voice/bin/pip" install --upgrade pip
"$root/.venv-voice/bin/pip" install -r "$root/config/voice-environment.lock.txt"
python3 "$root/bin/install-assets.py" --source "$assets"
"$root/bin/install-hardware.sh"
if [ ! -s "$root/config/access_token" ]; then
  "$root/.venv/bin/python" - <<'PY'
from pathlib import Path
import secrets
p=Path('/home/vlad/Explorer/config/access_token');p.write_text(secrets.token_urlsafe(24)+'\n');p.chmod(0o600)
PY
fi
python3 "$root/bin/install-services.py"
source "$root/bin/env.sh"
python3 -m pytest -q tests
echo 'Установка завершена. Настройте секреты по docs/INSTALL-AND-RECOVERY.ru.md и выполните: bin/explorer status'
