#!/bin/bash
# Isolated macOS installation; never formats disks or changes robot services.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
sim_volume="${EXPLORER_SIM_VOLUME:-/Volumes/ExplorerSim}"
if [ "$(uname -s)" != Darwin ]; then
  echo 'Этот установщик предназначен для Mac. На Jetson импортируется только пакет политики.'
  exit 1
fi
if ! mount | /usr/bin/grep -Fq " on $sim_volume ("; then
  echo "Подключите внешний диск $sim_volume; установщик не создаёт подмену диска на системном SSD."
  exit 1
fi
mkdir -p "$sim_volume/cache" "$sim_volume/tmp" "$sim_volume/assets/m3_reference" "$sim_volume/ExplorerSim"
# macOS may block launchd executables located on a removable volume.
# Keep the small standalone interpreter internal; packages and datasets stay external.
export UV_CACHE_DIR="$sim_volume/cache/uv" UV_PYTHON_INSTALL_DIR="$HOME/Library/Application Support/ExplorerSimulator/python" TMPDIR="$sim_volume/tmp"
if command -v uv >/dev/null 2>&1; then
  sim_uv="$(command -v uv)"
else
  python3 -m venv "$sim_volume/bootstrap"
  "$sim_volume/bootstrap/bin/python" -m pip install uv==0.13.0
  sim_uv="$sim_volume/bootstrap/bin/uv"
fi
"$sim_uv" python install --no-bin 3.12.14
if [ ! -x "$sim_volume/env/bin/python" ]; then
  sim_python="$("$sim_uv" python find --managed-python 3.12.14)"
  "$sim_uv" venv --python "$sim_python" "$sim_volume/env"
fi
"$sim_uv" pip sync --python "$sim_volume/env/bin/python" "$task_root/config/simulator-macos.lock.txt"
cd "$task_root"
EXPLORER_SIM_VOLUME="$sim_volume" "$sim_volume/env/bin/python" bin/simulator-assets.py
EXPLORER_SIM_VOLUME="$sim_volume" "$sim_volume/env/bin/python" bin/simulator-install-app.py
echo "Explorer Simulator установлен. Данные: $sim_volume/ExplorerSim"
