#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
sim_volume="${EXPLORER_SIM_VOLUME:-/Volumes/ExplorerSim}"
if ! mount | /usr/bin/grep -Fq " on $sim_volume ("; then
  echo 'Внешний диск ExplorerSim не подключён.'
  exit 0
fi
if [ ! -x "$sim_volume/env/bin/python" ]; then
  echo 'Сначала запустите bin/install-simulator.sh.'
  exit 1
fi
export EXPLORER_SIM_MESH_DIR="$sim_volume/assets/m3_reference/meshes"
export HF_HOME="$sim_volume/cache/huggingface" TORCH_HOME="$sim_volume/cache/torch"
export HF_LEROBOT_HOME="$sim_volume/cache/lerobot" TMPDIR="$sim_volume/tmp"
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
cd "$task_root"
sim_python="$HOME/Library/Application Support/ExplorerSimulator/runtime/bin/python"
if [ ! -x "$sim_python" ]; then
  sim_python="$sim_volume/env/bin/python"
fi
exec "$sim_python" -m simulator.server --port 8766
