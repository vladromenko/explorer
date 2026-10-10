#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd -P)"
task_dependencies="$task_root/.test-deps"
"$task_root/.venv/bin/python" -m pip install --target "$task_dependencies" -r "$task_root/config/test-requirements.txt"
source "$task_root/bin/env.sh"
export PYTHONPATH="$task_dependencies:$task_root/src:${PYTHONPATH:-}"
cd "$task_root"
"$task_root/.venv/bin/python" -m pytest -q tests "$@"
