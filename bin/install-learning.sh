#!/bin/bash
# Separate environment: never change the ROS/perception Python environment.
set -euo pipefail
task_root=/home/vlad/Explorer
python3 -m venv "$task_root/.venv-learning"
"$task_root/.venv-learning/bin/pip" install --upgrade pip uv
"$task_root/.venv-learning/bin/uv" pip install --python "$task_root/.venv-learning/bin/python" \
  --torch-backend cu130 'lerobot[training]==0.6.1' 'torch==2.11.0' 'torchvision==0.26.0'
"$task_root/.venv-learning/bin/pip" freeze > "$task_root/data/learning-environment.lock.txt"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "$task_root/.venv-learning/bin/python" "$task_root/bin/selftest-learning.py"
"$task_root/.venv-learning/bin/python" "$task_root/bin/check-learning.py"
