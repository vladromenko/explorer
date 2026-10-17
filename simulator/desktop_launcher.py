"""One desktop entry point for the neural learning dashboard."""
from pathlib import Path
import shlex

DASHBOARD_URL='http://127.0.0.1:8766/#learning-dashboard'


def render_launcher(source,support):
    source,support=Path(source),Path(support)
    q=shlex.quote
    return '''#!/bin/bash
set -euo pipefail
# Opening the app must never restart or resume an existing training owner.
if /usr/bin/curl --fail --silent --max-time 2 http://127.0.0.1:8766/api/session >/dev/null; then
  /usr/bin/open 'http://127.0.0.1:8766/#learning-dashboard'
  exit 0
fi
if ! /sbin/mount | /usr/bin/grep -Fq ' on /Volumes/ExplorerSim ('; then
  /usr/bin/osascript -e 'display alert "Подключите диск ExplorerSim"'
  exit 0
fi
cd {source}
export EXPLORER_SIM_MESH_DIR=/Volumes/ExplorerSim/assets/m3_reference/meshes
export PYTHONPATH={pythonpath}
export HF_HOME=/Volumes/ExplorerSim/cache/huggingface
export TORCH_HOME=/Volumes/ExplorerSim/cache/torch
export PYTORCH_ENABLE_MPS_FALLBACK=1
nohup {python} -m simulator.server --port 8766 >>{log} 2>>{errors} </dev/null &
for attempt in {{1..180}}; do
  if /usr/bin/curl --fail --silent --max-time 1 http://127.0.0.1:8766/api/session >/dev/null; then
    /usr/bin/open 'http://127.0.0.1:8766/#learning-dashboard'
    exit 0
  fi
  sleep 0.5
done
/usr/bin/open {errors}
'''.format(source=q(str(source)),pythonpath=q(str(source)+':'+str(source/'src')),
        python=q(str(support/'runtime/bin/python')),log=q(str(support/'neural-service.log')),
        errors=q(str(support/'neural-service-error.log')))
