#!/usr/bin/env python3
"""Install a user-owned launcher and local service; no privileged actions."""
import os
from pathlib import Path
import plistlib
import shlex
import shutil
import subprocess
import sys
import uuid

root = Path(__file__).resolve().parents[1]
volume = Path(os.environ.get("EXPLORER_SIM_VOLUME", "/Volumes/ExplorerSim"))
# LaunchAgents cannot reliably read Documents under macOS privacy controls.
# Install only the simulator and its public calibration inputs in Application Support.
# No photos, credentials or robot service code enter this runtime snapshot.
support = Path.home() / "Library/Application Support/ExplorerSimulator"
# launchd must invoke an internal executable path, even when an external venv
# symlink resolves to that interpreter. Reuse external packages through .pth.
runtime_env = support / "runtime"
subprocess.run([str(Path(sys.base_prefix) / "bin/python3.12"), "-m", "venv",
                "--without-pip", str(runtime_env)], check=True)
runtime_site = runtime_env / "lib/python3.12/site-packages"
(runtime_site / "explorer-external-packages.pth").write_text(
    str(volume / "env/lib/python3.12/site-packages") + "\n")
runtime_root = support / "source"
staging_id = uuid.uuid4().hex[:12]
installed_root = support / f"source-stage-{staging_id}"
installed_root.mkdir(parents=True, exist_ok=False)
shutil.copytree(root / "simulator", installed_root / "simulator", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name in ("config", "bin", "src", "docs"):
    (installed_root / name).mkdir(exist_ok=True)
for name in ("arm_model.json", "explorer.urdf", "explorer.srdf", "handeye-accepted.json",
             "lidar_geometry.json", "arm-motion.json", "controller-calibration.json", "gripper-reference.json"):
    shutil.copy2(root / "config" / name, installed_root / "config" / name)
for name in ("mobile_policy_contract.py", "learning_environment.py", "simulation_task_contract.py"):
    shutil.copy2(root / "src" / name, installed_root / "src" / name)
(installed_root / "src/__init__.py").write_text("")
shutil.copy2(root / "docs/SIMULATOR-AND-TRANSFER.ru.md", installed_root / "docs/SIMULATOR-AND-TRANSFER.ru.md")
shutil.copy2(root / "bin/start-simulator.command", installed_root / "bin/start-simulator.command")
target = f"gui/{os.getuid()}"
subprocess.run(["launchctl", "bootout", target + "/local.explorer.simulator"], capture_output=True)
if runtime_root.exists():
    runtime_root.rename(support / f"source-previous-{staging_id}")
installed_root.rename(runtime_root)
installed_root = runtime_root
app = Path.home() / "Applications/Explorer Simulator.app"
contents = app / "Contents"
(contents / "MacOS").mkdir(parents=True, exist_ok=True)
plist = {"CFBundleName": "Explorer Simulator", "CFBundleDisplayName": "Explorer Simulator",
         "CFBundleIdentifier": "local.explorer.simulator", "CFBundleVersion": "1.0",
         "CFBundlePackageType": "APPL", "CFBundleExecutable": "launch"}
(contents / "Info.plist").write_bytes(plistlib.dumps(plist))
launch = contents / "MacOS/launch"
launch.write_text("#!/bin/bash\nset -e\nexport EXPLORER_SIM_VOLUME=" + shlex.quote(str(volume)) + "\n"
    "if /usr/bin/curl --fail --silent --max-time 1 http://127.0.0.1:8766/api/state >/dev/null; then\n"
    "  open 'http://127.0.0.1:8766/'\n  exit 0\nfi\n"
    "nohup /bin/bash " + shlex.quote(str(installed_root / "bin/start-simulator.command")) +
    " >>" + shlex.quote(str(support / "service.log")) + " 2>>" +
    shlex.quote(str(support / "service-error.log")) + " </dev/null &\n"
    "for attempt in {1..60}; do\n"
    "  if /usr/bin/curl --fail --silent --max-time 1 http://127.0.0.1:8766/api/state >/dev/null; then\n"
    "    open 'http://127.0.0.1:8766/'\n    exit 0\n  fi\n  sleep 0.5\ndone\n"
    "open " + shlex.quote(str(support / "service-error.log")) + "\n")
launch.chmod(0o755)
service = Path.home() / "Library/LaunchAgents/local.explorer.simulator.plist"
service.parent.mkdir(parents=True, exist_ok=True)
(volume / "ExplorerSim/logs").mkdir(parents=True, exist_ok=True)
target = f"gui/{os.getuid()}"
subprocess.run(["launchctl", "bootout", target, str(service)], capture_output=True)
if service.exists():
    service.rename(support / f"disabled-launch-agent-{staging_id}.plist")
# Separate explicit interactive 3D physics viewer, preserving its macOS main-thread requirements.
viewer = volume / "Explorer 3D.command"
viewer.write_text("#!/bin/bash\nset -euo pipefail\ncd "+shlex.quote(str(installed_root))+"\nexport EXPLORER_SIM_MESH_DIR="+
    shlex.quote(str(volume / "assets/m3_reference/meshes"))+"\nexec "+
    shlex.quote(str(volume / "env/bin/mjpython"))+" -m simulator.viewer\n")
viewer.chmod(0o755)
print(str(app))
