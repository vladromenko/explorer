#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m venv "$HOME/Explorer-tools/ruckig-build"
build_python="$HOME/Explorer-tools/ruckig-build/bin/python"
"$build_python" -m pip install scikit-build-core==0.9.10 nanobind==2.4.0 cmake==3.31.6 ninja==1.11.1.3
wheel_dir="$HOME/Explorer-tools/ruckig-wheels"
mkdir -p "$wheel_dir"
CMAKE_BUILD_PARALLEL_LEVEL=2 "$build_python" -m pip wheel --no-deps --no-build-isolation ruckig==0.15.3 -w "$wheel_dir"
"$root/.venv/bin/python" -m pip install --no-deps "$wheel_dir"/ruckig-0.15.3-*.whl
"$root/.venv/bin/python" -c 'import ruckig; assert ruckig.__version__ == "0.15.3"; print("Ruckig Community 0.15.3 ready; no robot motion")'
