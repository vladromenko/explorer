#!/bin/bash
source /opt/ros/jazzy/setup.bash
export EXPLORER_ROOT=/home/vlad/Explorer
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$EXPLORER_ROOT/config/fastdds.xml"
for prefix in "$EXPLORER_ROOT"/vendor/ros/*; do
  export AMENT_PREFIX_PATH="$prefix:${AMENT_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$prefix/lib/python3.12/site-packages:${PYTHONPATH:-}"
done
export PYTHONPATH="$EXPLORER_ROOT/src:${PYTHONPATH:-}"
export PYTHONPATH="$EXPLORER_ROOT/vendor/moveit_root/usr/lib/python3/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$EXPLORER_ROOT/vendor/moveit_root/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$EXPLORER_ROOT/.venv/bin:$PATH"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
