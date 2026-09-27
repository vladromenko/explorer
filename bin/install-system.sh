#!/bin/bash
set -euo pipefail
if [ "$(id -u)" != 0 ]; then
  exec sudo /home/vlad/Explorer/bin/install-system.sh
fi
apt-get update
apt-get install -y --no-install-recommends ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox ros-jazzy-robot-localization \
  ros-jazzy-robot-state-publisher ros-jazzy-xacro ros-jazzy-joy
