#!/bin/bash
set -e
source /home/vlad/Explorer/bin/env.sh
cd "$EXPLORER_ROOT"
case "$1" in
 camera|camera-cache) # Avoid UDP fragment loss for local camera image messages.
   export FASTRTPS_DEFAULT_PROFILES_FILE="$EXPLORER_ROOT/config/fastdds-rgbd.xml";;
esac
case "$1" in
 mcu) # Native controller leases use its monotonic clock and must reconnect
      # even before NTP. Only the legacy micro-ROS epoch startup needs this wait.
      if [ ! -f "$EXPLORER_ROOT/config/controller-profile.json" ]; then
        python3 bin/wait-clock.py --network-grace 60
      fi;;
 camera|camera-cache)
   python3 bin/wait-clock.py;;
esac
case "$1" in
 mcu) if [ -f "$EXPLORER_ROOT/config/controller-profile.json" ]; then
        exec python3 src/controller_driver.py
      fi
      exec "$EXPLORER_ROOT/vendor/ros/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent" serial --dev /dev/explorer_mcu -b 2000000 -v3;;
 camera) # DCW2 SDK aggregation held depth for about 108 ms. Independent
         # delivery preserves registered geometry; Explorer pairs timestamps.
         exec ros2 launch "$EXPLORER_ROOT/bin/camera-rgbd.launch.py" enable_point_cloud:=false enable_ir:=false depth_registration:=true color_fps:=15 depth_fps:=15 enable_frame_sync:=false frame_aggregate_mode:=disable;;
 control) exec python3 src/core.py;;
 power) exec python3 src/power_manager.py;;
 watchdog) exec python3 src/watchdog.py;;
 oled) exec python3 src/oled.py;;
 camera-cache) exec python3 src/camera_cache.py;;
 web) exec python3 src/web.py;;
 *) echo 'Unknown Explorer component' >&2; exit 2;;
esac
