#!/bin/bash
set -e
source /home/vlad/Explorer/bin/env.sh
cd "$EXPLORER_ROOT"
case "$1" in
 mcu) exec "$EXPLORER_ROOT/vendor/ros/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent" serial --dev /dev/explorer_mcu -b 2000000 -v3;;
 camera) exec ros2 launch orbbec_camera dabai_dcw2.launch.py enable_point_cloud:=false enable_ir:=false depth_registration:=true color_fps:=15 depth_fps:=15;;
 control) exec python3 src/core.py;;
 watchdog) exec python3 src/watchdog.py;;
 oled) exec python3 src/oled.py;;
 perception) exec python3 src/perception.py;;
 web) exec python3 src/web.py;;
 state) exec python3 src/state_estimation.py;;
 geometry) exec python3 src/lidar_geometry.py;;
 mapview) exec python3 src/map_view.py;;
 slam) exec ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false slam_params_file:="$EXPLORER_ROOT/config/slam.yaml";;
 planning) exec ros2 launch "$EXPLORER_ROOT/src/planning.launch.py";;
 ekf) exec ros2 run robot_localization ekf_node --ros-args -r __node:=explorer_ekf --params-file "$EXPLORER_ROOT/config/ekf.yaml";;
 gamepad) exec ros2 run joy game_controller_node --ros-args -p deadzone:=0.15 -p autorepeat_rate:=20.0 -p sticky_buttons:=false;;
 llm) export LD_LIBRARY_PATH="$EXPLORER_ROOT/vendor/llama:$LD_LIBRARY_PATH"
      exec "$EXPLORER_ROOT/vendor/llama/llama-server" -m "$EXPLORER_ROOT/models/Qwen3.5-2B-Q4_K_M.gguf" --mmproj "$EXPLORER_ROOT/models/Qwen3.5-2B-mmproj-F16.gguf" --image-max-tokens 512 --host 127.0.0.1 --port 8081 --alias explorer -c 3072 -ngl 99 -t 2 -tb 2 --jinja --parallel 1;;
 *) echo 'Unknown Explorer component' >&2; exit 2;;
esac
