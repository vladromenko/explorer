"""Wait for real sensor/TF inputs, then supervise one owned Nav2 launch.

No motor publishers, goals, calibration changes, or automatic mission resume.
An inactive lifecycle manager is a failed startup even if ros2 launch is alive.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from navigation_health import LaunchHealth, fresh, power_errors

ROOT = Path(os.environ.get('EXPLORER_ROOT', '/home/vlad/Explorer'))
NODES = dict(planning=('planner_server',), navigation=('planner_server', 'controller_server', 'behavior_server', 'bt_navigator'))


def stop_child(child):
    if child is not None and child.poll() is None:
        for signum, timeout in ((signal.SIGINT, 5.), (signal.SIGTERM, 1.), (signal.SIGKILL, 1.)):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                    child.wait(timeout=timeout)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass


def run(component, check_only=False, timeout=10.):
    import rclpy
    import yaml
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.time import Time
    from rclpy.duration import Duration
    from sensor_msgs.msg import Imu, LaserScan
    from nav_msgs.msg import Odometry, OccupancyGrid
    from lifecycle_msgs.srv import GetState
    from tf2_ros import Buffer, TransformListener, TransformException

    class Monitor(Node):
        def __init__(self):
            super().__init__('explorer_'+component+'_supervisor')
            self.samples, self.states, self.pending = {}, {}, {}
            self.map_ready = False
            self.buffer = Buffer(cache_time=Duration(seconds=5))
            self.listener = TransformListener(self.buffer, self)
            for name, topic, kind in (('odom','/odom_raw',Odometry),('imu','/imu/data_raw',Imu),
                                      ('imu_planar','/explorer/imu_planar',Imu),
                                      ('scan0','/scan0',LaserScan),('scan1','/scan1',LaserScan),
                                      ("scan_merged","/explorer/scan",LaserScan)):
                self.create_subscription(kind, topic, lambda m, key=name: self.receive(key, m), qos_profile_sensor_data)
            self.create_subscription(OccupancyGrid, '/map', self.receive_map,
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.lifecycle_clients = {name: self.create_client(GetState, '/'+name+'/get_state') for name in NODES[component]}
            self.query_due = 0.
            self.power_check_due = 0.
            self.power_blockers = ['power_stale']
            power_config = yaml.safe_load((ROOT/'config/power.yaml').read_text())
            commissioning = json.loads((ROOT/'config/commissioning.json').read_text())
            self.stop_voltage = max(float(power_config['policy']['critical_v']),
                                    float(commissioning['battery_stop_voltage']))
            self.battery_timeout = float(power_config['policy']['sample_timeout_s'])

        def receive(self, key, message):
            valid = True
            if key == 'odom':
                linear, angular = message.twist.twist.linear, message.twist.twist.angular
                valid = all(math.isfinite(v) for v in (linear.x,linear.y,linear.z,angular.x,angular.y,angular.z))
            elif key == 'imu':
                linear, angular = message.linear_acceleration, message.angular_velocity
                valid = all(math.isfinite(v) for v in (linear.x,linear.y,linear.z,angular.x,angular.y,angular.z))
            elif key == 'imu_planar':
                valid = math.isfinite(message.angular_velocity.z) and abs(message.angular_velocity.z)<=5.
            elif key.startswith('scan'):
                valid = (len(message.ranges) > 10 and sum(math.isfinite(v) and
                         message.range_min < v < message.range_max for v in message.ranges) >= 10)
            stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9
            self.samples[key] = (stamp, time.monotonic(), valid)

        def receive_map(self, message):
            self.map_ready = (message.info.width > 0 and message.info.height > 0 and
                              len(message.data) == message.info.width*message.info.height and
                              math.isfinite(message.info.resolution) and message.info.resolution > 0)

        def errors(self):
            wall, mono = self.get_clock().now().nanoseconds/1e9, time.monotonic()
            if mono >= self.power_check_due:
                self.power_check_due = mono+.5
                records = []
                for name in ('status.json', 'power.json'):
                    try:
                        records.append(json.loads((ROOT/'data'/name).read_text()))
                    except (OSError, ValueError):
                        records.append({})
                self.power_blockers = power_errors(*records, time.time(), self.stop_voltage, self.battery_timeout)
            errors = list(self.power_blockers)
            for key in ("odom", "imu", "imu_planar", "scan0", "scan1", "scan_merged"):
                stamp, received, valid = self.samples.get(key, (0., -1e9, False))
                if not valid or not fresh(wall-stamp, mono-received):
                    errors.append(key+'_stale')
            if not self.map_ready:
                errors.append('map_missing')
            for frame in ('odom', 'map'):
                try:
                    transform = self.buffer.lookup_transform(frame, 'base_footprint', Time())
                    stamp = transform.header.stamp.sec+transform.header.stamp.nanosec/1e9
                    if not fresh(wall-stamp, 0.):
                        errors.append(frame+'_tf_stale')
                except TransformException:
                    errors.append(frame+'_tf_missing')
            return errors

        def query(self, now):
            for name, (future, since) in list(self.pending.items()):
                if future.done():
                    try:
                        result = future.result()
                        self.states[name] = (result.current_state.id, now)
                    except Exception:
                        self.states.pop(name, None)
                    del self.pending[name]
                elif now-since > 2.:
                    future.cancel()
                    del self.pending[name]
                    self.states.pop(name, None)
            if now >= self.query_due:
                self.query_due = now+1.
                for name, client in self.lifecycle_clients.items():
                    if name not in self.pending and client.service_is_ready():
                        self.pending[name] = (client.call_async(GetState.Request()), now)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Monitor()
    health = LaunchHealth(NODES[component])
    child = None
    started = time.monotonic()
    reported = None
    report_due = 0.
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
            now = time.monotonic()
            errors = node.errors()
            if check_only:
                if not errors or now-started >= timeout:
                    print(json.dumps(dict(component=component, inputs_ready=not errors, blockers=errors), ensure_ascii=False))
                    return 0 if not errors else 1
            else:
                if child is not None and child.poll() is not None:
                    node.get_logger().error('Owned Nav2 launch exited; systemd will restart the whole group')
                    return 1
                node.query(now)
                phase, blockers = health.evaluate(now, errors, node.states)
                if phase == 'start':
                    for future, _ in node.pending.values():
                        future.cancel()
                    node.pending.clear()
                    node.states.clear()
                    child = subprocess.Popen(['ros2','launch',str(ROOT/'src'/f'{component}.launch.py')], start_new_session=True)
                    health.launched(now)
                    phase = 'starting'
                if now >= report_due:
                    report_due = now+1.
                    report = dict(at=time.time(), component=component, stage=phase, waiting_on=blockers,
                                  lifecycle={k:v[0] for k,v in node.states.items()},
                                  inputs_ready=not errors, launch_pid=child.pid if child else None)
                    destination = ROOT/'data'/f'{component}-health.json'
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix('.tmp')
                    temporary.write_text(json.dumps(report, ensure_ascii=False))
                    temporary.replace(destination)
                    summary = (phase, tuple(blockers))
                    if summary != reported:
                        node.get_logger().info('Nav2 '+phase+(': '+', '.join(blockers) if blockers else ''))
                        reported = summary
                if phase == 'restart':
                    node.get_logger().error('Stopping owned launch: '+', '.join(blockers))
                    return 1
    except KeyboardInterrupt:
        return 0
    finally:
        stop_child(child)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component', choices=tuple(NODES))
    parser.add_argument('--check-only', action='store_true', help='Read inputs only; never launch or stop servers')
    parser.add_argument('--timeout', type=float, default=10.)
    args = parser.parse_args()
    raise SystemExit(run(args.component, args.check_only, max(0., min(args.timeout, 60.))))
