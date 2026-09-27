#!/usr/bin/env python3
"""One observed near-home arm command. Never run unattended or at boot.

The base stop does NOT interrupt servo motion. The observer must be able to cut
power. Successful DDS publication is not proof of the attained joint angles.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

import rclpy
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from arm_msgs.msg import ArmJoints
from nav_msgs.msg import Odometry
from arm_commissioning import HOME, validate_pose, stationary_status

ROOT = Path('/home/vlad/Explorer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', type=int, nargs=6, default=HOME)
    parser.add_argument('--runtime-ms', type=int, default=4000)
    parser.add_argument('--observed-clear', action='store_true', required=True,
                        help='An observer has checked the swept space and can cut power')
    args = parser.parse_args()
    validate_pose(args.pose, args.runtime_ms)
    lock = (ROOT/'data/arm-commissioning.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def check():
        stationary_status(json.loads((ROOT/'data/status.json').read_text()), time.time())

    check()
    rclpy.init()
    node = rclpy.create_node('explorer_arm_commissioning')
    pub = node.create_publisher(ArmJoints, '/arm6_joints', 1)
    odom = []
    def on_odom(msg):
        t = msg.twist.twist
        odom[:] = [time.monotonic(), t.linear.x, t.linear.y, t.angular.z]
    node.create_subscription(Odometry, '/odom_raw', on_odom, qos_profile_sensor_data)
    record = None
    sent = False
    try:
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            check()
            if pub.get_subscription_count() and odom:
                break
        if not pub.get_subscription_count() or not odom:
            raise RuntimeError('No connected arm subscriber or odometry; nothing sent')
        if time.monotonic()-odom[0] > .5 or any(not abs(v) < .01 for v in odom[1:]):
            raise RuntimeError('Measured chassis motion is not stationary; nothing sent')
        check()
        record = dict(at=time.time(), servo_deg=args.pose, runtime_ms=args.runtime_ms,
                      source='commanded_only', measured=False, attained=False,
                      observed_clear=args.observed_clear)
        with (ROOT/'data/arm-commissioning.jsonl').open('a') as log:
            log.write(json.dumps(dict(record, event='send_attempt'))+'\n')
        msg = ArmJoints(time=args.runtime_ms)
        for i, value in enumerate(args.pose, 1):
            setattr(msg, f'joint{i}', value)
        pub.publish(msg)
        sent = True
        acknowledged = pub.wait_for_all_acked(Duration(seconds=2))
        end = time.monotonic()+args.runtime_ms/1000+1
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
            check()
        record['dds_acknowledged'] = acknowledged
        with (ROOT/'data/arm-commissioning.jsonl').open('a') as log:
            log.write(json.dumps(dict(record, event='observation_due'))+'\n')
        print(json.dumps(record))
        print('Check the physical result. No joint feedback is available.')
    except Exception as exc:
        if sent:
            with (ROOT/'data/arm-commissioning.jsonl').open('a') as log:
                log.write(json.dumps(dict(record, event='monitor_failed', error=str(exc)))+'\n')
            print('Command was sent; it may finish despite this monitoring failure. '
                  'Observe the arm; this program has not stopped its servos.', flush=True)
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
