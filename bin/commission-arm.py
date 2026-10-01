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
from arm_commissioning import HOME, validate_pose, validate_incremental, stationary_status

ROOT = Path('/home/vlad/Explorer')


def main():
    profile=ROOT/'config/controller-profile.json'
    if profile.exists() and json.loads(profile.read_text()).get('manual_reference_version')==1:
        raise SystemExit('CommandOnly: используйте explorer arm calibrate / jog; legacy HOME отключён')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', type=int, nargs=6, default=HOME)
    parser.add_argument('--runtime-ms', type=int, default=4000)
    parser.add_argument('--incremental-from', type=int, nargs=6,
                        help='Camera-observed preceding command; one joint, <=10 degree step, CAD checked')
    parser.add_argument('--coordinated', action='store_true', help='Observed <=10 degree changes on multiple joints, entire interpolated CAD path checked')
    parser.add_argument('--observed-clear', action='store_true', required=True,
                        help='An observer has checked the swept space and can cut power')
    args = parser.parse_args()
    if args.coordinated and not args.incremental_from:parser.error('Coordinated motion requires an observed starting pose')
    if args.incremental_from:validate_incremental(args.incremental_from,args.pose,args.runtime_ms,args.coordinated)
    else:validate_pose(args.pose, args.runtime_ms)
    lock = (ROOT/'data/arm-commissioning.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    odom=[]

    def check():
        stationary_status(json.loads((ROOT/'data/status.json').read_text()), time.time())
        if odom and (time.monotonic()-odom[0]>.5 or any(abs(v)>=.01 for v in odom[1:])):
            raise ValueError('Measured base motion or stale odometry during arm operation')

    def state(record, phase):
        temporary=ROOT/'data/arm-state.tmp'
        temporary.write_text(json.dumps(dict(record,phase=phase,updated_at=time.time())))
        temporary.replace(ROOT/'data/arm-state.json')

    check()
    if args.incremental_from:
        history=[json.loads(line) for line in (ROOT/'data/arm-commissioning.jsonl').read_text().splitlines()]
        last=history[-1]
        if last.get('event')!='observation_due' or last.get('servo_deg')!=args.incremental_from or not 0<=time.time()-last['at']<900:
            raise ValueError('Prior command is missing, failed, mismatched or too old')
        if last.get('boot_id',boot_id)!=boot_id or last['at']<time.time()-float(Path('/proc/uptime').read_text().split()[0]):
            raise ValueError('Prior command belongs to a previous boot')
        from arm_model import ArmModel
        model=ArmModel()
        # Gripper-to-linkage calibration remains unknown; sample its conservative
        # usable shape range instead of substituting a fictitious measured angle.
        for q in (0.,-.2,-.4,-.6,-.8):
            if not model.path(args.incremental_from[:5],args.pose[:5],q)['valid']:
                raise ValueError('CAD collision in incremental path')
        check()
    rclpy.init(args=[])
    node = rclpy.create_node('explorer_arm_commissioning')
    pub = node.create_publisher(ArmJoints, '/arm6_joints', 1)
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
                      observed_clear=args.observed_clear,boot_id=boot_id,
                      ends_monotonic=time.monotonic()+args.runtime_ms/1000)
        with (ROOT/'data/arm-commissioning.jsonl').open('a') as log:
            log.write(json.dumps(dict(record, event='send_attempt'))+'\n')
        msg = ArmJoints(time=args.runtime_ms)
        for i, value in enumerate(args.pose, 1):
            setattr(msg, f'joint{i}', value)
        pub.publish(msg)
        sent = True
        state(record,'command_in_progress')
        acknowledged = pub.wait_for_all_acked(Duration(seconds=2))
        if not acknowledged:raise RuntimeError('DDS acknowledgement missing; actual arm state is unknown')
        # Agent acknowledgement is not an actuator acknowledgement. Repeat the
        # SAME bounded absolute target once after discovery has settled. This is
        # idempotent in target position; allow a full runtime from the second send.
        repeat_at=time.monotonic()+.3
        while time.monotonic()<repeat_at:
            rclpy.spin_once(node,timeout_sec=.05)
            check()
        pub.publish(msg)
        record['publish_count']=2
        record['ends_monotonic']=time.monotonic()+args.runtime_ms/1000
        state(record,'command_in_progress')
        acknowledged=pub.wait_for_all_acked(Duration(seconds=2))
        if not acknowledged:raise RuntimeError('Second DDS acknowledgement missing; actual arm state is unknown')
        end = time.monotonic()+args.runtime_ms/1000+1
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
            check()
        record['dds_acknowledged'] = acknowledged
        state(record,'command_elapsed_observation_required')
        with (ROOT/'data/arm-commissioning.jsonl').open('a') as log:
            log.write(json.dumps(dict(record, event='observation_due'))+'\n')
        print(json.dumps(record))
        print('Check the physical result. No joint feedback is available.')
    except Exception as exc:
        if sent:
            state(record,'monitor_failed_state_unknown')
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
