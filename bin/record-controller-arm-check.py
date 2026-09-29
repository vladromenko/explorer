#!/usr/bin/env python3
"""Record the shared ROS driver. Optional enable check sends NO position target."""
import argparse
import json
from pathlib import Path
import secrets
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from controller_release import atomic_json
from controller_feedback import load_calibration
from controller_arm_commissioning import require_arm_test_power


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--check-gate',action='store_true')
    args.add_argument('--commission-enable',action='store_true')
    args.add_argument('--move-joint',type=int,choices=range(1,7))
    args.add_argument('--ticks',type=int,default=20)
    options=args.parse_args()
    if (options.commission_enable or options.move_joint) and json.loads((ROOT/'config/controller-profile.json').read_text()).get('manual_reference_version')==1:
        args.error('CommandOnly: используйте explorer arm calibrate / jog; запись без движения доступна')
    if options.check_gate and options.commission_enable:args.error('choose one check')
    if options.move_joint and (not options.commission_enable or not 12<=abs(options.ticks)<=20):
        args.error('a motion check requires --commission-enable and 12..20 signed ticks')
    import rclpy
    from std_msgs.msg import String
    rclpy.init(args=[])
    node=rclpy.create_node('explorer_arm_trace')
    source='arm-check-'+secrets.token_hex(12)
    events=[];results={};sequence=0
    def receive(topic,message):
        data=json.loads(message.data)
        events.append(dict(topic=topic,at=time.time(),monotonic_ns=time.monotonic_ns(),data=data))
        if topic.endswith('controller_result') and data.get('source_id')==source:
            results[data.get('source_sequence')]=data
    subscriptions=[]
    for topic in ('/explorer/controller_result','/explorer/servo_sample','/explorer/controller_state'):
        subscriptions.append(node.create_subscription(String,topic,lambda m,t=topic:receive(t,m),200))
    pub=node.create_publisher(String,'/explorer/controller_request',10)
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.005)
    def command(operation,required=True,**fields):
        nonlocal sequence
        sequence+=1;now=time.monotonic_ns()
        request=dict(operation=operation,source_id=source,source_sequence=sequence,
            source_monotonic_ns=now,expires_monotonic_ns=now+200_000_000,**fields)
        events.append(dict(topic='request',monotonic_ns=now,data=request))
        pub.publish(String(data=json.dumps(request)))
        end=time.monotonic()+.3
        while sequence not in results and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.005)
        result=results.get(sequence)
        if result is None:raise RuntimeError('No matching result; not retried')
        if required and not result.get('accepted'):raise RuntimeError(json.dumps(result))
        return result
    permit=ROOT/'data/controller-arm-permit.json';created=False;error=None
    position_requested=False;position_accepted=False;attainment=None
    try:
        spin(1.)
        if options.check_gate or options.commission_enable:
            discovery_end=time.monotonic()+4.
            while pub.get_subscription_count()==0 and time.monotonic()<discovery_end:spin(.05)
        saved=json.loads((ROOT/'data/controller-state.json').read_text())
        profile=json.loads((ROOT/'config/controller-profile.json').read_text())
        if (not saved.get('telemetry_fresh') or
            not 0<=time.monotonic_ns()-saved.get('monotonic_ns',0)<200_000_000):
            raise RuntimeError('No fresh shared-driver state')
        if (options.check_gate or options.commission_enable) and pub.get_subscription_count()!=1:
            raise RuntimeError('Exactly one shared controller request subscriber is required; observed '+str(pub.get_subscription_count()))
        if options.check_gate or options.commission_enable:
            if saved.get('telemetry_only') is not True or profile.get('telemetry_only') is not True:
                raise RuntimeError('This check requires a telemetry-only profile')
        if options.check_gate:
            result=command('ARM_ENABLE',required=False)
            if result.get('accepted'):raise RuntimeError('Unexpected enable acceptance')
        elif options.commission_enable:
            # Fail before CLEAR/OPEN/enable when the motion test cannot proceed.
            if options.move_joint:require_arm_test_power(ROOT)
            if permit.exists():raise RuntimeError('Another arm commissioning permit already exists')
            identity=saved['identity'];now=time.monotonic_ns()
            grant=dict(source_id=source,issued_ns=now,expires_ns=now+10_000_000_000,
                calibration_sha256=profile['calibration_sha256'],
                **{k:identity[k] for k in ('uid','boot','source_sha256')})
            # O_EXCL prevents replacing another diagnostic owner's authority.
            with permit.open('x') as f:json.dump(grant,f)
            permit.chmod(0o600);created=True
            if saved.get('session_state')=='fault':command('CLEAR')
            command('OPEN');command('CALIBRATION')
            command('ARM_ENABLE')
            if options.move_joint:
                cal=load_calibration(ROOT/'config/controller-calibration.json')
                reference=json.loads((ROOT/'data/controller-state.json').read_text())
                samples=reference['arm']['joints']
                if not reference['arm']['all_fresh'] or len(samples)!=6:
                    raise RuntimeError('Fresh six-joint reference missing')
                positions=[];selected=options.move_joint-1;target_raw=None
                for index,(c,sample) in enumerate(zip(cal,samples)):
                    if sample['outside_soft_limit'] or sample['error'] or not sample['position_valid']:
                        raise RuntimeError('Motion probe needs all joints inside accepted work limits')
                    raw=sample['raw_ticks']+(options.ticks if index==selected else 0)
                    positions.append(c.target_position(raw*c.radians_per_tick+c.radians_at_raw_zero))
                    if index==selected:target_raw=raw
                sent_ns=time.monotonic_ns();position_requested=True
                command('ARM',position_rad=positions)
                position_accepted=True
                spin(.12)
                observed=[e['data'] for e in events if e['topic'].endswith('servo_sample') and
                    e['monotonic_ns']>sent_ns and e['data'].get('joint')==options.move_joint and
                    e['data'].get('error')==0 and e['data'].get('position_valid') and
                    e['data'].get('acquired_monotonic_ns',0)>sent_ns]
                reached=len(observed)>=3 and all(abs(v['raw_ticks']-target_raw)<=3 for v in observed[-3:])
                attainment=dict(joint=options.move_joint,initial_raw=samples[selected]['raw_ticks'],
                    target_raw=target_raw,last_raw=[v['raw_ticks'] for v in observed[-3:]],
                    measured_reached=reached)
                if not reached:raise RuntimeError('Measured target arrival not established; no retry')
            command('ARM_CANCEL')
            spin(.15)
            cancelled=json.loads((ROOT/'data/controller-state.json').read_text())
            if (not cancelled.get('telemetry_fresh') or cancelled['controller']['arm_enabled'] or
                cancelled['controller']['arm_cancel_pending']):
                raise RuntimeError('Completed cancellation not established')
        spin(.5)
    except Exception as exc:
        error=str(exc)
    finally:
        if created:
            now=time.monotonic_ns()
            pub.publish(String(data=json.dumps(dict(operation='ESTOP',source_id=source,
                source_sequence=sequence+1,source_monotonic_ns=now,expires_monotonic_ns=now+200_000_000))))
            spin(.3)
            if json.loads(permit.read_text()).get('source_id')==source:permit.unlink()
        folder=ROOT/'data/controller-acceptance-20260929';folder.mkdir(exist_ok=True)
        path=folder/('ros-arm-check-'+str(time.time_ns())+'.json')
        atomic_json(path,dict(error=error,check_gate=options.check_gate,
            commission_enable=options.commission_enable,position_command_requested=position_requested,
            position_command_accepted=position_accepted,attainment=attainment,events=events))
        print(json.dumps(dict(path=str(path),error=error,results=list(results.values()))),flush=True)
        node.destroy_node();rclpy.shutdown()
    if error:raise SystemExit(1)


if __name__=='__main__':main()
