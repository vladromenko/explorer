#!/usr/bin/env python3
"""Finite physics/data-mask preflight, never a task qualification or trainer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'))
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    if not args.root.parent.is_mount():raise ValueError('Connect the existing external disk')
    if args.output.exists():raise ValueError('Keep previous probes; choose a new output folder')
    args.output.mkdir(parents=True)
    from simulator.learning import write_json,read_bundle
    from simulator.checkpoint_monitor import paused
    from simulator.live_room_diagnostic import can_start,memory
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    def check():
        if stopped or not args.root.parent.is_mount():raise InterruptedError('Probe stopped or disk absent')
        marker=args.root/'neural-recovery/current.json'
        owner=json.loads(marker.read_text())['id'] if marker.exists() else ''
        if paused(args.root,owner,args.output):raise InterruptedError('Operator pause retained')
    with (args.root/'diagnostics/live-room.lock').open('a') as lease:
        waiting=time.monotonic()
        while True:
            check()
            try:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if can_start(memory()):break
                fcntl.flock(lease,fcntl.LOCK_UN)
            except BlockingIOError:pass
            if not args.wait or time.monotonic()-waiting>1800:
                raise InterruptedError('Existing diagnostic or resource reserve retained')
            write_json(args.output/'state.json',dict(state='waiting',pid=os.getpid(),at=time.time(),training_untouched=True))
            time.sleep(5)
        import numpy as np
        import torch
        from simulator.environment import ExplorerEnv,PERIOD
        from simulator.room_cleanup_port import RoomCleanupPort
        from simulator.room_learning import PhaseRecorder
        from simulator.room_intervention import student_prefix
        from simulator.room_validation import CollisionAudit,independent_outcome
        from simulator.policy_trace import encode_trace
        torch.set_num_threads(1)
        profile=json.loads(args.profile.read_text());read_bundle(args.model)
        env=ExplorerEnv();port=None;recorder=None
        try:
            check();env.reset(seed=1703000014,options=dict(task_id='sock_basket',scene_profile='sock_variants_v1'))
            port=RoomCleanupPort(env,args.output/'observed',args.model,profile,args.output/'unused-vision',cancelled=lambda:stopped)
            port.begin(check);port.geometry.move_angles(np.asarray(profile['initial_command'][:6]),port._ctx(check))
            audit=CollisionAudit(env);env._outcome_observer=audit.sample
            recorder=PhaseRecorder(env,args.output/'recording',profile,1703000014);recorder.begin('pick',{})
            started=float(env.data.time)
            receipt=student_prefix(port,recorder,{},20,check)
            elapsed=float(env.data.time)-started
            truth=independent_outcome(env)
            recorder.finish(False,truth)  # preflight rows NEVER enter the training set
            rows=[json.loads(s) for s in next((args.output/'recording').glob('*/samples.jsonl')).read_text().splitlines()]
            value=dict(kind='room_intervention_physics_and_mask_component_only',state='complete',
                source_revision=json.loads((ROOT/'source-revision.json').read_text())['revision'] if (ROOT/'source-revision.json').exists() else None,
                receipt=receipt,physics_elapsed_s=elapsed,expected_physics_s=receipt['applied_steps']*PERIOD,
                all_student_targets_masked=all(row['training_target_valid'] is False for row in rows),
                finite=bool(np.isfinite(env.data.qpos).all()),warnings=env.data.warning.number.tolist(),
                forbidden_contacts=audit.contacts,video=encode_trace(receipt['trace']),
                full_task_test=False,teacher_correction_verified=False,training_untouched=True,
                simulation_qualified=False,physical_ready=False)
            valid=(bool(rows) and value['all_student_targets_masked'] and abs(elapsed-value['expected_physics_s'])<=1e-6
                and value['finite'] and not any(value['warnings']) and not audit.contacts)
            value['state']='complete' if valid else 'failed'
            write_json(args.output/'result.json',value)
            write_json(args.output/'state.json',dict(state=value['state'],pid=os.getpid(),at=time.time(),
                result=str(args.output/'result.json'),training_untouched=True))
            if not valid:
                raise ValueError('Intervention physics/mask preflight did not pass; evidence retained')
        except Exception as error:
            write_json(args.output/'state.json',dict(state='paused' if isinstance(error,InterruptedError) else 'failed',
                pid=os.getpid(),at=time.time(),reason=str(error),training_untouched=True))
            raise
        finally:
            if recorder is not None and env.step!=recorder.original:recorder.finish(False,{})
            if port is not None:port.close()
            env.close()


if __name__=='__main__':main()
