#!/usr/bin/env python3
"""No direct UART. Run through bin/env.sh or bin/explorer arm."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ.get('EXPLORER_ROOT', '/home/vlad/Explorer'))
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description='Explorer: ручная опорная установка')
    subs=parser.add_subparsers(dest='operation', required=True)
    subs.add_parser('status')
    cal=subs.add_parser('calibrate')
    cal.add_argument('action',choices=('begin','capture','cancel'))
    cal.add_argument('--supported',action='store_true')
    jog=subs.add_parser('jog'); jog.add_argument('--joint',type=int,required=True)
    jog.add_argument('--delta-deg',type=float,required=True); jog.add_argument('--duration',type=float,default=1)
    move=subs.add_parser('move'); move.add_argument('--servo-deg',type=float,nargs=6,required=True)
    move.add_argument('--duration',type=float,default=2)
    subs.add_parser('stop'); subs.add_parser('read')
    args=parser.parse_args()
    if args.operation=='status':
        state=json.loads((ROOT/'data/controller-state.json').read_text())
        print(json.dumps(dict(identity=state.get('identity'),telemetry_fresh=state.get('telemetry_fresh'),
            manual_reference=state.get('manual_reference'),arm_state=state.get('arm')),ensure_ascii=False,indent=2))
        return
    from manual_reference_client import ManualClient
    client=None
    try:
        # STOP/cancel must not wait for another client's long-running move.
        if not (args.operation=='stop' or args.operation=='calibrate' and args.action=='cancel'):
            lock=(ROOT/'data/manual-reference-operator.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        client=ManualClient(ROOT)
        if args.operation=='calibrate':
            if args.action=='begin': result=client.begin(args.supported)
            elif args.action=='capture': result=client.capture(args.supported)
            else: result=client.send('MR_ABORT')
        elif args.operation=='jog': result=client.jog(args.joint,args.delta_deg,args.duration)
        elif args.operation=='move': result=client.move(args.servo_deg,args.duration)
        elif args.operation=='read': result=client.send('MR_DIAGNOSTIC')
        else: result=client.stop()
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        if client is not None:
            try: client.stop()
            except Exception: pass
        raise SystemExit('ОШИБКА: '+str(exc)) from exc
    finally:
        if client is not None: client.close()


if __name__=='__main__': main()
