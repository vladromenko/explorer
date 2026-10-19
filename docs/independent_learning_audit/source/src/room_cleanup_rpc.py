"""Owned sensor/action RPC for isolated room ACT inference on Jetson.

Requests are files inside one explicitly authorized observed session. No SSH,
shell command, dynamic import or arbitrary path is accepted from the worker.
"""
import json
import math
from pathlib import Path
import time

import numpy as np


def save(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False));temporary.replace(path)


class ClientTransport:
    def __init__(self,folder,session,*,timeout=80.):
        self.folder=Path(folder);self.session=session;self.counter=0;self.timeout=timeout;self.action_kind='view'

    def call(self,operation,**arguments):
        self.counter+=1;prefix=self.folder/f'{self.counter:06d}'
        save(prefix.with_suffix('.request.json'),dict(schema=1,session=self.session,operation=operation,arguments=arguments))
        result=prefix.with_suffix('.response.json');deadline=time.monotonic()+self.timeout
        while not result.exists():
            if (self.folder/'cancel.request').exists():raise InterruptedError('Room mission cancelled')
            if time.monotonic()>deadline:raise TimeoutError('Robot RPC did not complete: '+operation)
            time.sleep(.01)
        value=json.loads(result.read_text())
        if value.get('session')!=self.session:raise ValueError('RPC owner changed')
        if value.get('error'):raise ValueError(value['error'])
        if operation=='capture':
            with np.load(prefix.with_suffix('.frame.npz'),allow_pickle=False) as arrays:
                return dict(value['result'],rgb=arrays['rgb'].copy(),depth=arrays['depth'].copy())
        return value['result']

    def permit(self):self.call('permit')
    def preflight(self,contract):return self.call('preflight',contract=contract)
    def cancelled(self):return (self.folder/'cancel.request').exists()
    def state(self):return self.call('state')
    def stop(self):return self.call('stop')
    def apply(self,action,*,reference,period):return self.call('apply',action=action.tolist(),reference=reference.tolist(),period=period,action_kind=self.action_kind)
    def capture(self,after_frame=None):return self.call('capture',after_frame=after_frame)
    def scan(self,scanner):return self.call('scan',scanner=scanner)
    def plan(self,goal):return self.call('plan',goal=goal)
    def frontiers(self):return self.call('frontiers')


class RobotRPC:
    def __init__(self,transport,folder,session):self.transport,self.folder,self.session=transport,Path(folder),session;self.counter=0

    def serve_next(self):
        prefix=self.folder/f'{self.counter+1:06d}';request=prefix.with_suffix('.request.json')
        if not request.exists():return False
        try:
            value=json.loads(request.read_text());result=None
            if value.get('schema')!=1 or value.get('session')!=self.session:raise ValueError('RPC session mismatch')
            operation=value.get('operation');arguments=value.get('arguments')
            if not isinstance(arguments,dict):raise ValueError('Malformed RPC arguments')
            self.transport.permit()
            if operation in ('permit','state','stop','frontiers'):
                if arguments:raise ValueError('Unexpected RPC argument')
                result=getattr(self.transport,operation)()
            elif operation=='preflight':
                if set(arguments)!={'contract'} or not isinstance(arguments['contract'],str):raise ValueError('Invalid preflight binding')
                result=self.transport.preflight(arguments['contract'])
            elif operation=='scan':
                if set(arguments)!={'scanner'} or arguments['scanner'] not in (0,1):raise ValueError('Invalid scanner')
                result=self.transport.scan(arguments['scanner'])
            elif operation=='capture':
                if set(arguments)!={'after_frame'}:raise ValueError('Invalid capture contract')
                frame=self.transport.capture(arguments['after_frame']);result={k:v for k,v in frame.items() if k not in ('rgb','depth')}
                temporary=prefix.with_suffix('.frame.tmp.npz');np.savez_compressed(temporary,rgb=frame['rgb'],depth=frame['depth'])
                temporary.replace(prefix.with_suffix('.frame.npz'))
            elif operation=='plan':
                goal=arguments.get('goal',{})
                if set(arguments)!={'goal'} or set(goal)!={'x','y','yaw'} or not all(
                    type(v) in (int,float) and math.isfinite(v) and abs(v)<100 for v in goal.values()):raise ValueError('Invalid navigation goal')
                result=self.transport.plan(goal)
            elif operation=='apply':
                if (set(arguments)!={'action','reference','period','action_kind'} or arguments['period']!=.05
                        or arguments['action_kind'] not in ('view','policy')):raise ValueError('Invalid policy action contract')
                action=np.asarray(arguments['action'],float);reference=np.asarray(arguments['reference'],float)
                if action.shape!=(9,) or reference.shape!=(9,) or not np.isfinite(np.r_[action,reference]).all():raise ValueError('Invalid finite nine-coordinate action')
                result=self.transport.apply(action,reference=reference,period=.05,action_kind=arguments['action_kind'])
            else:raise ValueError('Operation is not in the room RPC allowlist')
            self.transport.permit();response=dict(session=self.session,result=result)
        except Exception as error:
            self.transport.stop();response=dict(session=self.session,error=str(error))
        save(prefix.with_suffix('.response.json'),response);self.counter+=1;return True
