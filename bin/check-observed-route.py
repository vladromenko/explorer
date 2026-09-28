#!/usr/bin/env python3
"""Two finite observed square routes. Never grants unattended/autonomous admission."""
import json
from pathlib import Path
import time
import urllib.request
import uuid
import numpy as np
from visual_odometry_check import compare

root=Path('/home/vlad/Explorer');folder=root/'data'/('route-check-'+time.strftime('%Y%m%d-%H%M%S'));folder.mkdir()
secret=(root/'config/access_token').read_text().strip();session=uuid.uuid4().hex;seq=0;records=[]
def api(path,body=None):
    req=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)
def state():return json.loads((root/'data/status.json').read_text())
def control(op):
    global seq
    seq+=1;r=api('control',dict(op=op,session=session,sequence=seq));deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        s=state()
        if s.get('last_request',{}).get('id')==r['id']:
            if not s['last_request']['ok']:raise ValueError(str(s['last_request']))
            return
        time.sleep(.05)
    raise TimeoutError('No controller acknowledgement')
def capture(name):
    with np.load(root/'data/rgbd-snapshot.npz',allow_pickle=False) as source:
        if not 0<=time.time()-float(source['stamp'])<1.5:raise ValueError('RGB-D stale')
        sample={k:source[k].copy() for k in source.files}
    np.savez_compressed(folder/(name+'.npz'),**sample)
    return sample
try:
    control('stop');control('clear_stop');initial=state()['raw_pose'];route_starts=[]
    for cycle in range(2):
        start=state()['raw_pose'];route_starts.append(start)
        for direction in ('backward','left','forward','right'):
            before=capture(f'{len(records):02d}-before');pose=state()['raw_pose']
            r=api('base/step',dict(direction=direction,duration=2.,observing=True,compact=True))
            deadline=time.monotonic()+20
            while r['busy'] and time.monotonic()<deadline:
                time.sleep(.1);r=api('base/step')
            s=state()
            after=capture(f'{len(records):02d}-after');visual=compare(before,after)
            record=dict(cycle=cycle,direction=direction,at=time.time(),before_pose=pose,after_pose=s['raw_pose'],
                        visual=visual,result=r)
            records.append(record);(folder/'records.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(dict(cycle=cycle,direction=direction,body_delta=r['result']['body_delta'],visual=visual)),flush=True)
            if r['phase']!='completed' or s['stop_latched'] or not s.get('base_hold_confirmed'):
                raise ValueError('Route interrupted: '+str(r))
        end=state()['raw_pose']
        print(json.dumps(dict(cycle=cycle,return_error_xy_m=float(np.hypot(end['x']-start['x'],end['y']-start['y'])))),flush=True)
finally:
    control('stop')
    (folder/'records.json').write_text(json.dumps(records,indent=2))
    print('Saved '+str(folder),flush=True)
