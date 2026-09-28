#!/usr/bin/env python3
"""Observed web STOP during one finite backward step; no link fault injection."""
import json,time,uuid,urllib.request
from pathlib import Path

root=Path('/home/vlad/Explorer');secret=(root/'config/access_token').read_text().strip()
session=uuid.uuid4().hex;sequence=0;record={}
def api(path,body=None):
    req=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=3) as r:return json.load(r)
def state():return json.loads((root/'data/status.json').read_text())
def control(op):
    global sequence
    sequence+=1
    return api('control',dict(op=op,session=session,sequence=sequence))
try:
    s=state()
    assert s['stop_latched'] and s['odom_velocity']==[0.,0.,0.] and 0<=time.time()-s['at']<1
    control('clear_stop');time.sleep(.6)
    accepted=api('base/step',dict(direction='backward',duration=2.,observing=True,compact=True))
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        s=state()
        if abs(s['odom_velocity'][0])>.01:break
        time.sleep(.03)
    if abs(s['odom_velocity'][0])<=.01:raise ValueError('No observed motion to test STOP')
    record['before_stop']=s;record['requested_at_monotonic']=time.monotonic()
    record['ack']=control('stop')
    deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        s=state()
        if s['stop_latched'] and s['velocity']==[0.,0.,0.] and s['odom_velocity']==[0.,0.,0.]:break
        time.sleep(.03)
    record['confirmed_at_monotonic']=time.monotonic();record['after_stop']=s
    record['confirmation_latency_s']=record['confirmed_at_monotonic']-record['requested_at_monotonic']
    record['latency_includes_status_sampling']=True
    deadline=time.monotonic()+8
    result=api('base/step')
    while result['busy'] and time.monotonic()<deadline:
        time.sleep(.1);result=api('base/step')
    record['base_step']=result
    record['passed']=bool(s['stop_latched'] and s['odom_velocity']==[0.,0.,0.] and record['base_step']['phase']=='interrupted')
    print(json.dumps({k:record[k] for k in ('passed','confirmation_latency_s','latency_includes_status_sampling')}))
finally:
    control('stop')
    (root/'data/observed-stop-20260928.json').write_text(json.dumps(record,indent=2))
