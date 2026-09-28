#!/usr/bin/env python3
"""Observed finite base→hold→empty gripper→base integration, not a delivery."""
import json,time,uuid,urllib.request
from pathlib import Path
root=Path('/home/vlad/Explorer');secret=(root/'config/access_token').read_text().strip()
session=uuid.uuid4().hex;sequence=0;records=[]
def api(path,body=None):
    q=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=10) as r:return json.load(r)
def state():return json.loads((root/'data/status.json').read_text())
def control(op):
    global sequence
    sequence+=1;r=api('control',dict(op=op,session=session,sequence=sequence))
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        s=state()
        if s.get('last_request',{}).get('id')==r['id'] and s['last_request']['ok']:return
        time.sleep(.05)
    raise TimeoutError('No acknowledgement')
def completed(path,result):
    deadline=time.monotonic()+25
    while result['busy'] and time.monotonic()<deadline:
        time.sleep(.1);result=api(path)
    records.append(dict(path=path,result=result,status=state()))
    expected='commanded' if path=='arm/trajectory' else 'completed'
    if result['phase']!=expected:raise ValueError(str(result))
    return result
try:
    s=state()
    assert s['arm_command_state']['servo_deg']==[90,125,3,0,90,30]
    control('stop');control('clear_stop')
    completed('base/step',api('base/step',dict(direction='backward',duration=.7,observing=True,compact=True)))
    assert state()['base_hold_confirmed'] and not state()['stop_latched']
    for jaw in (40,30):
        completed('arm/trajectory',api('arm/finite',dict(goal=[90,125,3,0,90,jaw],observing=True)))
        assert not state()['stop_latched'] and state()['mode']=='MANUAL'
    completed('base/step',api('base/step',dict(direction='forward',duration=.7,observing=True,compact=True)))
    print(json.dumps(dict(passed=True,stages=len(records),rearmed_between_stages=False,pickup_test=False)))
finally:
    control('stop')
    (root/'data/base-arm-handoff-20260928.json').write_text(json.dumps(records,indent=2))
