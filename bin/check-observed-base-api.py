#!/usr/bin/env python3
"""Physically observed API acceptance: two bounded motions, no fault injection.

Run only with clear space and a live observer. Leaves explicit STOP latched.
"""
import json
from pathlib import Path
import time
import urllib.request
import uuid

root=Path('/home/vlad/Explorer');secret=(root/'config/access_token').read_text().strip()
session=uuid.uuid4().hex;sequence=0;records=[]
def api(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)
def control(op):
    global sequence
    sequence+=1
    return api('control',dict(op=op,session=session,sequence=sequence))
def await_ack(request_id):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        s=json.loads((root/'data/status.json').read_text())
        if s.get('last_request',{}).get('id')==request_id:
            if not s['last_request']['ok']:raise RuntimeError(str(s['last_request']))
            return s
        time.sleep(.05)
    raise TimeoutError('Request acknowledgement unavailable')
try:
    await_ack(control('stop')['id'])
    await_ack(control('clear_stop')['id'])
    for direction in ('forward','backward'):
        accepted=api('base/step',dict(direction=direction,duration=.7,observing=True,compact=True))
        deadline=time.monotonic()+20;result=accepted
        while result['busy'] and time.monotonic()<deadline:
            time.sleep(.1);result=api('base/step')
        s=json.loads((root/'data/status.json').read_text())
        records.append(dict(direction=direction,result=result,controller=s))
        if result['phase']!='completed' or s['stop_latched'] or not s.get('base_hold_confirmed'):
            raise RuntimeError('Normal hold was not confirmed: '+str(result))
    print(json.dumps(dict(passed=True,motions=len(records),no_rearming_between_motions=True)))
finally:
    control('stop')
    (root/'data/observed-base-api-20260928.json').write_text(json.dumps(records))
