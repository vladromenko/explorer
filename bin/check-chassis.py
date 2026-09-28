#!/usr/bin/env python3
"""Stationary chassis acceptance. NEVER publishes drive, clears STOP or sets flags."""
import json
import math
from pathlib import Path
import time
import urllib.request
import urllib.error
import yaml
from holonomic_drive import velocity
from safety import SafetyGate

root=Path('/home/vlad/Explorer')
token=(root/'config/access_token').read_text().strip()
def api(path,payload=None):
    req=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
    except urllib.error.HTTPError as exc:return dict(error=json.load(exc).get('detail'),http_status=exc.code)

state=api('status')
assert time.time()-state['at']<2 and state['stop_latched'] and not any(state['velocity'])
cfg=yaml.safe_load((root/'config/gamepad.yaml').read_text())
inputs={'forward':{'1':0},'reverse':{'1':255},'left':{'0':0},'right':{'0':255},
        'ccw':{'2':0},'cw':{'2':255},'diagonal':{'0':0,'1':0},'diagonal_turn':{'0':0,'1':0,'2':255}}
mapping={k:velocity(v,cfg) for k,v in inputs.items()}
assert mapping['left'][1]>0 and mapping['right'][1]<0
assert math.hypot(*mapping['diagonal'][:2])<=cfg['profiles']['precision']['linear']+1e-9
# Pure in-memory controller copies; never writes commissioned flags to disk.
config=dict(state['commissioning'],base_commissioned=True,mcu_watchdog_verified=True)
checks={}
for direction,value in mapping.items():
    gate=SafetyGate(config);gate.estop=False;gate.submit(value,'manual',10)
    initial,reason=gate.tick(10.05,.05,True)
    expired,_=gate.tick(10.31,.05,True)
    assert any(initial) and expired==[0,0,0]
    gate.submit(value,'manual',11);obstacle,_=gate.tick(11,.05,True,True)
    assert obstacle==[0,0,0]
    gate.submit(value,'manual',12);lost,_=gate.tick(12,.05,False)
    assert lost==[0,0,0]
    gate.stop();checks[direction]=dict(ttl_zero=True,obstacle_zero=True,stale_sensor_zero=True)

pose=api('pose');frontiers=api('frontiers')
paths=[]
for candidate in frontiers.get('candidates',[])[:3]:
    result=api('plan',dict(x=candidate['x'],y=candidate['y']))
    paths.append(dict(target=candidate,result=result))
fresh=api('status');assert fresh['stop_latched'] and not any(fresh['velocity'])
report=dict(at=time.time(),motor_commands_sent=False,flags_changed=False,
    gamepad_mapping=mapping,software_gate_checks=checks,
    live_sensor_age=fresh['sensor_age'],dual_lidar=json.loads((root/'data/lidar_geometry.json').read_text()),
    map_pose=pose,frontiers=frontiers,planned_paths=paths,
    hardware_stop=json.loads((root/'config/mcu-link-loss-report.json').read_text()),
    physical_calibration_verified=False,autonomous_drive_admitted=False)
(root/'data/chassis-acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(dict(software_directions_checked=len(checks),map_pose_available='error' not in pose,
    frontier_count=len(frontiers.get('candidates',[])),paths=[dict(points=len(p['result'].get('points',[])),error=p['result'].get('error')) for p in paths],
    lidar_pair_ms=report['dual_lidar'].get('paired_stamp_delta_ms'),motor_commands_sent=False,
    hardware_stop_verified=False),ensure_ascii=False))
