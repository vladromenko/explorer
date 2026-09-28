#!/usr/bin/env python3
"""Verify Jetson's session revocation with stationary motors, never a driving test.

No nonzero velocity or arm command is sent. This does not validate an MCU
motor timeout, and it invalidates the arm command reference until preparation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--execute-stationary-test',action='store_true')
args=parser.parse_args()
if not args.execute_stationary_test:
    print(json.dumps(dict(executed=False,plan=['STOP and verify zero odometry',
        'Select autonomy without issuing a velocity or a mission lease',
        'Pause transport for 0.85 seconds while motors remain at zero',
        'Verify latched STOP and manual mode after connection recovery'])))
    raise SystemExit(0)

root=Path('/home/vlad/Explorer')
key=(root/'config/access_token').read_text().strip()
session=str(uuid.uuid4());sequence=0
def command(op,**values):
    global sequence
    sequence+=1
    body=dict(op=op,session=session,sequence=sequence,**values)
    request=urllib.request.Request('http://127.0.0.1:8080/api/control',
        data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)

def state():
    s=json.loads((root/'data/status.json').read_text())
    if not 0<=time.time()-s['at']<1:raise ValueError('Stale status')
    if any(abs(v)>.001 for v in s['velocity']):raise ValueError('Nonzero base command')
    if any(abs(v)>limit for v,limit in zip(s['odom_velocity'],(.005,.005,.02))):
        raise ValueError('Base is not stationary')
    return s

def wait_ack(result):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        s=state();ack=s.get('last_request',{})
        if ack.get('id')==result['id']:
            if not ack.get('ok'):raise ValueError('Request rejected')
            return s
        time.sleep(.05)
    raise TimeoutError('Request acknowledgement unavailable')

pid=int(subprocess.check_output(['systemctl','--user','show','explorer-mcu.service','-p','MainPID','--value']))
if pid<=1 or 'micro_ros_agent' not in str(Path(f'/proc/{pid}/exe').resolve()) or Path(f'/proc/{pid}').stat().st_uid!=os.getuid():
    raise ValueError('Expected the user-owned micro-ROS agent')
state()
arm=json.loads((root/'data/arm-state.json').read_text())
if arm.get('phase')!='command_elapsed_observation_required':raise ValueError('Arm must be idle')
report=dict(at=time.time(),nonzero_commands_sent=False,hardware_motor_timeout_verified=False,passed=False)
rescue=None;paused=False
try:
    before=wait_ack(command('stop'))
    if not before['stop_latched']:raise ValueError('STOP not latched')
    if before['sensor_age'].get('odom',99)>.3:raise ValueError('No fresh baseline odometry')
    wait_ack(command('mode',mode='AUTONOMOUS'))
    armed=wait_ack(command('clear_stop'))
    if armed['mode']!='AUTONOMOUS' or armed['stop_latched']:raise ValueError('Baseline mode not established')
    # An independent process resumes the existing agent even if this process dies.
    rescue=subprocess.Popen([sys.executable,'-c',
        'import os,signal,time;time.sleep(3);os.kill('+str(pid)+',signal.SIGCONT)'])
    start=time.monotonic();os.kill(pid,signal.SIGSTOP);paused=True
    time.sleep(.85)
    os.kill(pid,signal.SIGCONT);paused=False
    report['transport_pause_s']=time.monotonic()-start
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        recovered=state()
        if recovered['at']>armed['at']+1 and recovered['sensor_age'].get('odom',99)<.3:break
        time.sleep(.05)
    else:raise TimeoutError('No fresh odometry after recovery')
    report.update(stop_latched=recovered['stop_latched'],mode=recovered['mode'],
                  final_velocity=recovered['odom_velocity'])
    report['passed']=recovered['stop_latched'] is True and recovered['mode']=='MANUAL'
except Exception as exc:
    report['error']=str(exc)
finally:
    if paused:os.kill(pid,signal.SIGCONT)
    if rescue is not None:
        rescue.terminate();rescue.wait(timeout=2)
    try:wait_ack(command('stop'))
    finally:
        (root/'data/mcu-stationary-link-audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
if not report['passed']:raise SystemExit(1)
