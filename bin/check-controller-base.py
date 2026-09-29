#!/usr/bin/env python3
"""Bounded observed base acceptance; serial remains connected throughout."""
import json
from pathlib import Path
import struct
import sys
import time
import threading
import serial
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from controller_protocol import Kind,Parser,Session,ClockMapping,encode,base_payload
from controller_feedback import decode_status
s=serial.Serial(port=None,baudrate=2000000,timeout=.001,exclusive=True)
s.dtr=s.rts=False;s.port='/dev/explorer_mcu';s.open()
parser=Parser();session=Session();state={};results=[];history=[];serial_lock=threading.Lock()
def send(packet):
    with serial_lock:s.write(packet)
def poll(seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        for kind,p in parser.feed(s.read(8192)):
            if kind==Kind.STATUS:
                state.update(decode_status(p));history.append(dict(state))
            elif kind==Kind.RESULT:
                _,seq,op,result,mode,fault=struct.unpack('<QQBBBB',p)
                results.append(dict(op=op,result=result));session.result(Kind(op),seq,result)
            elif kind==Kind.IDENTITY:
                nonce,us,boot=struct.unpack_from('<QQQ',p)
                session.synchronize(ClockMapping.observation(boot,nonce,time.monotonic_ns(),us),state.get('highest_session',0))
def sync():
    send(encode(Kind.HELLO,struct.pack('<Q',time.monotonic_ns())));poll(.05)
def cmd(kind,data=b'',lease=150_000_000):
    now=time.monotonic_ns()
    send(session.prepare(kind,now,now+lease,now,data,source_id='observed-base-acceptance',source_sequence=now))
    poll(.04)
def ready():
    sync()
    if state.get('mode')==2:
        session.state='fault';cmd(Kind.CLEAR)
    session.state='disarmed';cmd(Kind.OPEN)
    assert state.get('mode')==1 and state.get('encoder_measurement_valid'),state
records=[];timer=None
cases=[('forward',[.06,0.,0.]),('left',[0.,.06,0.]),('ccw',[0.,0.,.18])]
if '--all-directions' in sys.argv:
    cases=[('forward',[.08,0.,0.]),('backward',[-.08,0.,0.]),
        ('left',[0.,.08,0.]),('right',[0.,-.08,0.]),
        ('ccw',[0.,0.,.25]),('cw',[0.,0.,-.25]),
        ('front_left',[.06,.06,0.]),('rear_right',[-.06,-.06,0.]),
        ('front_right',[.06,-.06,0.]),('rear_left',[-.06,.06,0.])]
try:
    sync();poll(.1)
    for name,velocity in cases:
        ready();before=len(history)
        # Independent fallback, even if the sampling loop raises or stalls.
        timer=threading.Timer(.4,lambda:send(encode(Kind.ESTOP)));timer.start()
        cmd(Kind.BASE,base_payload(velocity),lease=200_000_000)
        poll(.25)
        samples=history[before:]
        assert results[-1]['result']==0,results[-1]
        assert state['mode']==2 and state['fault']==9,state
        assert all(w['pwm']==0 and w['target_rad_s']==0 for w in state['wheels']),state
        timer.cancel();timer.join();timer=None
        record=dict(name=name,command=velocity,lease_ms=200,samples=samples,
            expired_with_zero_pwm=True,serial_disconnected=False)
        records.append(record)
        print(json.dumps(dict(name=name,delta=[sum(s['wheels'][i]['raw_delta'] for s in samples) for i in range(4)],
            expired_with_zero_pwm=True)),flush=True)
        poll(.5)
finally:
    if timer:timer.cancel();timer.join()
    send(encode(Kind.ESTOP));s.close()
    Path('/home/vlad/Explorer/data/controller-acceptance-20260929'/Path('base-tests-'+str(time.time_ns())+'.json')).write_text(json.dumps(records,indent=2))
