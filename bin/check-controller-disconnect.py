#!/usr/bin/env python3
"""Bounded base pulse followed by a real serial disconnect and readback."""
import json, struct, sys, time
from pathlib import Path
import serial
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from controller_protocol import Kind, Parser, Session, ClockMapping, encode, base_payload
from controller_feedback import decode_status

port='/dev/explorer_mcu'
def opened():
    x=serial.Serial(port=None,baudrate=2000000,timeout=.001,exclusive=True)
    x.dtr=x.rts=False;x.port=port;x.open();return x

s=opened(); parser=Parser(); session=Session(); state={}; results=[]
def poll(seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        for kind,p in parser.feed(s.read(8192)):
            if kind==Kind.STATUS: state.update(decode_status(p))
            elif kind==Kind.IDENTITY:
                nonce,us,boot=struct.unpack_from('<QQQ',p)
                session.synchronize(ClockMapping.observation(boot,nonce,time.monotonic_ns(),us),state.get('highest_session',0))
            elif kind==Kind.RESULT:
                _,seq,op,result,_,_=struct.unpack('<QQBBBB',p)
                results.append((op,result));session.result(Kind(op),seq,result)
def hello(): s.write(encode(Kind.HELLO,struct.pack('<Q',time.monotonic_ns())));poll(.06)
def cmd(kind,data=b'',lease=200_000_000):
    now=time.monotonic_ns();s.write(session.prepare(kind,now,now+lease,now,data,source_id='disconnect-test',source_sequence=now));poll(.12)
try:
    hello();poll(.12);hello()
    if state.get('mode')==2:session.state='fault';cmd(Kind.CLEAR)
    session.state='disarmed';cmd(Kind.OPEN)
    assert session.state=='active',(state,results,session.state)
    cmd(Kind.BASE,base_payload([.05,0.,0.]))
    assert results[-1][1]==0,results
    s.close();time.sleep(.35)
    s=opened();parser=Parser();session=Session();state={};results=[]
    hello();poll(.2)
    assert state.get('mode')==2 and state.get('fault')==9,state
    assert all(w['pwm']==0 and w['target_rad_s']==0 for w in state['wheels']),state
    print(json.dumps({'serial_disconnected':True,'disconnect_s':.35,'fault':state['fault'],'zero_pwm':True,'wheels':state['wheels']}))
finally:
    try:s.write(encode(Kind.ESTOP));s.close()
    except Exception:pass
