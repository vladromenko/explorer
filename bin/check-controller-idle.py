#!/usr/bin/env python3
"""Check recovery, idle and rejection of delayed zero commands. No movement."""
import json
from pathlib import Path
import struct
import sys
import time
import serial
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from controller_protocol import Kind, Parser, encode, Session, ClockMapping, base_payload
from controller_feedback import decode_status

s = serial.Serial(port=None, baudrate=2000000, timeout=.001, exclusive=True)
s.dtr = s.rts = False
s.port = '/dev/explorer_mcu'
s.open()
parser, session = Parser(), Session()
state, results = {}, []
def poll(seconds):
    until = time.monotonic()+seconds
    while time.monotonic()<until:
        for kind, p in parser.feed(s.read(8192)):
            if kind == Kind.STATUS:
                state.update(decode_status(p))
                assert all(w['pwm']==0 and w['target_rad_s']==0 for w in state['wheels'])
            elif kind == Kind.RESULT:
                _, seq, op, result, mode, fault = struct.unpack('<QQBBBB',p)
                results.append(dict(operation=op,result=result,mode=mode,fault=fault))
                session.result(Kind(op), seq, result)
            elif kind == Kind.IDENTITY:
                nonce, us, boot = struct.unpack_from('<QQQ',p)
                session.synchronize(ClockMapping.observation(boot,nonce,time.monotonic_ns(),us),state.get('highest_session',0))
def sync():
    sent = time.monotonic_ns()
    s.write(encode(Kind.HELLO,struct.pack('<Q',sent)))
    poll(.06)
def command(kind,data=b''):
    sync()
    now = time.monotonic_ns()
    s.write(session.prepare(kind,now,now+150_000_000,now,data,
        source_id='zero-only-acceptance',source_sequence=now))
    poll(.06)
try:
    sync()
    if state.get('mode')==2:
        session.state='fault'
        command(Kind.CLEAR)
    command(Kind.OPEN)
    command(Kind.BASE,base_payload([0.,0.,0.]))
    poll(.3)
    assert state.get('mode')==1 and state.get('fault')==0, state
    idle = dict(state)
    sync()
    now = time.monotonic_ns()
    expired_packet = session.prepare(Kind.BASE,now,now+100_000_000,now,base_payload([0.,0.,0.]),
        source_id='zero-only-acceptance',source_sequence=now)
    poll(.2)
    s.write(expired_packet)
    poll(.06)
    assert results[-1]['result']==9, results
    s.write(encode(Kind.ESTOP)); poll(.05)
    session.state='fault'
    command(Kind.CLEAR)
    command(Kind.OPEN)
    command(Kind.HOLD)
    command(Kind.CANCEL)
    assert state.get('mode')==0, state
    print(json.dumps(dict(zero_only=True,delayed_command_rejected=True,idle_stays_active=True,
        moving_lease_expiry_tested=False,recovery_disarmed=True,
        results=results,idle_status=idle,final_status=state),indent=2))
finally:
    # This terminal command also leaves motors disarmed even if an assertion failed.
    s.write(encode(Kind.ESTOP))
    s.close()
