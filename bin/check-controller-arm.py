#!/usr/bin/env python3
"""Small observed recovery toward the vendor work interval, base stays zero."""
import dataclasses
import json
import math
from pathlib import Path
import struct
import sys
import time
import serial
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from controller_protocol import Kind,Parser,Session,ClockMapping,encode,arm_payload
from controller_feedback import decode_status,vendor_calibration
s=serial.Serial(port=None,baudrate=2000000,timeout=.001,exclusive=True)
s.dtr=s.rts=False;s.port='/dev/explorer_mcu';s.open()
parser=Parser();session=Session();state={};measured={};results=[];records=[];events=[]
excursion=int(sys.argv[1]) if len(sys.argv)>1 else 0
assert 0<=excursion<=20
gripper_relief='--gripper-relief' in sys.argv
phase_four='--phase-four' in sys.argv
single_step='--single-step' in sys.argv
def poll(seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        for kind,p in parser.feed(s.read(8192)):
            if kind==Kind.STATUS:
                state.update(decode_status(p))
                events.append(dict(event='status',at=time.monotonic(),state=dict(state)))
                assert all(w['pwm']==0 for w in state['wheels']),state
            elif kind==Kind.RESULT:
                acquired,seq,op,result,_,_=struct.unpack('<QQBBBB',p)
                results.append(dict(op=op,result=result,seq=seq,at=time.monotonic(),acquired_us=acquired));session.result(Kind(op),seq,result)
                if result:raise RuntimeError(results[-1])
            elif kind==Kind.SERVO:
                _,joint,error,_,raw,stamp=struct.unpack_from('<QBBBHQ',p)
                events.append(dict(event='servo',at=time.monotonic(),joint=joint,error=error,raw=raw,
                    stamp=stamp,event_us=struct.unpack_from('<Q',p)[0],reply_hex=p[21:29].hex()))
                if error==0:measured[joint]=(raw,time.monotonic())
            elif kind==Kind.IDENTITY:
                nonce,us,boot=struct.unpack_from('<QQQ',p)
                session.synchronize(ClockMapping.observation(boot,nonce,time.monotonic_ns(),us),state.get('highest_session',0))
def sync():
    s.write(encode(Kind.HELLO,struct.pack('<Q',time.monotonic_ns())));poll(.03)
def cmd(kind,data=b''):
    if phase_four and kind==Kind.ARM_RECOVER:
        deadline=time.monotonic()+.12
        while True:
            poll(.001)
            now_ns=time.monotonic_ns()
            recent=next(e for e in reversed(events) if e['event']=='servo' and not e['error'])
            estimate=recent['event_us']+int((time.monotonic()-recent['at'])*1e6)
            tick=state['acquired_us']
            tick+=math.ceil((estimate+2000-tick)/10000)*10000
            steps=(4-recent['joint'])%6
            predicted=recent['stamp']+steps*2550
            while predicted<tick:predicted+=15300
            if 250<=predicted-tick<=2000 and 2000<=tick-estimate<=6000:
                events.append(dict(event='phase_dispatch',at=time.monotonic(),tick=tick,
                    predicted_joint4_us=predicted,latest_joint=recent['joint']))
                break
            if time.monotonic()>deadline:raise RuntimeError('No bounded phase slot; no command sent')
    now=time.monotonic_ns()
    s.write(session.prepare(kind,now,now+200_000_000,now,data,source_id='observed-arm-recovery',source_sequence=now))
    poll((.03 if phase_four else .10) if kind==Kind.ARM_RECOVER else .02)
def raw():
    now=time.monotonic()
    assert len(measured)==6 and all(now-stamp<.15 for _,stamp in measured.values())
    return [measured[i][0] for i in range(1,7)]
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
try:
    sync();poll(.15);sync()
    if state.get('mode')==2:session.state='fault';cmd(Kind.CLEAR)
    session.state='disarmed';cmd(Kind.OPEN)
    initial=raw();cal=[];bounds=[]
    for i,c in enumerate(vendor_calibration()):
        c=dataclasses.replace(c,radians_per_tick=f32(c.radians_per_tick),radians_at_raw_zero=f32(c.radians_at_raw_zero),
            lower=f32(c.lower),upper=f32(c.upper))
        legal=[r for r in range(c.command_min,c.command_max+1) if c.lower<=r*c.radians_per_tick+c.radians_at_raw_zero<=c.upper]
        lo,hi=min(legal),max(legal);bounds.append((lo,hi))
        if initial[i]<lo:c=dataclasses.replace(c,recovery_min=initial[i]-2,recovery_max=lo)
        if initial[i]>hi:c=dataclasses.replace(c,recovery_min=hi,recovery_max=initial[i]+2)
        cal.append(c)
    cmd(Kind.CALIBRATION,b''.join(c.payload() for c in cal))
    cmd(Kind.RECOVERY_ENABLE)
    enabled_us=next(r['acquired_us'] for r in reversed(results) if r['op']==int(Kind.RECOVERY_ENABLE))
    # Match the feedback snapshot used in the MCU enable interrupt. Later
    # readback can differ by a tick; it is not the frozen recovery target.
    initial=[next(e['raw'] for e in reversed(events) if e['event']=='servo' and
        e['joint']==j and e['error']==0 and e['stamp']<enabled_us) for j in range(1,7)]
    print(json.dumps(dict(enabled_reference=initial,bounds=bounds)),flush=True)
    start=time.monotonic();lastsync=start
    # Default is a no-displacement diagnostic; an explicit bounded excursion
    # is used only for an observed trial after the diagnostic passes.
    while time.monotonic()-start<1.8:
        now=time.monotonic()
        if now-lastsync>.4:sync();lastsync=now
        actual=raw();progress=min(1.,(now-start)/1.2)
        ease=progress*progress*(3-2*progress)
        if single_step:ease=1.
        target=[]
        for i,(c,(lo,hi)) in enumerate(zip(cal,bounds)):
            desired=initial[i]
            if initial[i]>hi:desired=max(hi,initial[i]-round(excursion*ease))
            elif initial[i]<lo:desired=min(lo,initial[i]+round(excursion*ease))
            # Recovery freezes joints already in-range at enable time; do not
            # replace their targets with one-tick measurement noise.
            else:desired=initial[i]
            if gripper_relief and i==5 and initial[i]<lo:desired=lo
            assert abs(desired-actual[i])<=24
            target.append(desired*c.radians_per_tick+c.radians_at_raw_zero)
        cmd(Kind.ARM_RECOVER,arm_payload(target))
        records.append(dict(elapsed=now-start,raw=actual))
        if single_step:break
    cmd(Kind.ARM_CANCEL);poll(.15)
    print(json.dumps(dict(initial=initial,final=raw(),bounds=bounds,results=results[-5:])),flush=True)
finally:
    s.write(encode(Kind.ESTOP));s.close()
    path=Path('/home/vlad/Explorer/data/controller-acceptance-20260929')/('arm-diagnostic-'+str(time.time_ns())+'.json')
    path.write_text(json.dumps(dict(excursion=excursion,gripper_relief=gripper_relief,
        phase_four=phase_four,single_step=single_step,records=records,results=results,events=events),indent=2))
    print(str(path),flush=True)
