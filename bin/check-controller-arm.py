#!/usr/bin/env python3
"""Small observed recovery toward the vendor work interval, base stays zero."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from controller_protocol import Kind,Parser,Session,ClockMapping,encode,arm_payload
from controller_feedback import decode_status,load_calibration
from arm_commissioning import stationary_status
ROOT=Path(__file__).resolve().parents[1]

def approved_calibration(root, state):
    """Preserve the persisted calibration; never derive a corridor from raw."""
    path=root/'config/controller-calibration.json'
    profile=json.loads((root/'config/controller-profile.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=profile.get('calibration_sha256'):
        raise ValueError('Calibration hash does not match the accepted profile')
    identity=state.get('identity') or {}
    if identity.get('source_sha256')!=profile.get('firmware_source_sha256'):
        raise ValueError('Fresh board source identity does not match profile')
    cal=load_calibration(path)
    data=json.loads(path.read_text())
    if any(c.recovery_min or c.recovery_max for c in cal):
        digest=data.get('recovery_evidence_sha256','')
        if len(digest)!=64 or any(ch not in '0123456789abcdef' for ch in digest):
            raise ValueError('Recovery acceptance hash missing')
        evidence=root/'data/controller-recovery-acceptance'/(digest+'.json')
        if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest()!=digest:
            raise ValueError('Verified recovery acceptance record is unavailable')
        record=json.loads(evidence.read_text())
        expected=[[c.recovery_min,c.recovery_max] for c in cal]
        if (record.get('source_sha256')!=identity.get('source_sha256') or
            record.get('board_uid')!=identity.get('uid') or
            record.get('corridors')!=expected or record.get('physically_verified') is not True):
            raise ValueError('Recovery evidence belongs to another board, image or corridor')
    return cal

def checked_bounds(cal, samples):
    if len(samples)!=6:
        raise ValueError('Six fresh measured joint positions required')
    bounds=[]
    for c,sample in zip(cal,samples):
        if (sample.get('joint')!=c.joint or not sample.get('fresh') or
            not sample.get('raw_valid') or not sample.get('position_valid') or
            sample.get('error')!=0 or sample.get('device_error')!=0):
            raise ValueError('Invalid measurement for joint '+str(c.joint))
        legal=[r for r in range(c.command_min,c.command_max+1)
               if c.lower<=r*c.radians_per_tick+c.radians_at_raw_zero<=c.upper]
        lo,hi=min(legal),max(legal);bounds.append((lo,hi))
        raw=sample['raw_ticks']
        if not lo<=raw<=hi:
            if not (c.recovery_min and c.recovery_min<=raw<=c.recovery_max and
                    ((c.recovery_max==lo and raw<lo) or (c.recovery_min==hi and raw>hi))):
                raise ValueError('Joint '+str(c.joint)+' is outside work limits; no verified recovery corridor. No UART opened.')
    return bounds

parser=None;session=None;state={};measured={};results=[];records=[];events=[]
s=None;excursion=0;gripper_relief=False;phase_four=False;single_step=False
expected_identity=None
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
                results.append(dict(op=op,result=result,seq=seq,at=time.monotonic(),acquired_us=acquired))
                session.result(Kind(op),seq,result)
                if result:raise RuntimeError(results[-1])
            elif kind==Kind.SERVO:
                _,joint,error,_,raw,stamp=struct.unpack_from('<QBBBHQ',p)
                events.append(dict(event='servo',at=time.monotonic(),joint=joint,error=error,raw=raw,
                    stamp=stamp,event_us=struct.unpack_from('<Q',p)[0],reply_hex=p[21:29].hex()))
                if error==0:measured[joint]=(raw,time.monotonic())
            elif kind==Kind.IDENTITY:
                nonce,us,boot=struct.unpack_from('<QQQ',p)
                if (len(p)!=88 or expected_identity is None or
                    p[36:48].hex()!=expected_identity['uid'] or
                    p[48:80].hex()!=expected_identity['source_sha256'] or
                    boot!=expected_identity['boot']):
                    raise RuntimeError('Live controller identity changed; no arm command authorized')
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
    before=len(results)
    now=time.monotonic_ns()
    packet=session.prepare(kind,now,now+200_000_000,now,data,
        source_id='observed-arm-recovery',source_sequence=now)
    sequence=session.sequence
    events.append(dict(event='command',at=time.monotonic(),op=int(kind),
        sequence=sequence,payload_hex=data.hex(),source_ns=now,expires_ns=now+200_000_000))
    if s.write(packet)!=len(packet):raise RuntimeError('Partial command write; not retried')
    deadline=time.monotonic()+.12
    while time.monotonic()<deadline:
        poll(.005)
        matching=[r for r in results[before:] if r['op']==int(kind) and r['seq']==sequence]
        if matching:
            if matching[-1]['result']:raise RuntimeError(matching[-1])
            return
    raise RuntimeError('No matching controller result; command will not be renewed or retried')

def raw():
    now=time.monotonic()
    assert len(measured)==6 and all(now-stamp<.15 for _,stamp in measured.values())
    return [measured[i][0] for i in range(1,7)]
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def main():
    global s,parser,session,excursion,gripper_relief,phase_four,single_step,expected_identity
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('excursion',type=int,nargs='?',default=0)
    args.add_argument('--execute',action='store_true',help='explicit finite actuator test; not a read-only operation')
    args.add_argument('--single-step',action='store_true')
    args.add_argument('--phase-four',action='store_true')
    args.add_argument('--gripper-relief',action='store_true')
    options=args.parse_args()
    if options.execute and json.loads((ROOT/'config/controller-profile.json').read_text()).get('manual_reference_version')==1:
        args.error('CommandOnly: используйте explorer arm calibrate / jog; старый recovery не применяется')
    if not 0<=options.excursion<=20:args.error('excursion must be 0..20')
    saved=json.loads((ROOT/'data/controller-state.json').read_text())
    if not options.execute:
        print(json.dumps(dict(read_only=True,uart_opened=False,commands_sent=False,
            file_age_seconds=time.time()-(ROOT/'data/controller-state.json').stat().st_mtime,
            state=saved),ensure_ascii=False))
        return
    if (not saved.get('telemetry_fresh') or
        not 0<=time.monotonic_ns()-saved.get('monotonic_ns',0)<=2_000_000_000):
        raise ValueError('Fresh saved board state is required before exclusive diagnostic ownership')
    cal=approved_calibration(ROOT,saved)
    bounds=checked_bounds(cal,saved.get('arm',{}).get('joints',[]))
    stationary_status(json.loads((ROOT/'data/status.json').read_text()),time.time())
    expected_identity=saved['identity']
    excursion=options.excursion;gripper_relief=options.gripper_relief
    phase_four=options.phase_four;single_step=options.single_step
    import serial
    s=serial.Serial(port=None,baudrate=2000000,timeout=.001,exclusive=True)
    s.dtr=s.rts=False;s.port='/dev/explorer_mcu';s.open()
    parser=Parser();session=Session()
    try:
        sync();poll(.15);sync()
        if state.get('mode')==2:session.state='fault';cmd(Kind.CLEAR)
        session.state='disarmed';cmd(Kind.OPEN)
        initial=raw()
        # Validate the live readback against the unchanged, accepted intervals.
        checked_bounds(cal,[dict(joint=i+1,raw_ticks=value,fresh=True,raw_valid=True,
            position_valid=True,error=0,device_error=0) for i,value in enumerate(initial)])
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
        while time.monotonic()-start<3.0:
            now=time.monotonic()
            if now-lastsync>.4:sync();lastsync=now
            actual=raw();progress=min(1.,(now-start)/.35)
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
            # ACK only means acceptance. Later position samples establish fresh
            # feedback; they do not prove target-register write or physical arrival.
            accepted_at=results[-1]['at']
            feedback_deadline=time.monotonic()+.14
            while min(stamp for _,stamp in measured.values()) <= accepted_at:
                poll(.005)
                if time.monotonic()>=feedback_deadline:
                    raise RuntimeError('No complete fresh six-servo readback after target')
            records.append(dict(elapsed=now-start,raw=actual))
            if single_step:
                issued=next(e for e in reversed(events) if e['event']=='command' and e['op']==int(Kind.ARM_RECOVER))
                # Observe one target without refreshing it or extending its lease.
                observation_end=issued['source_ns']/1e9+.160
                poll(max(0.,observation_end-time.monotonic()))
                records.append(dict(elapsed=time.monotonic()-start,raw=raw(),
                    arm_enabled=state.get('arm_enabled'),single_target_dwell=True))
                break
        cmd(Kind.ARM_CANCEL);poll(.15)
        print(json.dumps(dict(initial=initial,final=raw(),bounds=bounds,results=results[-5:])),flush=True)
    finally:
        s.write(encode(Kind.ESTOP));s.close()
        path=Path('/home/vlad/Explorer/data/controller-acceptance-20260929')/('arm-diagnostic-'+str(time.time_ns())+'.json')
        path.write_text(json.dumps(dict(excursion=excursion,gripper_relief=gripper_relief,
            phase_four=phase_four,single_step=single_step,records=records,results=results,events=events),indent=2))
        print(str(path),flush=True)

if __name__=="__main__":
    main()
