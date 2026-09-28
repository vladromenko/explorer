"""Linux input, bounded arm steps and gated holonomic drive requests.

The USB receiver is NOT a radio-link detector. Every arm decision requires a
new stick deflection after neutral, L1 held, and a live visible panel lease.
"""
import json
from holonomic_drive import velocity,blocked_by
import threading
import time
import yaml
from contextlib import closing

class GamepadPanel:
    def __init__(self,root,teaching,stop,drive=None,release=None):
        self.config=yaml.safe_load((root/'config/gamepad.yaml').read_text())
        self.teaching=teaching;self.stop=stop;self.root=root;self.drive=drive
        self.release=release or stop
        self.drive_active=False;self.drive_reasons=[];self.drive_vector=[0.,0.,0.]
        self.lock=threading.Lock();self.connected=False;self.keys=set();self.axes={}
        self.mode='DISARMED';self.joint=1;self.lease=0.;self.neutral=False
        self.last_event=0.;self.error=None;self.sequence=0;self.combo_at=None;self.proposal=None
        threading.Thread(target=self.run,daemon=True).start()

    def status(self):
        with self.lock:
            b=self.config['buttons']
            return dict(connected=self.connected,mode=self.mode,joint=self.joint,
                buttons=[k for k,v in b.items() if v in self.keys],axes=self.axes.copy(),
                last_event_age_s=time.monotonic()-self.last_event if self.last_event else None,
                sequence=self.sequence,error=self.error,proposal=self.proposal,radio_link_verified=False,
                bounded_steps_enabled=self.config.get('bounded_arm_steps_verified',False),
                drive_blocked_by=self.drive_reasons,drive_velocity=self.drive_vector,
                continuous_motion_enabled=bool(self.config.get('continuous_motion_enabled') and self.config.get('radio_loss_verified')),precision_step_deg=self.config['arm_step_deg'])

    def heartbeat(self,enabled):
        with self.lock:
            self.lease=time.monotonic()+.25 if enabled is True else 0.
            if not enabled:
                if self.drive_active:self.release();self.drive_active=False
                self.mode='DISARMED';self.neutral=False
        return self.status()

    def decide(self,code,value,kind,now):
        b=self.config['buttons'];decision=None
        if kind==1:
            if value==1:self.keys.add(code)
            elif value==0:self.keys.discard(code)
            if value==0 and code==b['l1'] and self.mode=='DRIVE':self.release()
            if value==1 and code==b['a'] and b['l1'] not in self.keys:
                if self.mode=='DRIVE':self.release();self.drive_active=False
                self.mode='ARM' if self.mode!='ARM' and now<self.lease else 'DISARMED'
                self.neutral=False
            if value==1 and code==b['x'] and b['l1'] not in self.keys:
                centered=not any(abs(v)>1e-6 for v in velocity(self.axes,self.config))
                self.mode='DRIVE' if centered and now<self.lease else 'DISARMED'
                self.neutral=False
            if value==1 and code==b['b']:
                self.mode='DISARMED';self.lease=0;self.stop()
        elif kind==3:
            self.axes[str(code)]=value
            if code==16 and value and b['l1'] not in self.keys:
                self.joint=min(6,max(1,self.joint+value))
                self.neutral=False
            if code==self.config['axes']['right_y']['code']:
                axis=self.config['axes']['right_y'];offset=(value-axis['center'])/127
                if abs(offset)<self.config['deadzone']:self.neutral=True
                elif abs(offset)>.65:
                    fresh=self.neutral;self.neutral=False
                    if fresh and self.mode=='ARM' and now<self.lease and b['l1'] in self.keys:
                        decision=(self.joint,self.config['arm_step_deg']*(-1 if offset>0 else 1))
        return decision

    def perform(self,joint,delta,deadline=None):
        self.proposal=dict(joint=joint,delta=delta,at=time.time(),executed=False,
                           blocked_by='Radio loss and servo cancellation are not verified')
        if self.config.get('bounded_arm_steps_verified',False) and deadline is not None:
            try:
                self.teaching.jog(joint,delta,True,deadline)
                self.proposal.update(executed=True,blocked_by=None,attainment_measured=False)
            except (OSError,ValueError) as exc:self.error=str(exc)

    def drive_tick(self,now):
        """All requests still pass the independent core sensor/obstacle gate."""
        held=self.config['buttons']['l1'] in self.keys
        if self.mode=='DRIVE' and now<self.lease and held:
            state=json.loads((self.root/'data/status.json').read_text())
            self.drive_reasons=blocked_by(state,self.config,time.time())
            self.drive_vector=velocity(self.axes,self.config)
            if not self.drive_reasons and self.drive is not None:
                self.drive(self.drive_vector);self.drive_active=True
            elif self.drive_active:
                self.release();self.drive_active=False
        else:
            self.drive_vector=[0.,0.,0.]
            if self.drive_active:self.release();self.drive_active=False

    def run(self):
        import select
        from evdev import InputDevice
        while True:
            try:
                with closing(InputDevice(self.config['device']['path'])) as device:
                    if device.info.vendor!=self.config['device']['vendor_id'] or device.info.product!=self.config['device']['product_id']:
                        raise ValueError('Подключён другой геймпад')
                    with self.lock:
                        self.connected=True;self.mode='DISARMED';self.keys.clear();self.neutral=False;self.error=None
                        self.axes={str(a['code']):device.absinfo(a['code']).value for a in self.config['axes'].values()}
                    pending=None
                    while True:
                        ready,_,_=select.select([device.fd],[],[],.05)
                        now=time.monotonic()
                        if ready:
                            for event in device.read():
                                with self.lock:
                                    self.last_event=now;self.sequence+=1
                                    decision=self.decide(event.code,event.value,event.type,now)
                                    if decision:pending=decision
                                    packet_done=event.type==0 and event.code==0
                                    permit=self.mode=='ARM' and now<self.lease and self.config['buttons']['l1'] in self.keys
                                if packet_done:
                                    if pending and permit and not self.teaching.lock.locked():
                                        threading.Thread(target=self.perform,args=(*pending,now+.1),daemon=True).start()
                                    pending=None
                        with self.lock:
                            combo=all(self.config['buttons'][k] in self.keys for k in self.config['estop_buttons'])
                            self.combo_at=(self.combo_at or now) if combo else None
                            if self.combo_at and now-self.combo_at>=self.config['estop_hold_s']:
                                self.stop();self.mode='DISARMED';self.lease=0;self.combo_at=now
                            if now>=self.lease:
                                if self.drive_active:self.release();self.drive_active=False
                                self.mode='DISARMED';self.neutral=False
                            self.drive_tick(now)
            except (OSError,ValueError,KeyError) as exc:
                with self.lock:
                    if self.drive_active:self.release();self.drive_active=False
                    self.connected=False;self.mode='DISARMED';self.keys.clear();self.neutral=False;self.error=str(exc)
                time.sleep(2)
