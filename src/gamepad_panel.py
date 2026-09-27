"""Linux input diagnostics and bounded operator decisions; never continuous motion.

The USB receiver is NOT a radio-link detector. Every arm decision requires a
new stick deflection after neutral, L1 held, and a live visible panel lease.
"""
import threading
import time
import yaml
from contextlib import closing

class GamepadPanel:
    def __init__(self,root,teaching,stop):
        self.config=yaml.safe_load((root/'config/gamepad.yaml').read_text())
        self.teaching=teaching;self.stop=stop
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
                continuous_motion_enabled=False,precision_step_deg=self.config['arm_step_deg'])

    def heartbeat(self,enabled):
        with self.lock:
            self.lease=time.monotonic()+.25 if enabled is True else 0.
            if not enabled:self.mode='DISARMED';self.neutral=False
        return self.status()

    def decide(self,code,value,kind,now):
        b=self.config['buttons'];decision=None
        if kind==1:
            if value==1:self.keys.add(code)
            elif value==0:self.keys.discard(code)
            if value==1 and code==b['a'] and b['l1'] not in self.keys:
                self.mode='ARM' if self.mode!='ARM' and now<self.lease else 'DISARMED'
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

    def perform(self,joint,delta):
        # A wireless-off test and a servo cancellation mechanism are outstanding.
        # Until then the stick selects a proposal; it cannot publish arm commands.
        self.proposal=dict(joint=joint,delta=delta,at=time.time(),executed=False,
                           blocked_by='Radio loss and servo cancellation are not verified')

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
                    while True:
                        ready,_,_=select.select([device.fd],[],[],.05)
                        now=time.monotonic()
                        if ready:
                            for event in device.read():
                                with self.lock:
                                    self.last_event=now;self.sequence+=1
                                    decision=self.decide(event.code,event.value,event.type,now)
                                if decision and not self.teaching.lock.locked():
                                    threading.Thread(target=self.perform,args=decision,daemon=True).start()
                        with self.lock:
                            combo=all(self.config['buttons'][k] in self.keys for k in self.config['estop_buttons'])
                            self.combo_at=(self.combo_at or now) if combo else None
                            if self.combo_at and now-self.combo_at>=self.config['estop_hold_s']:
                                self.stop();self.mode='DISARMED';self.lease=0;self.combo_at=now
                            if now>=self.lease:self.mode='DISARMED';self.neutral=False
            except (OSError,ValueError) as exc:
                with self.lock:
                    self.connected=False;self.mode='DISARMED';self.keys.clear();self.neutral=False;self.error=str(exc)
                self.stop();time.sleep(2)
