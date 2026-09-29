"""Linux input, bounded arm steps and gated holonomic drive requests.

The USB receiver is NOT a radio-link detector. Every arm decision requires a
new stick deflection after neutral, L1 held, and a live visible panel lease.
"""
import json
from holonomic_drive import velocity,blocked_by,axis_value
import threading
import time
import yaml
from contextlib import closing

class GamepadPanel:
    def __init__(self,root,teaching,stop,drive=None,release=None):
        self.config=yaml.safe_load((root/'config/gamepad.yaml').read_text())
        self.teaching=teaching;self.stop=stop;self.root=root;self.drive=drive
        self.arm_jog=None;self.arm_cartesian=None;self.arm_cancel=lambda:None
        self.release=release or stop
        self.drive_active=False;self.drive_reasons=[];self.drive_vector=[0.,0.,0.]
        self.lock=threading.Lock();self.connected=False;self.keys=set();self.axes={}
        self.mode='DISARMED';self.joint=1;self.lease=0.;self.neutral=False
        self.last_event=0.;self.error=None;self.sequence=0;self.combo_at=None;self.proposal=None
        threading.Thread(target=self.run,daemon=True).start()

    def bind_arm(self,arm,model):
        """Bind cancellation for both factory and native arm executors."""
        arm.gamepad_permit=lambda:self.mode in ('ARM','ARM_CARTESIAN') and time.monotonic()<self.lease and self.config['buttons']['l1'] in self.keys
        def jog(joint,delta,deadline):
            if not arm.gamepad_permit():raise ValueError('Выберите A, удерживайте L1 и держите панель открытой')
            return self.teaching.jog(joint,delta,True,deadline)
        self.arm_jog=jog
        self.arm_cartesian=lambda axis,direction,deadline:self.teaching.cartesian(model(),axis,direction,True,deadline)
        self.arm_cancel=arm.stop
        if getattr(arm,'profile',{}).get('manual_reference_version')==1:self.config['arm_step_deg']=3

    def drive_readiness(self):
        try:
            state=json.loads((self.root/'data/status.json').read_text())
            return blocked_by(state,self.config,time.time())
        except (OSError,ValueError,TypeError,AttributeError):
            return ['Нет достоверного состояния робота']

    def status(self):
        with self.lock:
            b=self.config['buttons']
            return dict(connected=self.connected,mode=self.mode,joint=self.joint,
                buttons=[k for k,v in b.items() if v in self.keys],axes=self.axes.copy(),
                last_event_age_s=time.monotonic()-self.last_event if self.last_event else None,
                sequence=self.sequence,error=self.error,proposal=self.proposal,radio_link_verified=self.config.get('radio_loss_verified') is True,
                bounded_steps_enabled=self.config.get('bounded_arm_steps_verified',False),
                drive_blocked_by=self.drive_readiness(),drive_velocity=self.drive_vector,
                operator_chassis_accepted=self.config.get('operator_chassis_accepted') is True,
                continuous_motion_enabled=bool(self.config.get('continuous_motion_enabled') and
                    (self.config.get('radio_loss_verified') or self.config.get('operator_chassis_accepted'))),precision_step_deg=self.config['arm_step_deg'])

    def heartbeat(self,enabled):
        with self.lock:
            # Browsers throttle sub-second timers in background tabs.  A one-second
            # visible-panel lease remains short, while surviving Wi-Fi/UI jitter.
            self.lease=time.monotonic()+1.0 if enabled is True else 0.
            if not enabled:
                if self.mode in ('ARM','ARM_CARTESIAN'):self.arm_cancel()
                if self.drive_active:self.release();self.drive_active=False
                self.mode='DISARMED';self.neutral=False
        return self.status()

    def select(self,mode):
        """Select an operator mode explicitly from the visible web panel."""
        if mode not in ('DRIVE','ARM','ARM_CARTESIAN','DISARMED'):
            raise ValueError('Неизвестный режим джойстика')
        with self.lock:
            if mode != 'DISARMED' and not self.connected:
                raise ValueError('Геймпад не подключён')
            if mode != 'DISARMED' and time.monotonic() >= self.lease:
                raise ValueError('Сначала включите панель джойстика')
            if self.mode in ('ARM','ARM_CARTESIAN') and self.mode != mode:
                self.arm_cancel()
            if self.mode == 'DRIVE' and mode != 'DRIVE':
                self.release(); self.drive_active=False
            if mode == 'DRIVE' and any(abs(v) > 1e-6 for v in velocity(self.axes,self.config)):
                raise ValueError('Отпустите стики в центр и повторите')
            self.mode=mode
            self.neutral=False
        return self.status()

    def decide(self,code,value,kind,now):
        b=self.config['buttons'];decision=None
        if kind==1:
            if value==1:self.keys.add(code)
            elif value==0:self.keys.discard(code)
            if value==0 and code==b['l1'] and self.mode=='DRIVE':self.release()
            if value==0 and code==b['l1'] and self.mode in ('ARM','ARM_CARTESIAN'):self.arm_cancel()
            if value==1 and code==b['a'] and b['l1'] not in self.keys:
                if self.mode in ('ARM','ARM_CARTESIAN'):self.arm_cancel()
                if self.mode=='DRIVE':self.release();self.drive_active=False
                self.mode='ARM' if self.mode!='ARM' and now<self.lease else 'DISARMED'
                self.neutral=False
            if value==1 and code==b['x'] and b['l1'] not in self.keys:
                if self.mode in ('ARM','ARM_CARTESIAN'):self.arm_cancel()
                centered=not any(abs(v)>1e-6 for v in velocity(self.axes,self.config))
                self.mode='DRIVE' if centered and now<self.lease else 'DISARMED'
                self.neutral=False
            if value==1 and code==b.get('y') and b['l1'] not in self.keys:
                if self.mode in ('ARM','ARM_CARTESIAN'):self.arm_cancel()
                if self.mode=='DRIVE':self.release();self.drive_active=False
                self.mode='ARM_CARTESIAN' if now<self.lease else 'DISARMED'
                self.joint=min(3,self.joint);self.neutral=False
            if value==1 and code==b['b']:
                self.mode='DISARMED';self.lease=0;self.stop()
            if (value==1 and self.mode=='ARM' and now<self.lease and b['l1'] in self.keys and
                    code in (b.get('l2'),b.get('r2'))):
                decision=(6,-self.config['gripper_step_deg'] if code==b.get('l2') else self.config['gripper_step_deg'])
        elif kind==3:
            self.axes[str(code)]=value
            if code==16 and value and b['l1'] not in self.keys:
                self.joint=min(3 if self.mode=='ARM_CARTESIAN' else 6,max(1,self.joint+value))
                self.neutral=False
            if code==self.config['axes']['right_y']['code']:
                axis=self.config['axes']['right_y'];offset=axis_value(value,axis,0.)
                if abs(offset)<self.config['deadzone']:self.neutral=True
                elif abs(offset)>.65:
                    fresh=self.neutral;self.neutral=False
                    if fresh and self.mode in ('ARM','ARM_CARTESIAN') and now<self.lease and b['l1'] in self.keys:
                        direction=1 if offset>0 else -1
                        decision=(('x','y','z')[self.joint-1],direction) if self.mode=='ARM_CARTESIAN' else (self.joint,self.config['arm_step_deg']*direction)
        return decision

    def perform(self,joint,delta,deadline=None):
        self.proposal=dict(joint=joint,delta=delta,at=time.time(),executed=False,
                           blocked_by='Radio loss and servo cancellation are not verified')
        if self.config.get('bounded_arm_steps_verified',False) and deadline is not None:
            try:
                self.error=None
                if isinstance(joint,str):
                    if self.arm_cartesian is None:raise ValueError('Cartesian control unavailable')
                    self.arm_cartesian(joint,delta,deadline)
                elif self.arm_jog is not None:self.arm_jog(joint,delta,deadline)
                else:self.teaching.jog(joint,delta,True,deadline)
                self.proposal.update(executed=True,blocked_by=None,attainment_measured=False)
            except (OSError,ValueError) as exc:
                self.error=str(exc);self.proposal.update(blocked_by=self.error)

    def drive_tick(self,now):
        """All requests still pass the independent core sensor/obstacle gate."""
        held=self.config['buttons']['l1'] in self.keys
        if self.mode=='DRIVE' and now<self.lease and held:
            self.drive_reasons=self.drive_readiness()
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
                                    permit=self.mode in ('ARM','ARM_CARTESIAN') and now<self.lease and self.config['buttons']['l1'] in self.keys
                                if packet_done:
                                    if pending and permit and not self.teaching.lock.locked():
                                        threading.Thread(target=self.perform,args=(*pending,now+.5),daemon=True).start()
                                    pending=None
                        with self.lock:
                            combo=all(self.config['buttons'][k] in self.keys for k in self.config['estop_buttons'])
                            self.combo_at=(self.combo_at or now) if combo else None
                            if self.combo_at and now-self.combo_at>=self.config['estop_hold_s']:
                                self.stop();self.mode='DISARMED';self.lease=0;self.combo_at=now
                            if now>=self.lease:
                                if self.mode in ('ARM','ARM_CARTESIAN'):self.arm_cancel()
                                if self.drive_active:self.release();self.drive_active=False
                                self.mode='DISARMED';self.neutral=False
                            self.drive_tick(now)
            except (OSError,ValueError,KeyError) as exc:
                with self.lock:
                    if self.mode in ('ARM','ARM_CARTESIAN'):self.arm_cancel()
                    if self.drive_active:self.release();self.drive_active=False
                    self.connected=False;self.mode='DISARMED';self.keys.clear();self.neutral=False;self.error=str(exc)
                time.sleep(2)
