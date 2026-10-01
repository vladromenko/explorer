"""Evdev adapter for the shared manual teleoperation backend."""
import json
import threading
import time
import yaml
from contextlib import closing
from holonomic_drive import blocked_by,axis_value
from manual_teleop import ManualTeleop


class GamepadPanel:
    def __init__(self,root,teaching,stop,drive=None,release=None,teleop=None,resume=None,takeover=None):
        self.root=root;self.config=yaml.safe_load((root/'config/gamepad.yaml').read_text())
        self.teaching=teaching;self.stop=stop;self.release=release or stop
        self.teleop=teleop or ManualTeleop(drive,self.release,stop,teaching,resume,takeover)
        self.lock=threading.RLock();self.connected=False;self.keys=set();self.axes={}
        self.mode='DISARMED';self.lease=0.;self.last_event=0.;self.error=None;self.sequence=0
        self.precision=False;self.proposal=None;self.drive_active=False;self.drive_vector=[0.,0.,0.]
        threading.Thread(target=self.run,daemon=True).start()

    def bind_arm(self,arm,model):self.teleop.bind_arm(arm,model)

    def drive_readiness(self):
        try:return blocked_by(json.loads((self.root/'data/status.json').read_text()),self.config,time.time())
        except (OSError,ValueError,TypeError,AttributeError):return ['Нет достоверного состояния робота']

    def status(self):
        with self.lock:
            buttons=self.config['buttons']
            return dict(connected=self.connected,mode=self.mode,buttons=[k for k,v in buttons.items() if v in self.keys],
                axes=self.axes.copy(),last_event_age_s=time.monotonic()-self.last_event if self.last_event else None,
                sequence=self.sequence,error=self.error,proposal=self.proposal,drive_blocked_by=self.drive_readiness(),
                drive_velocity=self.teleop._drive_vector(self._inputs(),self.precision),precision=self.precision,
                drive_profile='precision' if self.precision else 'normal',arm_speed='precision' if self.precision else 'normal',
                control_scheme='left stick chassis; L1/R1 rotate; right stick+D-pad XYZ; Y/A pitch; D-pad left/right wrist; L2/R2 gripper',
                shared_backend=self.teleop.status())

    def heartbeat(self,enabled):
        with self.lock:
            self.lease=time.monotonic()+1.0 if enabled is True else 0.
            if not enabled:self.mode='DISARMED';self.teleop.disconnect('gamepad')
        return self.status()

    def select(self,mode,drive_profile=None,arm_speed=None):
        if mode not in ('DRIVE','ARM','ARM_CARTESIAN','COORDINATED','TELEOP','DISARMED'):raise ValueError('Неизвестный режим джойстика')
        with self.lock:
            if mode!='DISARMED' and not self.connected:raise ValueError('Геймпад не подключён')
            if mode!='DISARMED' and time.monotonic()>=self.lease:raise ValueError('Сначала включите панель джойстика')
            if mode=='DISARMED':self.mode=mode;self.teleop.disconnect('gamepad');return self.status()
            if self._moving():raise ValueError('Отпустите стики и кнопки движения, затем повторите')
            self.mode='TELEOP';self.precision=drive_profile=='precision' or arm_speed=='precision'
            self.teleop.update('gamepad',{},True,self.precision)
        return self.status()

    def _axis(self,name):
        item=self.config['axes'][name];value=self.axes.get(str(item['code']),item['center'])
        return axis_value(value,item,self.config['deadzone'])

    def _inputs(self):
        b=self.config['buttons'];dpx=float(self.axes.get('16',0));dpy=float(self.axes.get('17',0))
        ly=self._axis('left_y');lx=self._axis('left_x');rx=self._axis('right_x');ry=self._axis('right_y')
        return dict(forward=max(0.,ly),backward=max(0.,-ly),left=max(0.,lx),right=max(0.,-lx),
            turn_left=float(b['l1'] in self.keys),turn_right=float(b['r1'] in self.keys),
            arm_x_forward=max(0.,-dpy),arm_x_back=max(0.,dpy),arm_y_left=max(0.,rx),arm_y_right=max(0.,-rx),
            arm_z_up=max(0.,ry),arm_z_down=max(0.,-ry),wrist_left=max(0.,-dpx),wrist_right=max(0.,dpx),
            pitch_up=float(b['y'] in self.keys),pitch_down=float(b['a'] in self.keys),
            grip_open=float(b['l2'] in self.keys),grip_close=float(b['r2'] in self.keys))

    def _moving(self):return self.teleop._moving(self._inputs())

    def decide(self,code,value,kind,now):
        b=self.config['buttons']
        if kind==1:
            if value==1:self.keys.add(code)
            elif value==0:self.keys.discard(code)
            if value==1 and code==b['x']:self.precision=not self.precision
            if value==1 and code==b['b']:
                self.mode='DISARMED';self.teleop.stop()
            if value==1 and code==b['start']:
                self.teleop.update('gamepad',self._inputs(),True,self.precision)
                self.teleop.resume('gamepad',True);self.mode='TELEOP'
        elif kind==3:self.axes[str(code)]=value
        return None

    def feed(self,now):
        with self.lock:
            fresh=self.last_event and now-self.last_event<=float(self.config.get('watchdog_timeout',.25))
            if self.mode=='TELEOP' and now<self.lease:
                self.teleop.update('gamepad',self._inputs() if fresh else {},True,self.precision)
            elif self.mode!='DISARMED':self.mode='DISARMED';self.teleop.disconnect('gamepad')

    def drive_tick(self,now):self.feed(now)
    def perform(self,*_args,**_kwargs):raise ValueError('Discrete gamepad arm mode replaced by shared Cartesian teleop')

    def run(self):
        import select
        from evdev import InputDevice
        while True:
            try:
                with closing(InputDevice(self.config['device']['path'])) as device:
                    if device.info.vendor!=self.config['device']['vendor_id'] or device.info.product!=self.config['device']['product_id']:
                        raise ValueError('Подключён другой геймпад')
                    with self.lock:
                        self.connected=True;self.mode='DISARMED';self.keys.clear();self.error=None
                        self.axes={str(a['code']):device.absinfo(a['code']).value for a in self.config['axes'].values()}
                    while True:
                        ready,_,_=select.select([device.fd],[],[],.05);now=time.monotonic()
                        if ready:
                            for event in device.read():
                                with self.lock:self.last_event=now;self.sequence+=1;self.decide(event.code,event.value,event.type,now)
                        self.feed(now)
            except (OSError,ValueError,KeyError) as exc:
                with self.lock:
                    self.teleop.disconnect('gamepad');self.connected=False;self.mode='DISARMED';self.keys.clear();self.error=str(exc)
                time.sleep(2)
