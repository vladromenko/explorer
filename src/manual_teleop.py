"""Single-owner manual teleoperation shared by browser keyboard and evdev."""
import math
import threading
import time


MOTION_KEYS=frozenset(('w','s','a','d','q','e','arrowup','arrowdown','arrowleft','arrowright',
                       'r','f','t','g','z','c','v','b','y','h','u','j','i','k'))

KEY_ACTIONS={'w':'forward','s':'backward','a':'left','d':'right','q':'turn_left','e':'turn_right',
             'r':'arm_x_forward','f':'arm_x_back','arrowleft':'arm_y_left','arrowright':'arm_y_right',
             'arrowup':'arm_z_up','arrowdown':'arm_z_down','t':'pitch_up','g':'pitch_down',
             'z':'wrist_left','c':'wrist_right','v':'grip_open','b':'grip_close',
             'y':'joint1_increase','h':'joint1_decrease','u':'joint2_increase','j':'joint2_decrease',
             'i':'joint3_increase','k':'joint3_decrease'}

def keyboard_inputs(keys):
    clean={str(k).lower() for k in keys}
    if not clean<=MOTION_KEYS|{'shift'}:raise ValueError('Неизвестная клавиша teleop')
    return {action:1. for key,action in KEY_ACTIONS.items() if key in clean}


class ManualTeleop:
    def __init__(self,drive,release,stop,teaching,resume_callback=None,takeover_callback=None):
        self.drive=drive;self.release=release;self.stop_all=stop;self.teaching=teaching
        self.lock=threading.RLock();self.owner=None;self.lease=0.;self.inputs={};self.precision=False
        self.stop_latched=True;self.neutral_seen=False;self.drive_active=False;self.arm_busy=False
        self.error=None;self.arm=None;self.model=None;self.last_arm=0.;self.closed=False;self.resume_callback=resume_callback
        self.takeover_callback=takeover_callback;self.takeover_active=False
        threading.Thread(target=self._run,daemon=True).start()

    def bind_arm(self,arm,model):
        self.arm=arm;self.model=model
        arm.gamepad_permit=lambda:self._arm_permitted()

    def _arm_permitted(self):
        with self.lock:return not self.stop_latched and self.owner is not None and time.monotonic()<self.lease

    def claim(self,source,observing):
        if source not in ('keyboard','gamepad'):raise ValueError('Неизвестный источник teleop')
        if observing is not True:raise ValueError('Подтвердите наблюдение за роботом')
        with self.lock:
            if self.owner not in (None,source):self._release_locked()
            self.owner=source;self.lease=time.monotonic()+.45
        return self.status()

    def update(self,source,inputs,observing=True,precision=False):
        self.claim(source,observing)
        clean={str(k).lower():float(v) for k,v in inputs.items() if math.isfinite(float(v)) and abs(float(v))<=1.0001}
        moving=self._moving(clean);announce=moving and not self.takeover_active
        self.takeover_active=moving
        if announce and self.takeover_callback:self.takeover_callback()
        with self.lock:
            self.inputs=clean;self.precision=bool(precision);self.lease=time.monotonic()+.45
            if self.stop_latched and not moving:self.neutral_seen=True
        return self.status()

    @staticmethod
    def _moving(values):return any(abs(v)>.08 for v in values.values())

    def stop(self):
        with self.lock:
            self.stop_latched=True;self.neutral_seen=not self._moving(self.inputs);self._release_locked()
            if self.arm:self.arm.stop()
        self.stop_all();return self.status()

    def resume(self,source,observing):
        self.claim(source,observing)
        with self.lock:
            if self._moving(self.inputs):raise ValueError('Сначала отпустите все органы движения')
            if not self.neutral_seen:raise ValueError('После STOP требуется подтверждённая нейтраль')
            if self.resume_callback:self.resume_callback()
            self.stop_latched=False;self.error=None
        return self.status()

    def disconnect(self,source):
        with self.lock:
            if self.owner==source:
                self.inputs={};self.owner=None;self.lease=0.;self._release_locked()
                if self.arm:self.arm.stop()
        return self.status()

    def _release_locked(self):
        if self.drive_active:self.release();self.drive_active=False

    def status(self):
        with self.lock:return dict(owner=self.owner,stop_latched=self.stop_latched,neutral_seen=self.neutral_seen,
            lease_age_s=max(0.,self.lease-time.monotonic()),precision=self.precision,inputs=dict(self.inputs),
            drive_active=self.drive_active,arm_busy=self.arm_busy,error=self.error,
            backend='shared_manual_teleop',measured_joint_feedback=False)

    def _drive_vector(self,v,precision):
        scale=.1 if precision else 1.
        x=(v.get('forward',0)-v.get('backward',0))*1.20*scale
        y=(v.get('left',0)-v.get('right',0))*1.08*scale
        yaw=(v.get('turn_left',0)-v.get('turn_right',0))*2.50*scale
        norm=math.hypot(x/1.20,y/1.08)
        if norm>1:x/=norm;y/=norm
        return [x,y,yaw]

    def _arm_intent(self,v,precision):
        xyz=[v.get('arm_x_forward',0)-v.get('arm_x_back',0),
             v.get('arm_y_left',0)-v.get('arm_y_right',0),
             v.get('arm_z_up',0)-v.get('arm_z_down',0)]
        joints=[v.get('joint1_increase',0)-v.get('joint1_decrease',0),
                v.get('joint2_increase',0)-v.get('joint2_decrease',0),
                v.get('joint3_increase',0)-v.get('joint3_decrease',0),
                v.get('pitch_up',0)-v.get('pitch_down',0),
                v.get('wrist_left',0)-v.get('wrist_right',0),
                v.get('grip_close',0)-v.get('grip_open',0)]
        factor=.45 if precision else 1.
        return [x*factor for x in xyz],[x*factor for x in joints]

    def _arm_segment(self,xyz,joints,deadline,precision):
        try:
            self.teaching.teleop(self.model(),xyz,joints,True,deadline,precision)
            self.error=None
        except (OSError,ValueError) as exc:self.error=str(exc)
        finally:
            with self.lock:self.arm_busy=False;self.last_arm=time.monotonic()

    def tick(self,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if self.owner is None:return
            if now>=self.lease:
                self.inputs={};self.owner=None;self._release_locked()
                if self.arm:self.arm.stop()
                return
            values=dict(self.inputs);precision=self.precision
            if self.stop_latched:
                self._release_locked();return
            drive=self._drive_vector(values,precision)
            if any(abs(x)>1e-6 for x in drive):self.drive(drive);self.drive_active=True
            else:self._release_locked()
            xyz,joints=self._arm_intent(values,precision)
            magnitude=max([abs(x) for x in xyz+joints],default=0.)
            interval=.09+(.18*(1-magnitude)) if magnitude>.08 else 999
            if magnitude>.08 and not self.arm_busy and now-self.last_arm>=interval and self.arm and self.model:
                self.arm_busy=True
                threading.Thread(target=self._arm_segment,args=(xyz,joints,now+.55,precision),daemon=True).start()

    def _run(self):
        while not self.closed:
            try:self.tick()
            except Exception as exc:self.error=str(exc)
            time.sleep(.05)
