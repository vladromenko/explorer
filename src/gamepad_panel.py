"""Evdev adapter for the shared manual teleoperation backend."""
import json
import threading
import time
import yaml
from contextlib import closing

from holonomic_drive import blocked_by, axis_value
from manual_teleop import ManualTeleop


class GamepadPanel:
    def __init__(self, root, teaching, stop, drive=None, release=None, teleop=None, resume=None, takeover=None):
        self.root = root
        self.config = yaml.safe_load((root/'config/gamepad.yaml').read_text())
        self.teaching = teaching
        self.stop = stop
        self.release = release or stop
        self.teleop = teleop or ManualTeleop(drive, self.release, stop, teaching, resume, takeover)
        self.teleop.state_path = root/'data/manual-teleop.json'
        self.lock = threading.RLock()
        self.connected = False
        self.keys = set()
        self.pending_arm_taps = set()
        self.axes = {}
        self.mode = 'DISARMED'
        self.arm_mode = 'cartesian'
        self.arm_neutral_required = False
        self.lease = 0.0
        self.last_event = 0.0
        self.error = None
        self.sequence = 0
        self.precision = False
        self.arm_speed = 'normal'
        self.proposal = None
        self.drive_active = False
        self.drive_vector = [0.0, 0.0, 0.0]
        threading.Thread(target=self.run, daemon=True).start()

    def bind_arm(self, arm, model):
        self.teleop.bind_arm(arm, model)

    def drive_readiness(self):
        try:
            return blocked_by(json.loads((self.root/'data/status.json').read_text()), self.config, time.time())
        except (OSError, ValueError, TypeError, AttributeError):
            return ['Нет достоверного состояния робота']

    def status(self):
        with self.lock:
            buttons = self.config['buttons']
            return dict(connected=self.connected, mode=self.mode, arm_mode=self.arm_mode,
                arm_neutral_required=self.arm_neutral_required,
                buttons=[name for name, code in buttons.items() if code in self.keys],
                axes=self.axes.copy(), last_event_age_s=time.monotonic()-self.last_event if self.last_event else None,
                sequence=self.sequence, error=self.error, proposal=self.proposal, drive_blocked_by=self.drive_readiness(),
                drive_velocity=self.teleop._drive_vector(self._inputs(), self.precision), precision=self.precision,
                drive_profile='precision' if self.precision else 'normal',
                arm_speed='precision' if self.precision else self.arm_speed, control_scheme='explorer-manual-v2',
                shared_backend=self.teleop.status())

    def heartbeat(self, enabled):
        with self.lock:
            self.lease = time.monotonic()+1.0 if enabled is True else 0.0
            if not enabled:
                self.mode = 'DISARMED'
                self.pending_arm_taps.clear()
                self.teleop.disconnect('gamepad')
        return self.status()

    def select(self, mode, drive_profile=None, arm_speed=None):
        if mode not in ('DRIVE', 'ARM', 'ARM_CARTESIAN', 'COORDINATED', 'TELEOP', 'DISARMED'):
            raise ValueError('Неизвестный режим джойстика')
        with self.lock:
            if mode != 'DISARMED' and not self.connected:
                raise ValueError('Геймпад не подключён')
            if mode != 'DISARMED' and time.monotonic() >= self.lease:
                raise ValueError('Сначала включите панель джойстика')
            if mode == 'DISARMED':
                self.mode = mode
                self.pending_arm_taps.clear()
                self.teleop.disconnect('gamepad')
                return self.status()
            if self._moving():
                raise ValueError('Отпустите стики и кнопки движения, затем повторите')
            self.mode = 'TELEOP'
            if arm_speed not in (None,'normal','fast','precision'):
                raise ValueError('Неизвестная скорость руки')
            self.precision = drive_profile == 'precision' or arm_speed == 'precision'
            self.arm_speed = 'fast' if arm_speed == 'fast' else 'normal'
            self.arm_mode = 'cartesian'
            self.arm_neutral_required = False
            self.pending_arm_taps.clear()
            self.teleop.select_source('gamepad', True, True)
            self.teleop.set_arm_mode('gamepad', self.arm_mode)
            self.teleop.set_arm_speed('gamepad', self.arm_speed)
            self.teleop.update('gamepad', {}, True, self.precision)
        return self.status()

    def _axis(self, name):
        item = self.config['axes'][name]
        value = self.axes.get(str(item['code']), item['center'])
        return axis_value(value, item, self.config['deadzone'])

    def set_arm_speed(self, speed):
        if speed not in ('normal','fast'):
            raise ValueError('Неизвестная скорость руки')
        with self.lock:
            if self.teleop.owner != 'gamepad':
                raise ValueError('Сначала выберите джойстик')
            self.arm_speed = speed
            self.teleop.set_arm_speed('gamepad', speed)
        return self.status()

    def _drive_axis(self, value):
        expo = float(self.config.get('drive_expo', 0.5))
        return (1-expo)*value+expo*value**3

    def _arm_controls_neutral(self):
        b = self.config['buttons']
        rx, ry = self._axis('right_x'), self._axis('right_y')
        dpx, dpy = float(self.axes.get('16', 0)), float(self.axes.get('17', 0))
        return max(abs(rx), abs(ry), abs(dpx), abs(dpy)) <= 0.08 and not ({b['a'], b['y']} & self.keys)

    def _inputs(self):
        b = self.config['buttons']
        dpx, dpy = float(self.axes.get('16', 0)), float(self.axes.get('17', 0))
        ly, lx = self._axis('left_y'), self._axis('left_x')
        rx, ry = self._axis('right_x'), self._axis('right_y')
        drive_y, drive_x = self._drive_axis(ly), self._drive_axis(lx)
        values = dict(forward=max(0.0, drive_y), backward=max(0.0, -drive_y),
            left=max(0.0, drive_x), right=max(0.0, -drive_x),
            turn_left=float(b['l1'] in self.keys), turn_right=float(b['r1'] in self.keys),
            grip_open=float(b['l2'] in self.keys), grip_close=float(b['r2'] in self.keys))
        if self.arm_neutral_required:
            return values
        if self.arm_mode == 'cartesian':
            values.update(arm_x_forward=max(0.0, ry), arm_x_back=max(0.0, -ry),
                arm_y_left=max(0.0, rx), arm_y_right=max(0.0, -rx),
                arm_z_up=float(b['y'] in self.keys), arm_z_down=float(b['a'] in self.keys))
        else:
            values.update(joint1_decrease=max(0.0, rx), joint1_increase=max(0.0, -rx),
                joint2_increase=max(0.0, ry), joint2_decrease=max(0.0, -ry),
                joint3_increase=float(b['y'] in self.keys), joint3_decrease=float(b['a'] in self.keys),
                pitch_up=max(0.0, -dpy), pitch_down=max(0.0, dpy),
                wrist_left=max(0.0, -dpx), wrist_right=max(0.0, dpx))
        return values

    def _moving(self):
        return self.teleop._moving(self._inputs())

    def decide(self, code, value, kind, now):
        b = self.config['buttons']
        if kind == 1:
            if value == 1:
                self.keys.add(code)
                if code in (b['y'], b['a']):
                    self.pending_arm_taps.add(code)
            elif value == 0:
                self.keys.discard(code)
            if value == 1 and code == b['x']:
                self.precision = not self.precision
            if value == 1 and code == b['right_stick']:
                self.arm_speed = 'normal' if self.arm_speed == 'fast' else 'fast'
                if self.teleop.owner == 'gamepad':
                    self.teleop.set_arm_speed('gamepad', self.arm_speed)
            if value == 1 and code == b['select']:
                self.arm_mode = 'joint' if self.arm_mode == 'cartesian' else 'cartesian'
                self.arm_neutral_required = not self._arm_controls_neutral()
                if self.teleop.owner == 'gamepad':
                    self.teleop.set_arm_mode('gamepad', self.arm_mode)
            if value == 1 and code == b['b']:
                self.mode = 'DISARMED'
                self.pending_arm_taps.clear()
                self.teleop.stop()
            if value == 1 and code == b['start'] and b['b'] not in self.keys:
                if self.teleop.owner != 'gamepad':
                    if self._moving():
                        raise ValueError('Сначала отпустите все органы движения')
                    self.teleop.select_source('gamepad', True, True)
                    self.teleop.set_arm_mode('gamepad', self.arm_mode)
                    self.teleop.set_arm_speed('gamepad', self.arm_speed)
                self.teleop.update('gamepad', self._inputs(), True, self.precision)
                self.teleop.resume('gamepad', True)
                self.mode = 'TELEOP'
        elif kind == 3:
            self.axes[str(code)] = value
        return None

    def feed(self, now):
        with self.lock:
            fresh = bool(self.last_event and now-self.last_event <= float(self.config.get('watchdog_timeout', 0.25)))
            if self.arm_neutral_required and self._arm_controls_neutral():
                self.arm_neutral_required = False
            if self.mode == 'TELEOP' and now < self.lease and self.teleop.owner == 'gamepad':
                values = self._inputs() if fresh else {}
                if fresh and not self.arm_neutral_required:
                    for code in self.pending_arm_taps - self.keys:
                        action = ('arm_z_up' if code == self.config['buttons']['y'] else 'arm_z_down') if self.arm_mode == 'cartesian' else ('joint3_increase' if code == self.config['buttons']['y'] else 'joint3_decrease')
                        values[action] = 1.0
                self.pending_arm_taps.clear()
                self.teleop.update('gamepad', values, True, self.precision)
            elif self.mode != 'DISARMED':
                self.mode = 'DISARMED'
                self.pending_arm_taps.clear()
                self.teleop.disconnect('gamepad')

    def drive_tick(self, now):
        self.feed(now)

    def perform(self, *_args, **_kwargs):
        raise ValueError('Discrete gamepad arm mode replaced by shared proportional teleop')

    def poll_device(self, device, now):
        keys = set(device.active_keys())
        codes = {item['code'] for item in self.config['axes'].values()} | {16, 17}
        axes = {str(code): device.absinfo(code).value for code in codes}
        with self.lock:
            self.keys = keys
            self.axes = axes
            self.last_event = now

    def run(self):
        import select
        from evdev import InputDevice
        while True:
            try:
                with closing(InputDevice(self.config['device']['path'])) as device:
                    if device.info.vendor != self.config['device']['vendor_id'] or device.info.product != self.config['device']['product_id']:
                        raise ValueError('Подключён другой геймпад')
                    with self.lock:
                        self.connected = True
                        self.mode = 'DISARMED'
                        self.keys.clear()
                        self.pending_arm_taps.clear()
                        self.error = None
                        self.axes = {str(axis['code']): device.absinfo(axis['code']).value for axis in self.config['axes'].values()}
                    while True:
                        ready, _, _ = select.select([device.fd], [], [], 0.05)
                        now = time.monotonic()
                        if ready:
                            for event in device.read():
                                with self.lock:
                                    self.last_event = now
                                    self.sequence += 1
                                    self.decide(event.code, event.value, event.type, now)
                        self.poll_device(device, now)
                        self.feed(now)
            except (OSError, ValueError, KeyError) as exc:
                with self.lock:
                    self.teleop.disconnect('gamepad')
                    self.connected = False
                    self.mode = 'DISARMED'
                    self.keys.clear()
                    self.pending_arm_taps.clear()
                    self.error = str(exc)
                time.sleep(2)
