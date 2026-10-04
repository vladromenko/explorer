"""Single-owner manual teleoperation shared by browser keyboard and evdev."""
import json
import math
import threading
import time

from manual_controls import ALL_ACTIONS, ARM_ACTIONS, ARM_MODES, SCHEME_ID, keyboard_inputs


class ManualTeleop:
    def __init__(self, drive, release, stop, teaching, resume_callback=None, takeover_callback=None):
        self.drive = drive
        self.release = release
        self.stop_all = stop
        self.teaching = teaching
        self.lock = threading.RLock()
        self.owner = None
        self.lease = 0.0
        self.inputs = {}
        self.precision = False
        self.arm_mode = "cartesian"
        self.generation = 0
        self.stop_latched = True
        self.neutral_seen = False
        self.drive_active = False
        self.arm_busy = False
        self.error = None
        self.arm = None
        self.model = None
        self.last_arm = 0.0
        self.last_integrator = time.monotonic()
        self.joint_residual = [0.0] * 6
        self.xyz_residual = [0.0] * 3
        self.closed = False
        self.resume_callback = resume_callback
        self.takeover_callback = takeover_callback
        self.takeover_active = False
        self.state_path = None
        threading.Thread(target=self._run, daemon=True).start()

    def _record_locked(self, event):
        if self.state_path is None:
            return
        record = dict(at=time.time(), event=event, input_source=self.owner, arm_mode=self.arm_mode,
            precision=self.precision, normalized_actions=dict(self.inputs), control_frame="base_footprint",
            control_scheme=SCHEME_ID, generation=self.generation, stop_latched=self.stop_latched,
            measured_joint_feedback=False)
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, allow_nan=False))
        temporary.replace(self.state_path)

    def bind_arm(self, arm, model):
        self.arm = arm
        self.model = model
        arm.gamepad_permit = lambda: self._arm_permitted()

    def _arm_permitted(self):
        with self.lock:
            return not self.stop_latched and self.owner is not None and time.monotonic() < self.lease

    def _reset_arm_locked(self):
        self.joint_residual = [0.0] * 6
        self.xyz_residual = [0.0] * 3
        self.last_integrator = time.monotonic()

    def select_source(self, source, observing, neutral=True):
        if source not in ("keyboard", "gamepad"):
            raise ValueError("Неизвестный источник teleop")
        if observing is not True:
            raise ValueError("Подтвердите наблюдение за роботом")
        if neutral is not True:
            raise ValueError("Сначала отпустите органы движения выбранного источника")
        with self.lock:
            changed = self.owner != source
            if changed:
                self._release_locked()
                if self.arm:
                    self.arm.stop()
                self.generation += 1
                self._reset_arm_locked()
            self.owner = source
            self.inputs = {}
            self.precision = False
            self.lease = time.monotonic() + 0.45
            if self.stop_latched:
                self.neutral_seen = True
            self._record_locked("source_selected")
        return self.status()

    def claim(self, source, observing):
        return self.select_source(source, observing, True)

    def set_arm_mode(self, source, mode):
        if mode not in ARM_MODES:
            raise ValueError("Неизвестный режим руки")
        with self.lock:
            if self.owner != source:
                raise ValueError("Сначала явно выберите источник управления")
            if self.arm_mode != mode:
                self.arm_mode = mode
                self.inputs = {key: value for key, value in self.inputs.items() if key not in ARM_ACTIONS}
                self.generation += 1
                self._reset_arm_locked()
                if self.arm:
                    self.arm.stop()
                self._record_locked("arm_mode_changed")
        return self.status()

    def update(self, source, inputs, observing=True, precision=False):
        if observing is not True:
            raise ValueError("Подтвердите наблюдение за роботом")
        clean = {}
        for key, value in inputs.items():
            number = float(value)
            if key not in ALL_ACTIONS or not math.isfinite(number) or abs(number) > 1.0001:
                raise ValueError("Неверное действие ручного управления")
            clean[str(key)] = number
        moving = self._moving(clean)
        with self.lock:
            if self.owner != source:
                raise ValueError("Источник управления не выбран")
            announce = moving and not self.takeover_active
            self.takeover_active = moving
            self.inputs = clean
            self.precision = bool(precision)
            self.lease = time.monotonic() + 0.45
            if self.stop_latched and not moving:
                self.neutral_seen = True
            self._record_locked("input")
        if announce and self.takeover_callback:
            self.takeover_callback()
        return self.status()

    @staticmethod
    def _moving(values):
        return any(abs(value) > 0.08 for value in values.values())

    def stop(self):
        with self.lock:
            self.stop_latched = True
            self.neutral_seen = not self._moving(self.inputs)
            self.generation += 1
            self._reset_arm_locked()
            self._release_locked()
            if self.arm:
                self.arm.stop()
            self._record_locked("stop")
        self.stop_all()
        return self.status()

    def resume(self, source, observing):
        if observing is not True:
            raise ValueError("Подтвердите наблюдение за роботом")
        with self.lock:
            if self.owner != source:
                raise ValueError("Сначала явно выберите источник управления")
            if self._moving(self.inputs):
                raise ValueError("Сначала отпустите все органы движения")
            if not self.neutral_seen:
                raise ValueError("После STOP требуется подтверждённая нейтраль")
            if self.resume_callback:
                self.resume_callback()
            self.stop_latched = False
            self.error = None
            self.lease = time.monotonic() + 0.45
            self._record_locked("resumed")
        return self.status()

    def disconnect(self, source):
        with self.lock:
            if self.owner == source:
                self.inputs = {}
                self.owner = None
                self.lease = 0.0
                self.takeover_active = False
                self.generation += 1
                self._reset_arm_locked()
                self._release_locked()
                if self.arm:
                    self.arm.stop()
                self._record_locked("disconnected")
        return self.status()

    def _release_locked(self):
        if self.drive_active:
            self.release()
            self.drive_active = False

    def status(self):
        with self.lock:
            return dict(owner=self.owner, stop_latched=self.stop_latched, neutral_seen=self.neutral_seen,
                lease_age_s=max(0.0, self.lease-time.monotonic()), precision=self.precision,
                arm_mode=self.arm_mode, generation=self.generation, inputs=dict(self.inputs),
                drive_active=self.drive_active, arm_busy=self.arm_busy, error=self.error,
                backend="shared_manual_teleop", control_scheme=SCHEME_ID, measured_joint_feedback=False)

    def _drive_vector(self, values, precision):
        nx = values.get("forward", 0.0)-values.get("backward", 0.0)
        ny = values.get("left", 0.0)-values.get("right", 0.0)
        nz = values.get("turn_left", 0.0)-values.get("turn_right", 0.0)
        translation = math.hypot(nx, ny)
        if translation > 1.0:
            nx /= translation
            ny /= translation
        wheel_peak = max(abs(nx-ny-nz), abs(nx+ny+nz), abs(nx+ny-nz), abs(nx-ny+nz), 1.0)
        nx, ny, nz = nx/wheel_peak, ny/wheel_peak, nz/wheel_peak
        scale = 0.1 if precision else 1.0
        return [nx*0.80*scale, ny*0.72*scale, nz*1.67*scale]

    def _arm_intent(self, values, precision):
        xyz = [values.get("arm_x_forward", 0.0)-values.get("arm_x_back", 0.0),
               values.get("arm_y_left", 0.0)-values.get("arm_y_right", 0.0),
               values.get("arm_z_up", 0.0)-values.get("arm_z_down", 0.0)]
        joints = [values.get("joint1_increase", 0.0)-values.get("joint1_decrease", 0.0),
                  values.get("joint2_increase", 0.0)-values.get("joint2_decrease", 0.0),
                  values.get("joint3_increase", 0.0)-values.get("joint3_decrease", 0.0),
                  values.get("pitch_up", 0.0)-values.get("pitch_down", 0.0),
                  values.get("wrist_right", 0.0)-values.get("wrist_left", 0.0),
                  values.get("grip_close", 0.0)-values.get("grip_open", 0.0)]
        factor = 0.35 if precision else 1.0
        return [value*factor for value in xyz], [value*factor for value in joints]

    @staticmethod
    def _integer_step(value):
        return math.floor(value) if value >= 0.0 else math.ceil(value)

    def _prepare_arm_segment_locked(self, values, now, precision):
        dt = max(0.0, min(0.12, now-self.last_integrator))
        self.last_integrator = now
        if self.arm_busy or dt <= 0.0:
            return None
        xyz_velocity, joint_velocity = self._arm_intent(values, precision)
        if self.arm_mode == "cartesian":
            joint_velocity[:5] = [0.0] * 5
        else:
            xyz_velocity = [0.0] * 3
        xyz_rate = 0.035
        joint_rates = [24.0, 24.0, 24.0, 30.0, 30.0, 36.0]
        for index, value in enumerate(xyz_velocity):
            self.xyz_residual[index] = self.xyz_residual[index]+value*xyz_rate*dt if abs(value) > 0.08 else 0.0
        for index, value in enumerate(joint_velocity):
            self.joint_residual[index] = self.joint_residual[index]+value*joint_rates[index]*dt if abs(value) > 0.08 else 0.0
        xyz_norm = math.sqrt(sum(value*value for value in self.xyz_residual))
        xyz_delta = [0.0] * 3
        if xyz_norm >= 0.003:
            scale = min(1.0, 0.006/xyz_norm)
            xyz_delta = [value*scale for value in self.xyz_residual]
            self.xyz_residual = [old-sent for old, sent in zip(self.xyz_residual, xyz_delta)]
        joint_delta = []
        for index, residual in enumerate(self.joint_residual):
            step = max(-3, min(3, self._integer_step(residual)))
            joint_delta.append(step)
            self.joint_residual[index] -= step
        if not any(abs(value) > 0.0 for value in xyz_delta) and not any(joint_delta):
            return None
        return xyz_delta, joint_delta, self.arm_mode, self.generation

    def _arm_segment(self, xyz_delta, joint_delta, arm_mode, generation, deadline, precision):
        try:
            with self.lock:
                if generation != self.generation or self.stop_latched:
                    raise ValueError("Команда устарела после смены режима или STOP")
            self.teaching.teleop(self.model(), xyz_delta, joint_delta, True, deadline, precision, arm_mode)
            self.error = None
        except (OSError, ValueError) as exc:
            self.error = str(exc)
        finally:
            with self.lock:
                self.arm_busy = False
                self.last_arm = time.monotonic()

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.owner is None:
                self.last_integrator = now
                return
            if now >= self.lease:
                self.inputs = {}
                self.owner = None
                self.generation += 1
                self._reset_arm_locked()
                self._release_locked()
                if self.arm:
                    self.arm.stop()
                return
            values = dict(self.inputs)
            precision = self.precision
            if self.stop_latched:
                self._reset_arm_locked()
                self._release_locked()
                return
            drive = self._drive_vector(values, precision)
            if any(abs(value) > 1e-6 for value in drive):
                self.drive(drive)
                self.drive_active = True
            else:
                self._release_locked()
            segment = self._prepare_arm_segment_locked(values, now, precision)
            if segment is not None and self.arm and self.model:
                self.arm_busy = True
                args = (*segment, now+0.8, precision)
                threading.Thread(target=self._arm_segment, args=args, daemon=True).start()

    def _run(self):
        while not self.closed:
            try:
                self.tick()
            except Exception as exc:
                self.error = str(exc)
            time.sleep(0.025)
