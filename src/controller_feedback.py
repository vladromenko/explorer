"""Measured controller data, calibration and freshness independent of ROS."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import math
import struct


def float32(value):
    """Use the coefficients/positions actually decoded by the MCU."""
    return struct.unpack('<f', struct.pack('<f', value))[0]


def adjacent_float32(value, positive):
    value = float32(value)
    if value == 0:
        return struct.unpack('<f', struct.pack('<I', 1 if positive else 0x80000001))[0]
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits += 1 if (value > 0) == positive else -1
    return struct.unpack('<f', struct.pack('<I', bits))[0]


@dataclass(frozen=True)
class Calibration:
    joint: int
    radians_per_tick: float
    radians_at_raw_zero: float
    lower: float
    upper: float
    command_min: int
    command_max: int
    physical_degrees_per_tick: float
    physical_degrees_at_raw_zero: float
    recovery_min: int = 0
    recovery_max: int = 0

    def canonical(self):
        return replace(self, **{name: float32(getattr(self, name)) for name in
            ('radians_per_tick', 'radians_at_raw_zero', 'lower', 'upper')})

    def observe(self, raw):
        if type(raw) is not int or not 96 <= raw <= 4000:
            raise ValueError('invalid raw measurement')
        q = raw*self.radians_per_tick + self.radians_at_raw_zero
        physical = raw*self.physical_degrees_per_tick + self.physical_degrees_at_raw_zero
        return dict(raw_ticks=raw, physical_deg=physical, position_rad=q,
                    outside_soft_limit=not self.lower <= q <= self.upper,
                    raw_valid=True, position_valid=True)

    def target_raw(self, q):
        if not math.isfinite(q) or not self.lower <= q <= self.upper:
            raise ValueError('joint target outside calibrated soft limits')
        coordinate = (q-self.radians_at_raw_zero)/self.radians_per_tick
        raw = round(coordinate)
        if not self.command_min-.5 <= coordinate <= self.command_max+.5:
            raise ValueError('joint target outside command raw interval')
        choices = [candidate for candidate in (raw-1, raw, raw+1)
                   if self.command_min <= candidate <= self.command_max and
                   self.lower <= candidate*self.radians_per_tick+self.radians_at_raw_zero <= self.upper and
                   abs(candidate-coordinate) <= 1.01]
        if not choices:
            raise ValueError('no representable tick inside the joint limit')
        return min(choices, key=lambda candidate: abs(candidate-coordinate))

    def target_position(self, q):
        """Encode a representable target using exactly the MCU calibration.

        Quantization never expands limits or clips observed positions. At an
        exact boundary one adjacent wire float may be needed to stay inside.
        """
        calibration = self.canonical()
        raw = calibration.target_raw(q)
        center = float32(raw*calibration.radians_per_tick+calibration.radians_at_raw_zero)
        for position in (center, adjacent_float32(center, True), adjacent_float32(center, False)):
            try:
                if calibration.target_raw(position) == raw:
                    return position
            except ValueError:
                pass
        raise ValueError('joint target cannot be represented on the controller wire')

    def payload(self):
        return struct.pack('<4H4fHHI', 96, 4000, self.command_min, self.command_max,
                           self.radians_per_tick, self.radians_at_raw_zero,
                           self.lower, self.upper, self.recovery_min, self.recovery_max, 0)


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 1 or len(data.get('joints', [])) != 6:
        raise ValueError('explicit six-joint calibration required')
    joints = [Calibration(**item).canonical() for item in data['joints']]
    for index, c in enumerate(joints):
        if c.joint != index+1 or not all(math.isfinite(v) for v in (
            c.radians_per_tick, c.radians_at_raw_zero, c.lower, c.upper,
            c.physical_degrees_per_tick, c.physical_degrees_at_raw_zero)):
            raise ValueError('invalid calibration identity/coefficient')
        if c.radians_per_tick == 0 or c.lower >= c.upper or not 96 <= c.command_min < c.command_max <= 4000:
            raise ValueError('invalid calibration interval')
        if c.recovery_min or c.recovery_max:
            if not data.get('recovery_evidence_sha256'):
                raise ValueError('recovery corridor needs a separate physical acceptance record')
            if not 96 <= c.recovery_min < c.recovery_max <= 4000:
                raise ValueError('invalid recovery corridor')
    return joints


def vendor_calibration():
    """Nominal coefficients; physical calibration acceptance is a separate record.

    The vendor's +0.5 integer rounding bias is intentionally not a joint offset.
    Servo 6 is published as an actuator angle, NOT an invented jaw aperture.
    """
    result = []
    for joint in range(1, 7):
        if joint <= 4:
            slope, offset = -180/2200, 180+900*180/2200
            lo, hi = 900, 3100
            sign, zero = (-1 if joint == 1 else 1), 90
            lower, upper = -math.pi/2, math.pi/2
        elif joint == 5:
            slope, offset = 270/3320, -380*270/3320
            lo, hi, sign, zero = 380, 3700, 1, 90
            lower, upper = -math.pi/2, math.pi
        else:
            slope, offset = 180/2200, -900*180/2200
            lo, hi, sign, zero = 1267, 2978, 1, 0
            lower, upper = math.radians(30), math.radians(170)
        result.append(Calibration(joint, math.radians(slope)*sign, math.radians(offset-zero)*sign,
                                  lower, upper, lo, hi, slope, offset))
    return [calibration.canonical() for calibration in result]


def decode_status(payload):
    if len(payload) != 216:
        raise ValueError('incompatible status layout')
    times = struct.unpack_from('<6Q', payload)
    mode, fault, arm_enabled, arm_cancel = payload[48:52]
    wheels = []
    for i, name in enumerate(('front_left', 'front_right', 'rear_left', 'rear_right')):
        raw, delta, target, measured, pwm, integral, motor = struct.unpack_from('<Hi4fI', payload, 52+26*i)
        if motor != (1, 3, 2, 4)[i] or not all(math.isfinite(v) for v in (target, measured, pwm, integral)):
            raise ValueError('bad wheel telemetry')
        wheels.append(dict(name=name, motor=motor, raw_counter=raw, raw_delta=delta,
                           target_rad_s=target, measured_rad_s=measured, pwm=pwm, integral=integral))
    velocity = struct.unpack_from('<3f', payload, 156)
    validity = struct.unpack_from('<I', payload, 212)[0]
    if validity not in (0, 1):
        raise ValueError('incompatible measurement validity flags')
    if not all(math.isfinite(v) for v in velocity):
        raise ValueError('bad measured base velocity')
    names = ('control_max_us', 'command_drops', 'result_drops', 'telemetry_drops', 'rejected_commands')
    return dict(acquired_us=times[0], boot=times[1], session=times[2], sequence=times[3],
                expires_us=times[4], finite_end_us=times[5], mode=mode, fault=fault,
                arm_enabled=bool(arm_enabled), arm_cancel_pending=bool(arm_cancel),
                wheels=wheels, velocity=list(velocity),
                encoder_measurement_valid=bool(validity),
                diagnostics=dict(zip(names, struct.unpack_from('<5I', payload, 168)),
                    host_rx_overruns=struct.unpack_from('<I', payload, 204)[0],
                    host_uart_errors=struct.unpack_from('<I', payload, 208)[0]),
                arm_sent_generation=struct.unpack_from('<Q', payload, 188)[0],
                highest_session=struct.unpack_from('<Q', payload, 196)[0])


class ArmFeedback:
    def __init__(self, calibration=None):
        self.calibration = calibration or vendor_calibration()
        self.samples = [dict(joint=i+1, raw_valid=False, position_valid=False,
                             error='not_sampled', acquired_monotonic_ns=None) for i in range(6)]

    def consume(self, payload, clock, now_ns):
        if len(payload) != 30:
            raise ValueError('incompatible servo record')
        event_us, joint, error, device, raw, acquired_us = struct.unpack_from('<QBBBHQ', payload)
        if not 1 <= joint <= 6:
            raise ValueError('unknown servo')
        sample = dict(joint=joint, error=error, device_error=device, raw_valid=False,
                      position_valid=False, acquired_monotonic_ns=None,
                      reply_hex=payload[21:29].hex(), reply_header2=payload[29])
        clock.source_host_ns(event_us, now_ns, 500_000_000)
        if error == 0:
            acquired = clock.source_host_ns(acquired_us, now_ns, 250_000_000)
            # Revalidate the original response: not just the MCU's boolean flag.
            reply = payload[21:29]
            if (reply[:2] != bytes((255, 245)) or reply[2:5] != bytes((joint, 4, 0)) or
                ((~sum(reply[2:7])) & 255) != reply[7] or int.from_bytes(reply[5:7], 'big') != raw):
                raise ValueError('servo raw evidence disagrees with record')
            sample.update(self.calibration[joint-1].observe(raw), acquired_monotonic_ns=acquired)
        self.samples[joint-1] = sample
        return sample

    def snapshot(self, now_ns):
        samples = []
        for original in self.samples:
            sample = dict(original)
            stamp = sample['acquired_monotonic_ns']
            sample['fresh'] = stamp is not None and 0 <= now_ns-stamp <= 250_000_000
            samples.append(sample)
        return dict(source='stm32_uart3_readback', measured=True, joints=samples,
                    all_fresh=all(s['fresh'] and s['position_valid'] for s in samples),
                    gripper_aperture_measured=False, force_measured=False)


class ScanAssembler:
    def __init__(self):
        self.ranges = [math.nan]*666
        self.intensities = [0.]*666
        self.started_us = None
        self.last_us = None
        self.covered = set()

    def packet(self, payload):
        if len(payload) < 25:
            raise ValueError('lidar packet truncated')
        stamp, errors = struct.unpack_from('<QI', payload)
        data = payload[12:]
        header, ct, count, first, last, checksum = struct.unpack_from('<HBBHHH', data)
        if header != 0x55aa or not count or len(data) != 10 + 3*count or not (first & 1 and last & 1):
            raise ValueError('lidar packet header/angles invalid')
        check = header ^ (ct | count << 8) ^ first ^ last
        for i in range(count):
            intensity, distance = struct.unpack_from('<BH', data, 10+3*i)
            check ^= intensity ^ distance
        if check != checksum:
            raise ValueError('lidar checksum invalid')
        complete = None
        if ct & 1:
            if self.started_us is not None and self.last_us is not None and 50000 <= stamp-self.started_us <= 400000:
                complete = dict(acquired_us=self.started_us, ended_us=stamp, ranges=self.ranges,
                                intensities=self.intensities, covered=len(self.covered), packet_errors=errors)
            self.ranges, self.intensities, self.covered = [math.nan]*666, [0.]*666, set()
            self.started_us = stamp
        if self.last_us is not None and not 0 < stamp-self.last_us < 100000:
            self.started_us = None
        self.last_us = stamp
        start_deg, end_deg = (first >> 1)/64., (last >> 1)/64.
        difference = (end_deg-start_deg) % 360
        for i in range(count):
            intensity, raw = struct.unpack_from('<BH', data, 10+3*i)
            distance = (raw >> 2) / 1000.
            angle = (start_deg + (difference*i/(count-1) if count > 1 else 0)) % 360
            index = round(angle*666/360) % 666
            self.ranges[index] = distance if .05 <= distance <= 12 else math.nan
            self.intensities[index] = float(intensity)
            self.covered.add(index)
        return complete
