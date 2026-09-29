"""Explorer controller protocol v1; no ROS, serial side effects or clock epoch.

The originating control task supplies source/expiry monotonic timestamps.
The serial driver never renews a command or retries a rejected motion packet.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
import struct
import zlib

MAX_PAYLOAD = 1024
MAX_FRAME = 1040
MAX_LEASE_NS = 250_000_000


class Kind(IntEnum):
    HELLO = 1
    OPEN = 2
    BASE = 3
    HOLD = 4
    CANCEL = 5
    ESTOP = 6
    CLEAR = 7
    ARM = 8
    ARM_ENABLE = 9
    ARM_CANCEL = 10
    RGB = 11
    BEEP = 12
    CALIBRATION = 13
    RECOVERY_ENABLE = 14
    ARM_RECOVER = 15
    STATUS = 64
    RESULT = 65
    SERVO = 66
    IMU = 67
    BATTERY = 68
    LIDAR0 = 69
    LIDAR1 = 70
    IDENTITY = 71
    SENSOR_DIAGNOSTICS = 72


def encode(kind, payload=b''):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError('controller payload too large')
    raw = struct.pack('<BBH', 1, int(kind), len(payload)) + payload
    raw += struct.pack('<I', zlib.crc32(raw))
    output = bytearray([0])
    code_at, code = 0, 1
    for byte in raw:
        if byte == 0:
            output[code_at] = code
            code_at, code = len(output), 1
            output.append(0)
        else:
            output.append(byte)
            code += 1
            if code == 255:
                output[code_at] = code
                code_at, code = len(output), 1
                output.append(0)
    output[code_at] = code
    output.append(0)
    return bytes(output)


def decode(encoded):
    if not encoded or len(encoded) >= MAX_FRAME or 0 in encoded:
        raise ValueError('invalid COBS envelope')
    raw = bytearray()
    index = 0
    while index < len(encoded):
        code = encoded[index]
        index += 1
        if index + code - 1 > len(encoded):
            raise ValueError('truncated COBS block')
        raw.extend(encoded[index:index + code - 1])
        index += code - 1
        if code < 255 and index < len(encoded):
            raw.append(0)
    if len(raw) < 8 or len(raw) > MAX_PAYLOAD + 8:
        raise ValueError('invalid decoded length')
    version, kind, length = struct.unpack_from('<BBH', raw)
    if version != 1 or length != len(raw) - 8:
        raise ValueError('incompatible protocol version/length')
    if struct.unpack_from('<I', raw, len(raw) - 4)[0] != zlib.crc32(raw[:-4]):
        raise ValueError('CRC mismatch')
    return Kind(kind), bytes(raw[4:-4])


class Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.overflow = False
        self.errors = 0

    def feed(self, data):
        frames = []
        for byte in data:
            if byte:
                if len(self.buffer) < MAX_FRAME and not self.overflow:
                    self.buffer.append(byte)
                else:
                    self.overflow = True
            else:
                if self.buffer and not self.overflow:
                    try:
                        frames.append(decode(self.buffer))
                    except ValueError:
                        self.errors += 1
                elif self.overflow:
                    self.errors += 1
                self.buffer.clear()
                self.overflow = False
        return frames


@dataclass(frozen=True)
class ClockMapping:
    boot: int
    host_received_ns: int
    offset_low_us: int
    offset_high_us: int

    @classmethod
    def observation(cls, boot, sent_ns, received_ns, mcu_us):
        if not boot or not 0 <= received_ns - sent_ns <= 20_000_000:
            raise ValueError('clock round trip exceeds 20 ms or boot identity missing')
        return cls(boot, received_ns, mcu_us - (received_ns + 999) // 1000,
                   mcu_us - sent_ns // 1000)

    def interval(self, host_ns, now_ns):
        age = now_ns - self.host_received_ns
        if not 0 <= age <= 2_000_000_000:
            raise ValueError('clock mapping stale')
        # 200 ppm drift allowance plus one timer tick; must be checked on bench.
        drift = math.ceil(age / 5_000_000) + 2
        return (host_ns // 1000 + self.offset_low_us - drift,
                (host_ns + 999) // 1000 + self.offset_high_us + drift)

    def lease(self, source_ns, expires_ns, now_ns):
        if not source_ns <= now_ns < expires_ns:
            raise ValueError('source command is future or expired')
        if not 0 < expires_ns - source_ns <= MAX_LEASE_NS:
            raise ValueError('source lease exceeds controller budget')
        issued, _ = self.interval(source_ns, now_ns)
        expires, _ = self.interval(expires_ns, now_ns)
        _, now_high = self.interval(now_ns, now_ns)
        if issued < 0 or expires <= now_high:
            raise ValueError('not enough lease left after clock uncertainty')
        return issued, expires

    def source_host_ns(self, mcu_us, now_ns, max_age_ns):
        low, high = self.interval(now_ns, now_ns)
        if mcu_us > high or (high - mcu_us) * 1000 > max_age_ns:
            raise ValueError('stale or future MCU sample')
        # Oldest possible acquisition time avoids presenting stale data as fresh.
        return min(now_ns, (mcu_us - self.offset_high_us) * 1000)


class Session:
    def __init__(self):
        self.clock = None
        self.session = 0
        self.highest = 0
        self.sequence = 0
        self.state = 'disconnected'
        self.source_sequences = {}

    def synchronize(self, clock, highest_session=0):
        if self.clock is None or self.clock.boot != clock.boot:
            self.session = 0
            self.sequence = 0
            self.state = 'disarmed'
            self.source_sequences.clear()
            self.highest = highest_session
        else:
            self.highest = max(self.highest, highest_session)
        self.clock = clock

    def disconnected(self):
        self.clock = None
        self.session = 0
        self.state = 'disconnected'
        self.source_sequences.clear()

    def prepare(self, kind, source_ns, expires_ns, now_ns, data=b'', *, source_id, source_sequence):
        if self.clock is None:
            raise ValueError('controller handshake required')
        if not isinstance(source_id, str) or not source_id or source_sequence <= self.source_sequences.get(source_id, 0):
            raise ValueError('replayed source command')
        issued, expires = self.clock.lease(source_ns, expires_ns, now_ns)
        if kind in (Kind.OPEN, Kind.CLEAR):
            if self.state not in ('disarmed', 'fault'):
                raise ValueError('explicit recovery must start from stopped state')
            session = self.highest + 1
            sequence = 1
        else:
            if self.state != 'active' or not self.session:
                raise ValueError('explicit session authorization required')
            session, sequence = self.session, self.sequence + 1
        payload = struct.pack('<QQQQQ', self.clock.boot, session, sequence, issued, expires) + data
        packet = encode(kind, payload)
        self.source_sequences[source_id] = source_sequence
        self.session, self.sequence = session, sequence
        if kind in (Kind.OPEN, Kind.CLEAR):
            self.highest = session
            self.state = 'opening' if kind == Kind.OPEN else 'clearing'
        elif kind == Kind.CANCEL:
            self.state = 'cancelling'
        return packet

    def result(self, kind, sequence, result):
        if sequence != self.sequence:
            return
        if result:
            self.state = 'fault'
        elif kind == Kind.OPEN:
            self.state = 'active'
        elif kind in (Kind.CLEAR, Kind.CANCEL):
            self.state = 'disarmed'


def base_payload(velocity, finite_us=0):
    if len(velocity) != 3 or any(not math.isfinite(v) for v in velocity):
        raise ValueError('three finite signed velocities required')
    if any(abs(v) > limit for v, limit in zip(velocity, (.7, .7, 4.2))):
        raise ValueError('velocity exceeds controller envelope')
    if type(finite_us) is not int or not 0 <= finite_us <= 250000:
        raise ValueError('finite motion must fit its command lease')
    return struct.pack('<fffQ', *velocity, finite_us)


def arm_payload(position):
    if len(position) != 6 or any(not math.isfinite(v) for v in position):
        raise ValueError('six finite calibrated joint positions required')
    return struct.pack('<6fH', *position, 0)
