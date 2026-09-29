#!/usr/bin/env python3
"""Read telemetry and send HELLO only; never opens a motion session."""
import collections
import json
from pathlib import Path
import struct
import sys
import time
import serial
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from controller_protocol import Parser, Kind, encode
from controller_feedback import decode_status

p = Parser()
counts = collections.Counter()
latest = {}
identities = []
total = 0
s = serial.Serial(port=None, baudrate=2000000, timeout=.02, exclusive=True)
s.dtr = s.rts = False
s.port = '/dev/explorer_mcu'
s.open()
start = time.monotonic()
next_hello = start
try:
    while time.monotonic() - start < 8:
        now = time.monotonic()
        if now >= next_hello:
            s.write(encode(Kind.HELLO, struct.pack('<Q', time.monotonic_ns())))
            next_hello = now + .5
        data = s.read(8192)
        total += len(data)
        for kind, payload in p.feed(data):
            counts[kind.name] += 1
            if kind == Kind.IDENTITY:
                fields = struct.unpack_from('<QQQIIH', payload)
                identities.append(dict(boot=fields[2], device=fields[3], revision=fields[4],
                    flash_kib=fields[5], source=payload[49:81].hex(),
                    reset_flags=struct.unpack_from('<I', payload, 84)[0]))
            elif kind == Kind.STATUS:
                latest['status'] = decode_status(payload)
            elif kind == Kind.SERVO:
                values = struct.unpack_from('<QBBBHQ', payload)
                latest['servo'+str(values[1])] = dict(error=values[2], raw=values[4], reply=payload[21:29].hex())
            elif kind == Kind.BATTERY:
                values = struct.unpack('<QBHf', payload)
                latest['battery'] = dict(error=values[1], raw=values[2], volts=values[3])
            elif kind == Kind.IMU:
                latest['imu'] = dict(error=payload[8], mag_error=payload[9])
finally:
    s.close()
print(json.dumps(dict(received_bytes=total, parser_errors=p.errors, counts=counts,
                     identities=identities, latest=latest), indent=2))
