import random
import struct
import unittest
from controller_protocol import ClockMapping, Kind, Parser, Session, base_payload, decode, encode


class ControllerProtocolTests(unittest.TestCase):
    def test_all_packet_lengths(self):
        rng = random.Random(49)
        for n in range(1025):
            data = rng.randbytes(n)
            packet = encode(Kind.SERVO, data)
            self.assertEqual(decode(packet[:-1]), (Kind.SERVO, data))
            parser = Parser()
            self.assertEqual(parser.feed(packet[:5]) + parser.feed(packet[5:]), [(Kind.SERVO, data)])

    def test_corruption_overflow_and_resynchronization(self):
        parser = Parser()
        parser.feed(b'\x01' * 5000 + b'\0')
        packet = bytearray(encode(Kind.BASE, base_payload([.1, -.1, .2])))
        packet[8] ^= 32
        self.assertEqual(parser.feed(packet), [])
        self.assertEqual(parser.feed(encode(Kind.HELLO, b'12345678')), [(Kind.HELLO, b'12345678')])
        self.assertEqual(parser.errors, 2)

    def test_mapping_does_not_extend_source_lease(self):
        clock = ClockMapping.observation(123, 1_000_000_000, 1_004_000_000, 50_002_000)
        issued, expiry = clock.lease(1_010_000_000, 1_210_000_000, 1_020_000_000)
        self.assertLessEqual(issued, 50_010_000)
        self.assertLessEqual(expiry, 50_210_000)
        with self.assertRaises(ValueError):
            clock.lease(1_010_000_000, 1_210_000_000, 1_210_000_000)
        with self.assertRaises(ValueError):
            clock.interval(4_000_000_000, 4_000_000_000)

    def test_new_boot_never_restarts_motion(self):
        session = Session()
        clock = ClockMapping.observation(123, 0, 1_000_000, 100_000)
        session.synchronize(clock, 9)
        arguments = dict(source_ns=2_000_000, expires_ns=102_000_000, now_ns=3_000_000,
                         source_id='core-generation-a', source_sequence=1)
        with self.assertRaises(ValueError):
            session.prepare(Kind.BASE, data=base_payload([.1, 0, 0]), **arguments)
        packet = session.prepare(Kind.OPEN, **arguments)
        kind, raw = decode(packet[:-1])
        self.assertEqual(struct.unpack_from('<QQQ', raw), (123, 10, 1))
        self.assertEqual(session.state, 'opening')
        session.result(kind, 1, 0)
        self.assertEqual(session.state, 'active')
        with self.assertRaises(ValueError):
            session.prepare(Kind.HOLD, **arguments)
        session.synchronize(ClockMapping.observation(124, 0, 1_000_000, 50_000))
        self.assertEqual(session.state, 'disarmed')

    def test_source_timestamp_preserved(self):
        session = Session()
        session.synchronize(ClockMapping.observation(42, 0, 1000000, 100000))
        session.prepare(Kind.OPEN, 2000000, 102000000, 3000000, source_id='core', source_sequence=1)
        session.result(Kind.OPEN, 1, 0)
        first = session.prepare(Kind.BASE, 10000000, 110000000, 20000000,
                                base_payload([-.1, .1, -.2]), source_id='core', source_sequence=2)
        _, payload = decode(first[:-1])
        self.assertLessEqual(struct.unpack_from('<Q', payload, 32)[0], 210000)
        with self.assertRaises(ValueError):
            session.prepare(Kind.BASE, 10000000, 110000000, 120000000,
                            base_payload([-.1, .1, -.2]), source_id='core', source_sequence=3)


if __name__ == '__main__':
    unittest.main()
