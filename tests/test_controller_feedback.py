import math
import struct
import unittest
from controller_feedback import ArmFeedback, Calibration, ScanAssembler, vendor_calibration
from controller_protocol import ClockMapping
from arm_feedback import decode_position, describe


class ControllerFeedbackTests(unittest.TestCase):
    def test_every_wire_target_uses_wire_calibration_and_stays_in_limits(self):
        for c in vendor_calibration():
            wire = struct.unpack('<4H4fHHI', c.payload())
            slope, offset, lower, upper = wire[4:8]
            self.assertEqual((c.radians_per_tick, c.radians_at_raw_zero, c.lower, c.upper),
                             (slope, offset, lower, upper))
            valid = [raw for raw in range(c.command_min, c.command_max+1)
                     if lower <= raw*slope+offset <= upper]
            for raw in valid:
                target = c.target_position(raw*slope+offset)
                # This independent calculation mirrors the MCU's incoming
                # float32 -> double inverse, including C's positive rounding.
                incoming = struct.unpack('<f', struct.pack('<f', target))[0]
                self.assertTrue(lower <= incoming <= upper)
                self.assertEqual(int((incoming-offset)/slope+.5), raw)
            for boundary in (lower, upper):
                incoming = c.target_position(boundary)
                self.assertTrue(lower <= incoming <= upper)
            with self.assertRaises(ValueError):
                c.target_position(lower-.01)

    def test_boundary_float_rounding_chooses_inside_without_widening(self):
        # A narrow synthetic boundary lies at one valid tick after canonical
        # coefficient serialization; exercise its nearest representable float.
        c = Calibration(1, .001, -.1, 0., .001, 96, 4000, 1., 0.).canonical()
        values = [raw for raw in range(96, 110) if c.lower <= raw*c.radians_per_tick+c.radians_at_raw_zero <= c.upper]
        self.assertTrue(values)
        for raw in values:
            q = c.target_position(raw*c.radians_per_tick+c.radians_at_raw_zero)
            self.assertEqual(c.target_raw(q), raw)
            self.assertTrue(c.lower <= q <= c.upper)

    def test_web_readback_keeps_signed_positions_and_expires(self):
        samples=[dict(joint=i+1,position_valid=True,fresh=True,error=0,
            physical_deg=-12.,position_rad=-.2,raw_ticks=3300,outside_soft_limit=True) for i in range(6)]
        state=dict(sensor_age=dict(arm=.02),arm_measurements=dict(source='stm32_uart3_readback',
            all_fresh=True,joints=samples))
        self.assertEqual(describe(state)['measured']['values'],[-12.]*6)
        state['sensor_age']['arm']=1.
        self.assertIsNone(describe(state)['measured']['values'])
        self.assertFalse(describe(state)['measured']['available'])
    def test_negative_physical_angle_remains_valid_measurement(self):
        c = vendor_calibration()[1]
        sample = c.observe(3634)
        self.assertAlmostEqual(sample['physical_deg'], -43.6909090909)
        self.assertTrue(sample['position_valid'])
        self.assertTrue(sample['outside_soft_limit'])
        with self.assertRaises(ValueError):
            c.target_raw(sample['position_rad'])

    def test_echo_cannot_be_feedback(self):
        request = bytes([255, 255, 1, 4, 2, 0x38, 2])
        request += bytes([(~sum(request[2:])) & 255])
        with self.assertRaises(ValueError):
            decode_position(request, 1)
        with self.assertRaises(ValueError):
            decode_position(request, 1, 255)

    def test_per_joint_validity_and_source_timestamp(self):
        clock = ClockMapping.observation(1, 0, 1000000, 100000)
        feedback = ArmFeedback()
        reply = bytes([255, 245, 2, 4, 0, 3634 >> 8, 3634 & 255])
        reply += bytes([(~sum(reply[2:])) & 255])
        payload = struct.pack('<QBBBHQ', 110000, 2, 0, 0, 3634, 110000) + reply + bytes([245])
        measured = feedback.consume(payload, clock, 12000000)
        self.assertTrue(measured['position_valid'])
        self.assertLess(measured['physical_deg'], 0)
        snapshot = feedback.snapshot(12000000)
        self.assertFalse(snapshot['all_fresh'])
        self.assertTrue(snapshot['joints'][1]['fresh'])
        self.assertFalse(snapshot['joints'][0]['fresh'])
        self.assertFalse(feedback.snapshot(400000000)['joints'][1]['fresh'])

    def test_lidar_missing_angles_are_unknown_not_free_space(self):
        assembler = ScanAssembler()
        def packet(stamp, ct, first, last, distances):
            f, l = round(first*64)*2+1, round(last*64)*2+1
            checksum = 0x55aa ^ (ct | len(distances) << 8) ^ f ^ l
            body = b''
            for distance in distances:
                raw = round(distance*1000)*4
                body += struct.pack('<BH', 50, raw)
                checksum ^= 50 ^ raw
            return struct.pack('<QI HBBHHH', stamp, 0, 0x55aa, ct, len(distances), f, l, checksum) + body
        assembler.packet(packet(10000, 1, 0, 1, [1, 1]))
        assembler.packet(packet(60000, 0, 90, 91, [2, 2]))
        complete = assembler.packet(packet(150000, 1, 0, 1, [1, 1]))
        self.assertIsNotNone(complete)
        self.assertTrue(any(math.isnan(v) for v in complete['ranges']))
        self.assertFalse(any(math.isinf(v) for v in complete['ranges']))
        self.assertEqual(complete['acquired_us'], 10000)


if __name__ == '__main__':
    unittest.main()
