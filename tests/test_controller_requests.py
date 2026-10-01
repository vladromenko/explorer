import unittest
from controller_protocol import Kind
from controller_requests import PendingRequests


class PendingRequestTests(unittest.TestCase):
    def add(self, pending, sequence, operation, source, now=100, boot=1, session=2):
        pending.remember(boot=boot, session=session, sequence=sequence, operation=operation,
            request=dict(source_id=source, source_sequence=sequence+1000), now_ns=now,
            earliest_result_us=now)

    def match(self, pending, sequence, operation, now=200, boot=1, session=2, acquired=200):
        return pending.match(boot=boot, session=session, sequence=sequence, operation=operation,
            acquired_us=acquired, now_ns=now)

    def test_interleaved_base_arm_results_keep_origin_and_are_consumed_once(self):
        pending = PendingRequests()
        self.add(pending, 10, Kind.ARM, 'arm')
        self.add(pending, 11, Kind.BASE, 'base')
        self.assertEqual(self.match(pending, 11, Kind.BASE)['source_id'], 'base')
        result = self.match(pending, 10, Kind.ARM)
        self.assertEqual((result['source_id'], result['source_sequence']), ('arm', 1010))
        self.assertIsNone(self.match(pending, 10, Kind.ARM))

    def test_reused_sequence_cannot_match_previous_boot_session_or_earlier_result(self):
        pending = PendingRequests()
        self.add(pending, 1, Kind.OPEN, 'old')
        self.add(pending, 1, Kind.OPEN, 'new', now=300, boot=2, session=1)
        self.assertIsNone(self.match(pending, 1, Kind.OPEN, now=310, acquired=200, boot=2, session=1))
        self.assertEqual(self.match(pending, 1, Kind.OPEN, now=310, acquired=305,
                                   boot=2, session=1)['source_id'], 'new')
        self.add(pending, 1, Kind.CLEAR, 'next', now=400, boot=2, session=2)
        self.assertIsNone(self.match(pending, 1, Kind.OPEN, now=410, acquired=405, boot=2, session=2))
        self.assertEqual(self.match(pending, 1, Kind.CLEAR, now=410, acquired=405,
                                   boot=2, session=2)['source_id'], 'next')

    def test_disconnect_expiry_and_capacity_drop_old_results(self):
        pending = PendingRequests(capacity=2, lifetime_ns=50)
        self.add(pending, 1, Kind.ARM, 'arm', now=100)
        self.add(pending, 2, Kind.ARM, 'arm', now=101)
        self.add(pending, 3, Kind.ARM, 'arm', now=102)
        self.assertIsNone(self.match(pending, 1, Kind.ARM, now=103))
        self.assertIsNone(self.match(pending, 2, Kind.ARM, now=152))
        pending.clear()
        self.assertIsNone(self.match(pending, 3, Kind.ARM, now=103))


if __name__ == '__main__':
    unittest.main()
