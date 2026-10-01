"""Bounded request/result correlation without ROS or serial side effects."""
from collections import OrderedDict


class PendingRequests:
    def __init__(self, capacity=256, lifetime_ns=1_000_000_000):
        if capacity < 1 or lifetime_ns <= 0:
            raise ValueError('positive request capacity and lifetime required')
        self.capacity, self.lifetime_ns = capacity, lifetime_ns
        self.context = None
        self.pending = OrderedDict()

    def clear(self):
        self.context = None
        self.pending.clear()

    def _context(self, boot, session):
        context = (boot, session)
        if self.context != context:
            self.clear()
            self.context = context

    def expire(self, now_ns):
        for key, value in list(self.pending.items()):
            if not 0 <= now_ns-value['sent_ns'] <= self.lifetime_ns:
                del self.pending[key]

    def remember(self, *, boot, session, sequence, operation, request, now_ns, earliest_result_us):
        self._context(boot, session)
        self.expire(now_ns)
        key = (sequence, int(operation))
        if key in self.pending:
            raise ValueError('duplicate wire request identity')
        while len(self.pending) >= self.capacity:
            self.pending.popitem(last=False)
        self.pending[key] = dict(source_id=request['source_id'],
            source_sequence=request['source_sequence'], boot=boot, session=session,
            sent_ns=now_ns, earliest_result_us=earliest_result_us)

    def match(self, *, boot, session, sequence, operation, acquired_us, now_ns):
        # An ACK has no session field on the wire. The live session, send-time
        # floor and operation all have to agree; a reused sequence alone is not
        # sufficient. Reconnect/reboot/session changes discard pending ACKs.
        self._context(boot, session)
        self.expire(now_ns)
        key = (sequence, int(operation))
        value = self.pending.get(key)
        if value is None or acquired_us < value['earliest_result_us']:
            return None
        self.pending.pop(key)
        return {key: value[key] for key in ('source_id', 'source_sequence', 'boot', 'session')}
