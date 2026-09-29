"""Core-owned stamped base commands, explicit recovery, no reconnect resume."""
class BaseTransport:
    def __init__(self, send):
        self.send = send
        self.sequence = 0
        self.state = {}
        self.open_deadline = None
        self.boot = None

    def emit(self, operation, now_ns, source_ns=None, **fields):
        source_ns = now_ns if source_ns is None else source_ns
        expires = source_ns+200_000_000
        if expires <= now_ns:
            return False
        self.sequence += 1
        self.send(dict(operation=operation, source_id='explorer_control',
            source_sequence=self.sequence, source_monotonic_ns=source_ns,
            expires_monotonic_ns=expires, **fields))
        return True

    def observe(self, state, now_ns):
        boot=(state.get('identity') or {}).get('boot')
        if self.boot is not None and boot != self.boot:
            self.open_deadline=None
        self.boot=boot
        self.state=state
        if self.open_deadline is not None:
            if now_ns >= self.open_deadline:
                self.open_deadline=None
            elif state.get('session_state') == 'disarmed':
                self.emit('OPEN', now_ns)
                self.open_deadline=None

    def start(self, now_ns):
        if not self.state.get('telemetry_fresh'):
            raise ValueError('Нет свежей связи со STM32')
        if self.state.get('telemetry_only', True):
            raise ValueError('Приёмка STM32: доступна только телеметрия')
        current=self.state.get('session_state')
        if current == 'fault':
            self.emit('CLEAR',now_ns)
            self.open_deadline=now_ns+200_000_000
        elif current == 'disarmed':
            self.emit('OPEN',now_ns)
        elif current != 'active':
            raise ValueError('Дождитесь завершения открытия сессии STM32')

    def stop(self, now_ns):
        self.open_deadline=None
        self.emit('ESTOP',now_ns)

    def velocity(self, values, source_ns, now_ns):
        if self.state.get('session_state') != 'active':
            return
        if not any(values) or source_ns+200_000_000<=now_ns:
            self.emit('HOLD',now_ns)
        else:
            self.emit('BASE',now_ns,source_ns,velocity=list(values))
