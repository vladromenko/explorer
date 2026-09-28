"""Local mission ownership. A lease cannot create or resurrect a session."""
class MotionSession:
    def __init__(self):
        self.mission = None
        self.lease_at = -1e9
        self.held = True
        self.retired = set()

    def begin(self, mission, now):
        if not isinstance(mission, str) or not 1 <= len(mission) <= 64:
            raise ValueError('Invalid mission identity')
        if mission in self.retired or self.mission is not None:
            raise ValueError('Mission already active or retired')
        self.mission, self.lease_at, self.held = mission, now, True

    def require(self, mission):
        if not mission or mission != self.mission:
            raise ValueError('No matching live mission')

    def renew(self, mission, now):
        self.require(mission)
        if now - self.lease_at > .25:
            self.end(mission)
            raise ValueError('Mission lease expired; new session required')
        self.lease_at = now

    def hold(self, mission):
        self.require(mission)
        self.held = True

    def resume(self, mission, now):
        self.renew(mission, now)
        self.held = False

    def end(self, mission=None):
        if mission is not None:
            self.require(mission)
        if self.mission:
            self.retired.add(self.mission)
        self.mission, self.lease_at, self.held = None, -1e9, True

    def permits(self, now):
        return self.mission is not None and not self.held and 0 <= now-self.lease_at <= .25
