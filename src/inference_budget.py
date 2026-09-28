"""One wall-clock budget shared by warmup, inference and planning rounds."""
import time

class Budget:
    def __init__(self, seconds=10):
        self.deadline=time.monotonic()+seconds

    def remaining(self, maximum=None):
        value=self.deadline-time.monotonic()
        if value<=0:raise TimeoutError('Истёк бюджет решения; старое предложение не исполняется')
        return value if maximum is None else min(value,maximum)
