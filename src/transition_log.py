"""Bounded nonblocking event queue; disk writes happen off the control thread."""
import json
import copy
import logging
from logging.handlers import RotatingFileHandler
import queue
import threading
import time

class TransitionLog:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=256)
        self.dropped = 0
        self.last = None
        self.handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3)
        threading.Thread(target=self.run, daemon=True).start()

    def emit(self, initiator, reason, before, after, **context):
        event = copy.deepcopy(dict(monotonic=time.monotonic(), at=time.time(), initiator=initiator,
                     reason_code=reason, before=before, after=after, **context))
        self.last = event
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def run(self):
        while True:
            event = self.queue.get()
            try:
                self.handler.emit(logging.LogRecord('motion', logging.INFO, '', 0,
                    json.dumps(event, allow_nan=False), (), None))
            except (OSError, ValueError, TypeError):
                self.dropped += 1
            finally:
                self.queue.task_done()
