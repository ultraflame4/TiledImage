import time


class ClockTimer:
    def __init__(self):
        self.start_ = 0
        self.last = 0

    def start(self):
        self.start_ = time.perf_counter()
        self.last = self.start_

    def getTimeSince(self):
        self.last = time.perf_counter() - self.last
        return self.last
