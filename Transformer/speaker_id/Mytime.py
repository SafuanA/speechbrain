import time

class MyTime:
    def __init__(self):
        self.start_time = time.time()

    def GetElapsed(self):
        return time.time() - self.start_time