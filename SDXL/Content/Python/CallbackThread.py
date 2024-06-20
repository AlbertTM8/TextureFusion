import threading

class ThreadWithCallback(threading.Thread):
    def __init__(self, target, callback, args=()):
        super().__init__()
        self._target = target
        self._args = args
        self.callback = callback

    def run(self):
        try:
            if self._target:
                self._target(*self._args)
        finally:
            if self.callback:
                self.callback()