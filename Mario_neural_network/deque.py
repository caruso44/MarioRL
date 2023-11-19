from collections import deque


class dictdeque(deque):
    def __init__(self, max_size):
        super().__init__(maxlen=max_size)
