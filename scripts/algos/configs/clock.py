from __future__ import annotations

import threading


class GlobalStepClock:
    """Process-global monotonic step counter.

    This is not a torch module.
    This is not saved in checkpoints.
    This is shared only inside one Python process.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._step = 0

    def reset(self, value: int = 0) -> None:
        value = int(value)
        if value < 0:
            raise ValueError(f"Global step must be non-negative, got {value}")
        with self._lock:
            self._step = value

    def tick(self, n: int = 1) -> int:
        n = int(n)
        if n <= 0:
            raise ValueError(f"tick increment must be positive, got {n}")
        with self._lock:
            self._step += n
            return self._step

    def get(self) -> int:
        with self._lock:
            return int(self._step)


_CLOCK = GlobalStepClock()


def reset_global_step(value: int = 0) -> None:
    _CLOCK.reset(value)


def tick_global_step(n: int = 1) -> int:
    return _CLOCK.tick(n)


def get_global_step() -> int:
    return _CLOCK.get()