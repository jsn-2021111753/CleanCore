"""Named wall-clock timers for experiments."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


@dataclass
class ExperimentTimer:
    durations: Dict[str, float] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        if name in self._starts:
            raise RuntimeError(f"Timer '{name}' is already running.")
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name not in self._starts:
            raise RuntimeError(f"Timer '{name}' is not running.")
        elapsed = time.perf_counter() - self._starts.pop(name)
        self.durations[name] = self.durations.get(name, 0.0) + elapsed
        return elapsed

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def as_dict(self) -> Dict[str, float]:
        return {f"{name}_time_sec": float(value) for name, value in self.durations.items()}

