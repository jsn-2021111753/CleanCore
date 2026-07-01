"""Incremental, timeout-resilient timing records for experiment runs."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IncrementalTimingRecorder:
    """Record nested phases and persist progress while a process is running."""

    def __init__(self, path: Path, heartbeat_interval_sec: float = 5.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_interval_sec = max(0.01, float(heartbeat_interval_sec))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._stack: list[tuple[str, float, str]] = []
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._events: list[dict[str, object]] = []
        self._status = "running"
        self._error: Optional[str] = None
        self._started_perf = time.perf_counter()
        self._started_at = _utc_now()
        self._write_locked()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="timing-heartbeat", daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_sec):
            with self._lock:
                self._write_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        now = time.perf_counter()
        current_phase = self._stack[-1][0] if self._stack else None
        current_started_at = self._stack[-1][2] if self._stack else None
        current_elapsed = now - self._stack[-1][1] if self._stack else 0.0
        return {
            "status": self._status,
            "started_at_utc": self._started_at,
            "heartbeat_at_utc": _utc_now(),
            "elapsed_sec": float(now - self._started_perf),
            "current_phase": current_phase,
            "current_phase_started_at_utc": current_started_at,
            "current_phase_elapsed_sec": float(current_elapsed),
            "active_phase_stack": [name for name, _, _ in self._stack],
            "stage_timing_totals": {key: float(value) for key, value in self._totals.items()},
            "stage_timing_counts": {key: int(value) for key, value in self._counts.items()},
            "events": list(self._events[-200:]),
            "error": self._error,
        }

    def _write_locked(self) -> None:
        payload = self._snapshot_locked()
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def start_phase(self, name: str) -> None:
        phase = str(name)
        with self._lock:
            started_at = _utc_now()
            self._stack.append((phase, time.perf_counter(), started_at))
            self._events.append({"event": "start", "phase": phase, "at_utc": started_at})
            self._write_locked()

    def finish_phase(self, name: str) -> float:
        phase = str(name)
        with self._lock:
            if not self._stack or self._stack[-1][0] != phase:
                raise RuntimeError(f"Timing phase stack mismatch: expected {phase!r}.")
            _, started, _ = self._stack.pop()
            elapsed = time.perf_counter() - started
            self._totals[phase] = self._totals.get(phase, 0.0) + elapsed
            self._counts[phase] = self._counts.get(phase, 0) + 1
            self._events.append({"event": "finish", "phase": phase, "at_utc": _utc_now(), "elapsed_sec": float(elapsed)})
            self._write_locked()
            return float(elapsed)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.start_phase(name)
        try:
            yield
        finally:
            self.finish_phase(name)

    def add_duration(self, name: str, elapsed_sec: float) -> None:
        phase = str(name)
        elapsed = max(0.0, float(elapsed_sec))
        with self._lock:
            self._totals[phase] = self._totals.get(phase, 0.0) + elapsed
            self._counts[phase] = self._counts.get(phase, 0) + 1
            self._events.append({"event": "duration", "phase": phase, "at_utc": _utc_now(), "elapsed_sec": elapsed})
            self._write_locked()

    def totals(self) -> dict[str, float]:
        with self._lock:
            return {key: float(value) for key, value in self._totals.items()}

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {key: int(value) for key, value in self._counts.items()}

    def complete(self) -> None:
        self._finish("completed", None)

    def fail(self, error: object) -> None:
        self._finish("failed", str(error))

    def _finish(self, status: str, error: Optional[str]) -> None:
        with self._lock:
            self._status = str(status)
            self._error = error
            self._stack.clear()
            self._write_locked()
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(0.1, self.heartbeat_interval_sec * 2.0))


def mark_timing_timeout(path: Path, timeout_sec: Optional[float]) -> None:
    """Mark a child timing file after the parent runner enforces a timeout."""

    timing_path = Path(path)
    if not timing_path.exists():
        return
    try:
        payload = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    payload["status"] = "timeout"
    payload["timeout_sec"] = None if timeout_sec is None else float(timeout_sec)
    payload["timeout_marked_at_utc"] = _utc_now()
    temp = timing_path.with_name(timing_path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(timing_path)
