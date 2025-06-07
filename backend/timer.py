# timer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility micro‑stopwatch used throughout the project.

Adds an `.elapsed` attribute and returns the elapsed seconds from `stop()`, so
call‑sites (e.g. *run.py*) can access the number without re‑timing.
"""
import time

class TimerError(Exception):
    """Custom exception for mis‑use of the Timer class."""

class Timer:
    def __init__(self):
        self._start_time: float | None = None
        self.elapsed: float | None = None  # populated by .stop()

    # ------------------------------------------------------------------
    def start(self):
        """Start a new timer."""
        if self._start_time is not None:
            raise TimerError("Timer is already running. Use .stop() first.")
        self._start_time = time.perf_counter()
        self.elapsed = None  # reset

    # ------------------------------------------------------------------
    def stop(self) -> float:
        """Stop the timer and return the elapsed time in seconds."""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() first.")

        self.elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {self.elapsed:0.4f} seconds")
        return self.elapsed
