"""Per-user limits on the tools that spend LLM credit (hosted mode only).

In-memory, so counters reset when the server restarts; fine for one small instance.
"""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager


class RateLimitError(Exception):
    """Raised when a user exceeds a limit; the message is shown to the user."""


class SlidingWindow:
    """At most `max_events` per user in any `window_seconds` interval."""

    def __init__(self, max_events: int, window_seconds: int, what: str):
        self.max_events, self.window, self.what = max_events, window_seconds, what
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, user: str, n: int = 1) -> None:
        """Record n events for `user`, or raise RateLimitError without recording any."""
        now = time.time()
        with self._lock:
            events = self._events[user]
            while events and events[0] <= now - self.window:
                events.popleft()
            if len(events) + n > self.max_events:
                wait = int(events[0] + self.window - now) + 1 if events else self.window
                raise RateLimitError(
                    f"Limit reached: {self.max_events} {self.what} per {_duration(self.window)}. "
                    f"Try again in {_duration(wait)}.")
            events.extend([now] * n)


class OneAtATime:
    """At most one running job per user."""

    def __init__(self, what: str):
        self.what = what
        self._running: set[str] = set()
        self._lock = threading.Lock()

    @contextmanager
    def slot(self, user: str):
        with self._lock:
            if user in self._running:
                raise RateLimitError(f"You already have {self.what} running; wait for it to finish.")
            self._running.add(user)
        try:
            yield
        finally:
            with self._lock:
                self._running.discard(user)


def _duration(seconds: int) -> str:
    if seconds >= 3600:
        hours = round(seconds / 3600)
        return f"{hours} hour" + ("s" if hours != 1 else "")
    minutes = max(1, round(seconds / 60))
    return f"{minutes} minute" + ("s" if minutes != 1 else "")
