"""Thread-safe shared-state utilities for the pipeline.

Queues carry consumable streams between exactly one producer and one
consumer; this module covers the other pattern — a "latest value" that
one thread writes and any number of threads may read without consuming.
Like `config`, this is a foundational utility any module may import; it
is not a pipeline stage.
"""

from __future__ import annotations

import threading
from typing import Any


class LatestValue:
    """Lock-protected holder for the most recent value of something.

    Reads do not clear the value — every reader always sees whatever was
    written last (e.g. the latest live transcript snippet), even if it
    has not changed since their previous read.
    """

    def __init__(self, initial: Any = "") -> None:
        self._lock = threading.Lock()
        self._value: Any = initial

    def set(self, value: Any) -> None:
        """Replace the held value."""
        with self._lock:
            self._value = value

    def get(self) -> Any:
        """Return the most recently set value."""
        with self._lock:
            return self._value
