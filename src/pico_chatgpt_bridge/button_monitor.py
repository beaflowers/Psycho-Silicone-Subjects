"""Background Pico serial listener for live button state."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import os
import threading

from .pico_serial import open_pico_serial, read_line
from .prompting import parse_button_line


class ButtonMonitor:
    """Track the last selected button mode without blocking terminal input."""

    def __init__(
        self,
        on_event: Callable[[tuple[str, ...]], None] | None = None,
        on_status: Callable[[str, str], None] | None = None,
        on_raw_line: Callable[[str], None] | None = None,
    ) -> None:
        self._selected_buttons: tuple[str, ...] = ()
        self._events: deque[tuple[str, ...]] = deque()
        self._on_event = on_event
        self._on_status = on_status
        self._on_raw_line = on_raw_line
        self._debug = os.environ.get("PICO_DEBUG_SERIAL", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start listening for Pico updates in a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get_buttons(self) -> tuple[str, ...]:
        """Return the last latched non-empty button selection."""
        with self._lock:
            return self._selected_buttons

    def drain_events(self) -> tuple[tuple[str, ...], ...]:
        """Return all queued button press events since the last poll."""
        with self._lock:
            events = tuple(self._events)
            self._events.clear()
            return events

    def _record_event(self, buttons: tuple[str, ...]) -> bool:
        """Queue every non-empty button event and store the latest selection."""
        if not buttons:
            return False

        with self._lock:
            self._selected_buttons = buttons
            self._events.append(buttons)
            return True

    def _run(self) -> None:
        """Read button events forever; fall back gracefully if unavailable."""
        try:
            with open_pico_serial() as connection:
                print(
                    "Pico serial connected. Button mode selection is live.",
                    flush=True,
                )
                if self._on_status is not None:
                    self._on_status("connected", "Pico serial connected. Button mode selection is live.")
                while True:
                    line = read_line(connection)
                    if self._on_raw_line is not None:
                        self._on_raw_line(line)
                    if self._debug:
                        print(f"Raw Pico line: {line!r}", flush=True)
                    buttons = parse_button_line(line)
                    if self._record_event(buttons):
                        if self._on_event is not None:
                            self._on_event(buttons)
        except Exception as exc:
            if self._on_status is not None:
                self._on_status("error", str(exc))
            print(
                f"Pico serial unavailable. Continuing without live buttons: {exc}",
                flush=True,
            )
