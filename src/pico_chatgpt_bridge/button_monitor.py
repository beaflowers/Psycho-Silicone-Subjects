"""Background Pico serial listener for live button state."""

from __future__ import annotations

import os
import threading

from .pico_serial import open_pico_serial, read_line
from .prompting import describe_button_state, parse_button_line


class ButtonMonitor:
    """Track the last selected button mode without blocking terminal input."""

    def __init__(self) -> None:
        self._selected_buttons: tuple[str, ...] = ()
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

    def _set_buttons(self, buttons: tuple[str, ...]) -> bool:
        """Latch the latest non-empty button state and report whether it changed."""
        if not buttons:
            return False

        with self._lock:
            if buttons == self._selected_buttons:
                return False
            self._selected_buttons = buttons
            return True

    def _run(self) -> None:
        """Read button events forever; fall back gracefully if unavailable."""
        try:
            with open_pico_serial() as connection:
                print(
                    "Pico serial connected. Button mode selection is live.",
                    flush=True,
                )
                while True:
                    line = read_line(connection)
                    if self._debug:
                        print(f"Raw Pico line: {line!r}", flush=True)
                    buttons = parse_button_line(line)
                    if self._set_buttons(buttons):
                        print(
                            f"Selected mode: {describe_button_state(buttons)}",
                            flush=True,
                        )
        except Exception as exc:
            print(
                f"Pico serial unavailable. Continuing without live buttons: {exc}",
                flush=True,
            )
