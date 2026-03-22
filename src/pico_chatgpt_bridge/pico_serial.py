"""Helpers for reading USB serial data from a Pico."""

from __future__ import annotations

import os

import serial
import serial.tools.list_ports


def find_serial_port() -> str:
    """Return a serial port path for the Pico, using env config or heuristics."""
    configured_port = os.environ.get("PICO_SERIAL_PORT")
    if configured_port:
        return configured_port

    for port in serial.tools.list_ports.comports():
        descriptor = " ".join(
            part for part in (port.device, port.description, port.manufacturer) if part
        ).lower()
        if "circuitpython" in descriptor or "pico" in descriptor:
            return port.device

    raise RuntimeError(
        "Could not find a Pico serial port automatically. Set PICO_SERIAL_PORT."
    )


def open_pico_serial(port: str | None = None, baudrate: int = 115200) -> serial.Serial:
    """Open the Pico serial connection."""
    target_port = port or find_serial_port()
    return serial.Serial(target_port, baudrate=baudrate, timeout=1)


def read_line(connection: serial.Serial) -> str:
    """Read one non-empty line from the Pico."""
    while True:
        raw_line = connection.readline()
        if not raw_line:
            continue

        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        return line
