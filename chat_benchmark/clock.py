"""Helper-clock ("bracket") timing that works for any client.

Between two consecutive helper commands the only thing that happens is one model
response: the client returns the previous tool result, the model generates the
next tool call, and the client launches the helper again. The interval from the
previous helper's exit to this helper's process start therefore brackets that
response. It includes client and tool-dispatch overhead, which is roughly
constant per stage and cancels in the length-sweep slope (see estimate.py).
"""
from __future__ import annotations

import os
import time

MAX_BRACKET_MS = 30 * 60 * 1000


def process_start_ms(fallback_ms: float | None = None) -> float:
    """Wall-clock creation time of this process; excludes interpreter start-up."""
    fallback_ms = time.time() * 1000 if fallback_ms is None else fallback_ms
    if os.name != "nt":
        return fallback_ms
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(kernel32.GetCurrentProcess(), ctypes.byref(created),
                                        ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            return fallback_ms
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        value = ticks / 10_000 - 11_644_473_600_000  # 100 ns ticks since 1601 -> Unix ms
        # Trust it only when it is a plausible moment shortly before the fallback.
        return value if 0 <= fallback_ms - value < 60_000 else fallback_ms
    except (OSError, AttributeError, ValueError):
        return fallback_ms


def bracket(clock: dict | None, invoked_ms: float) -> float | None:
    """Milliseconds from the previous helper exit to this helper's start, if sane."""
    previous = (clock or {}).get("exit_ms")
    if not isinstance(previous, (int, float)):
        return None
    value = invoked_ms - previous
    return value if 0 < value < MAX_BRACKET_MS else None


def stamp(command: str) -> dict:
    return {"exit_ms": time.time() * 1000, "command": command}
