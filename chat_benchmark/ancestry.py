"""Which agent CLI launched this helper? Read from the parent-process chain.

A skill installed for one client is often loaded by another (several clients read
each other's skill folders), so a hard-coded provider name cannot be trusted on
its own. The executable names above this process are direct evidence.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

MAX_DEPTH = 24


def _windows_table():
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        return {}
    table, entry = {}, Entry()
    entry.dwSize = ctypes.sizeof(Entry)
    try:
        more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            table[entry.th32ProcessID] = (entry.th32ParentProcessID, [entry.szExeFile])
            more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def _proc_table():
    table = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text(encoding="utf-8", errors="replace")
            parent = int(stat.rsplit(")", 1)[1].split()[1])
            names = [part for part in (directory / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")[:3] if part]
            table[int(directory.name)] = (parent, names or [stat.split("(", 1)[1].rsplit(")", 1)[0]])
        except (OSError, ValueError, IndexError):
            continue
    return table


def _ps_table():
    done = subprocess.run(["ps", "-Ao", "pid=,ppid=,args="], capture_output=True, text=True, timeout=3)
    table = {}
    for line in done.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2].split()[:3])
    return table


def process_table():
    try:
        if os.name == "nt":
            return _windows_table()
        return _proc_table() if Path("/proc/self/stat").exists() else _ps_table()
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return {}


def ancestors(table=None, pid=None):
    """Lower-cased executable/script stems from this process upwards, nearest first."""
    table = process_table() if table is None else table
    pid = os.getpid() if pid is None else pid
    chain, seen = [], set()
    for _ in range(MAX_DEPTH):
        if pid in seen or pid not in table:
            break
        seen.add(pid)
        parent, names = table[pid]
        chain.append({Path(name.replace("\\", "/")).stem.lower() for name in names})
        pid = parent
    return chain[1:]  # Skip this process itself.


def nearest(names_by_key: dict[str, tuple[str, ...]], chain=None):
    """Key of the client whose process name appears closest above this process."""
    chain = ancestors() if chain is None else chain
    for stems in chain:
        for key, names in names_by_key.items():
            if stems & set(names):
                return key
    return None
