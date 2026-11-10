"""Own and remove only the temporary files created for a benchmark sample."""
from __future__ import annotations

import json
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def temp_root():
    return Path(tempfile.gettempdir()) / "chat-benchmark-samples"


def cleanup(directory: Path, run_id: str, root: Path | None = None):
    root = (root or temp_root()).resolve()
    directory = Path(directory)
    # Check resolved absolute targets BEFORE any deletion, including junctions.
    resolved = directory.resolve()
    if resolved.parent != root or not resolved.name.startswith(run_id + "-") or directory.is_symlink():
        raise ValueError("Refusing to clean a path outside this benchmark's temporary directory.")
    if not resolved.exists():
        return
    owner = resolved / "owner.json"
    if owner.is_symlink() or json.loads(owner.read_text(encoding="utf-8")).get("run_id") != run_id:
        raise ValueError("Temporary directory ownership did not match.")
    children = list(resolved.iterdir())
    for path in children:
        if path.name not in ("sample.txt", "owner.json") or path.is_symlink() or path.resolve().parent != resolved or not path.is_file():
            raise ValueError("Unexpected content in benchmark temporary directory; left it untouched.")
    for path in children:
        path.unlink()
    resolved.rmdir()


@contextmanager
def sample_file(run_id: str, text: str, root: Path | None = None):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid run identity for temporary output.")
    root = (root or temp_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=run_id + "-", dir=root))
    (directory / "owner.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    try:
        path = directory / "sample.txt"
        # This is supplied model output, never synthesized reference content.
        path.write_text(text, encoding="utf-8", newline="")
        yield path
    finally:
        cleanup(directory, run_id, root)


# ---- File delivery: the model writes each sample with its own file tool. ----
# Some shells cut long commands (Git Bash on Windows stops near 8,191 characters),
# so the sample cannot always travel as shell input. The inbox lives inside the
# chat's working directory, where every client's file tool may write.

INBOX_PARENT = ".chat-benchmark-tmp"
MAX_SAMPLE_BYTES = 1024 * 1024


def inbox_name(run_id: str, index: int) -> str:
    return f"cbench-{run_id}-{index}.txt"


def _own_inbox(directory: Path, run_id: str) -> Path:
    resolved = Path(directory).resolve()
    if resolved.name != f"{run_id}-inbox" or resolved.parent.name != INBOX_PARENT or Path(directory).is_symlink():
        raise ValueError("Refusing to use a path that is not this run's sample inbox.")
    owner = resolved / "owner.json"
    if owner.is_symlink() or not owner.is_file() or json.loads(owner.read_text(encoding="utf-8")).get("run_id") != run_id:
        raise ValueError("Sample inbox ownership did not match.")
    return resolved


def create_inbox(base: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid run identity for the sample inbox.")
    directory = Path(base).resolve() / INBOX_PARENT / f"{run_id}-inbox"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "owner.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    return directory


def take_inbox_sample(directory: Path, run_id: str, index: int) -> str:
    """Read the model-written sample and delete it; nothing else is touched."""
    resolved = _own_inbox(directory, run_id)
    path = resolved / inbox_name(run_id, index)
    if not path.is_file() or path.is_symlink() or path.resolve().parent != resolved:
        raise ValueError(f"No sample file yet. First create {path.as_posix()} with your file-writing tool, "
                         "containing the literal reference rows, then run write-sample again.")
    if path.stat().st_size > MAX_SAMPLE_BYTES:
        path.unlink()
        raise ValueError("The sample file exceeds 1 MiB and was discarded.")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    path.unlink()
    return text


def remove_inbox(directory: Path, run_id: str) -> bool:
    """Delete this run's inbox (owner file and leftover samples only)."""
    try:
        resolved = _own_inbox(directory, run_id)
    except (OSError, ValueError):
        return False
    pattern = re.compile(rf"cbench-{run_id}-[1-9][0-9]*\.txt")
    children = list(resolved.iterdir())
    if any(not (c.name == "owner.json" or pattern.fullmatch(c.name)) or c.is_symlink() or not c.is_file() for c in children):
        return False  # Something else is in there; leave it all untouched.
    for child in children:
        child.unlink()
    resolved.rmdir()
    try:
        resolved.parent.rmdir()  # Only succeeds when no other run is using it.
    except OSError:
        pass
    return True


def recover_stale(root: Path | None = None, age_seconds=3600):
    """Recover abandoned owned files after a hard process kill, on a later run."""
    root = (root or temp_root()).resolve()
    removed = 0
    if root.exists():
        for directory in root.iterdir():
            match = re.fullmatch(r"([0-9a-f]{32})-[A-Za-z0-9_-]+", directory.name)
            if match and directory.is_dir() and time.time() - directory.stat().st_mtime > age_seconds:
                try:
                    cleanup(directory, match[1], root)
                    removed += 1
                except (OSError, ValueError):
                    pass
    return removed
