#!/usr/bin/env python3
"""Entry point for the local, current-conversation benchmark."""
import time

_STARTED_MS = time.time() * 1000  # Before heavy imports: the helper clock starts here.

from chat_benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    main(_STARTED_MS)
