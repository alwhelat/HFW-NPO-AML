"""Progress messages: printed to the terminal and appended to a log file
(config.LOG_FILE) that can be watched live in VS Code, including messages
from parallel worker processes."""

from __future__ import annotations

import os
import time

import config


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    path = getattr(config, "LOG_FILE", None)
    if path:
        try:
            os.makedirs(os.path.dirname(str(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
