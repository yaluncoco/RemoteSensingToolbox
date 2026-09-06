"""Small output transactions: never replace a completed file with partial work."""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(filename)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        for suffix in (".aux.xml", ".msk"):
            Path(str(temporary) + suffix).unlink(missing_ok=True)
