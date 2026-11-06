"""Small file helpers shared by the forestry pipeline; no MCP dependency."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData

from PointCloud.cloudcompare import check_cancel


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            check_cancel()
            digest.update(block)
    return digest.hexdigest()


def read_cloud(path):
    vertex = PlyData.read(path, mmap=False)["vertex"].data
    xyz = np.column_stack([vertex[n] for n in "xyz"]).astype(np.float64)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("CloudCompare 输出为空或包含无效坐标。")
    fields = {n: vertex[n] for n in vertex.dtype.names if n not in ("x", "y", "z")}
    return SimpleNamespace(xyz=xyz, fields=fields)


@contextmanager
def project_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if stream.read(1) == b"":
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def publish_new(temp, target):
    """Publish only inside a newly created task directory, without overwriting."""
    if os.name == "nt":
        os.rename(temp, target)  # Windows rename fails if the target exists; works on exFAT too.
    else:
        os.link(temp, target)
        Path(temp).unlink()
