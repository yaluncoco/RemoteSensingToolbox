"""Open3D file I/O with Windows Unicode paths and atomic output replacement."""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from Toolbox.file_io import atomic_output


@contextmanager
def _ascii_workspace():
    candidates = [Path(tempfile.gettempdir()), Path(os.environ.get("PUBLIC", "C:/Users/Public")) / "Documents"]
    workspace = None
    for candidate in candidates:
        if not str(candidate).isascii() or not candidate.is_dir():
            continue
        try:
            workspace = tempfile.TemporaryDirectory(prefix="remote-sensing-", dir=candidate)
            break
        except OSError:
            continue
    if workspace is None:
        raise OSError("点云库需要一个可写的英文临时目录来处理中文路径。请检查系统临时目录权限。")
    with workspace as directory:
        yield Path(directory)


def read_point_cloud(filename, **kwargs):
    import open3d as o3d

    path = Path(filename)
    if os.name == "nt" and not str(path).isascii():
        with _ascii_workspace() as directory:
            temporary = directory / ("cloud" + path.suffix)
            shutil.copyfile(path, temporary)
            return o3d.io.read_point_cloud(str(temporary), **kwargs)
    return o3d.io.read_point_cloud(str(path), **kwargs)


def write_point_cloud(filename, cloud, **kwargs):
    import open3d as o3d

    with atomic_output(filename) as output:
        if os.name == "nt" and not str(output).isascii():
            with _ascii_workspace() as directory:
                temporary = directory / ("cloud" + output.suffix)
                if not o3d.io.write_point_cloud(str(temporary), cloud, **kwargs):
                    raise OSError(f"点云保存失败：{filename}")
                shutil.copyfile(temporary, output)
        elif not o3d.io.write_point_cloud(str(output), cloud, **kwargs):
            raise OSError(f"点云保存失败：{filename}")
    return True
