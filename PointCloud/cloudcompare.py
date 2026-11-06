"""External CloudCompare runner with per-task configuration and cancellation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable


class SegmentationCancelled(Exception):
    pass


@dataclass
class Execution:
    binary: Path
    cancelled: Callable = lambda: False
    notify: Callable = lambda data: None


_execution = ContextVar("forest_execution", default=None)
_spawn_lock = threading.Lock()


def find_binary(configured="") -> Path | None:
    """An explicit but invalid path must not silently select a different version."""
    configured = str(configured).strip() or os.environ.get("CLOUDCOMPARE_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_file() else None
    candidates = [shutil.which("CloudCompare"), shutil.which("cloudcompare")]
    for folder in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if folder:
            candidates.append(str(Path(folder) / "CloudCompare" / "CloudCompare.exe"))
    candidates += [r"D:\apps\CloudCompare\CloudCompare.exe",
                   "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare", "/usr/bin/cloudcompare"]
    return next((Path(p).resolve() for p in candidates if p and Path(p).is_file()), None)


@contextmanager
def execution(binary, cancelled=lambda: False, notify=lambda data: None):
    resolved = find_binary(binary)
    if resolved is None:
        raise FileNotFoundError("找不到 CloudCompare，请选择有效的 CloudCompare.exe。")
    token = _execution.set(Execution(resolved, cancelled, notify))
    try:
        yield
    finally:
        _execution.reset(token)


def check_cancel():
    current = _execution.get()
    if current is not None and current.cancelled():
        raise SegmentationCancelled("任务已取消；中间文件保留在本次输出目录。")


def report(data):
    check_cancel()
    current = _execution.get()
    if current is not None:
        current.notify(data)


def external_environment():
    # Never load the toolbox's Qt plugins or frozen runtime DLLs into CloudCompare.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("QT_") and k not in ("PYTHONHOME", "PYTHONPATH")}
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        prefix = os.path.normcase(os.path.abspath(bundled))
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                      if not os.path.normcase(os.path.abspath(p)).startswith(prefix))
    return env


def _popen(command, **kwargs):
    # PyInstaller sets a process DLL search directory, inherited by child processes.
    with _spawn_lock:
        if os.name != "nt" or not getattr(sys, "frozen", False):
            return subprocess.Popen(command, env=external_environment(), **kwargs)
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        buffer = ctypes.create_unicode_buffer(32768)
        kernel.GetDllDirectoryW(len(buffer), buffer)
        previous = buffer.value
        kernel.SetDllDirectoryW(None)
        try:
            return subprocess.Popen(command, env=external_environment(), **kwargs)
        finally:
            kernel.SetDllDirectoryW(previous or None)


def run(args, job, timeout=180):
    current = _execution.get()
    binary = current.binary if current else find_binary()
    if binary is None:
        raise FileNotFoundError("请设置 CloudCompare.exe 路径。")
    if not args or any(not isinstance(a, str) or "\0" in a for a in args):
        raise ValueError("CloudCompare 参数必须是非空字符串列表。")
    check_cancel()
    job = Path(job).resolve()
    job.mkdir(parents=True, exist_ok=True)
    log = job / "cloudcompare.log"
    command = [str(binary), "-SILENT", "-LOG_FILE", str(log), "-AUTO_SAVE", "OFF", "-NO_TIMESTAMP", *args]
    with (job / "stdout.log").open("wb") as stdout, (job / "stderr.log").open("wb") as stderr:
        process = _popen(command, cwd=job, stdout=stdout, stderr=stderr,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                check_cancel()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"CloudCompare 超过 {timeout} 秒，日志：{log}")
                time.sleep(.1)
            check_cancel()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    if process.returncode:
        error = (job / "stderr.log").read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"CloudCompare 执行失败（{process.returncode}）。日志：{log}\n{text[-2500:]}\n{error[-1000:]}")
    return {"exit_code": process.returncode, "command": command, "log": str(log), "log_tail": text[-5000:]}


def probe(binary, cancelled=lambda: False):
    with execution(binary, cancelled), tempfile.TemporaryDirectory(prefix="toolbox-cc-check-") as directory:
        result = run(["-HELP"], Path(directory), timeout=45)
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in Path(directory).glob("*.log"))
        commands = set(re.findall(r"(?<!\w)-([A-Z][A-Z0-9_]+)\s*:", text))
        missing = sorted({"CSF", "TREEISO"} - commands)
        return {"binary": result["command"][0], "available": not missing,
                "missing": missing, "commands": sorted(commands)}


def open_cloud(binary, path):
    resolved = find_binary(binary)
    if resolved is None:
        raise FileNotFoundError("CloudCompare 路径无效，请重新选择。")
    path = Path(path).resolve(strict=True)
    return _popen([str(resolved), str(path)], cwd=resolved.parent)
