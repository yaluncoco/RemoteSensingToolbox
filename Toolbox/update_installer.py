"""Cancellable, verified Windows installer downloads and an explicit install handoff."""
from __future__ import annotations

import hashlib
from http.client import HTTPException
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from PyQt5.QtCore import QObject, QStandardPaths, Qt, pyqtSignal

from Toolbox.updater import InstallerAsset, _UpdateRedirectHandler
from Toolbox.version import APP_ID, APP_VERSION


class DownloadCancelled(Exception):
    pass


def _check_cancel(cancel):
    if cancel.is_set():
        raise DownloadCancelled()


def _emit(signal, *args):
    try:
        signal.emit(*args)
    except RuntimeError:
        pass  # The main window may already have closed.


def verify_installer(path, asset, cancel):
    """Verify size, SHA-256 and the Windows executable signature before any execution."""
    if path.stat().st_size != asset.size:
        raise ValueError("安装包大小不符，请重新下载。")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("下载内容不是 Windows 安装程序，请重新下载。")
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _check_cancel(cancel)
            digest.update(chunk)
    _check_cancel(cancel)
    if digest.hexdigest() != asset.sha256:
        raise ValueError("安装包 SHA-256 校验失败，请重新下载。")


def _download_installer(asset, target, cancel, progress):
    if urlsplit(asset.url).scheme != "https":
        raise ValueError("更新下载地址必须使用 HTTPS。")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        progress(asset.size, asset.size, "verifying")
        try:
            verify_installer(target, asset, cancel)
            return target
        except (ValueError, OSError):
            target.unlink(missing_ok=True)
    _check_cancel(cancel)
    if shutil.disk_usage(target.parent).free < asset.size + 10 * 1024 * 1024:
        raise OSError("更新缓存所在磁盘空间不足，请释放空间后重试。")
    descriptor, temporary = tempfile.mkstemp(prefix="download-", suffix=".part", dir=target.parent)
    partial = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = Request(asset.url, headers={"User-Agent": f"{APP_ID}/{APP_VERSION}",
                                                  "Accept": "application/octet-stream"})
            with build_opener(_UpdateRedirectHandler()).open(request, timeout=15) as response:
                if response.status != 200:
                    raise ValueError("更新服务器返回了异常响应，请重试。")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != asset.size:
                    raise ValueError("服务器返回的安装包大小不符，请重新检查更新。")
                received, last_progress = 0, 0.0
                progress(0, asset.size, "downloading")
                while True:
                    _check_cancel(cancel)
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > asset.size:
                        raise ValueError("下载内容超出安装包大小，已停止下载。")
                    output.write(chunk)
                    if time.monotonic() - last_progress >= .1:
                        progress(received, asset.size, "downloading")
                        last_progress = time.monotonic()
        _check_cancel(cancel)
        progress(received, asset.size, "verifying")
        verify_installer(partial, asset, cancel)
        _check_cancel(cancel)
        os.replace(partial, target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def launch_installer(path):
    """Start Inno Setup without a shell. It waits for this process before updating files."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        raise RuntimeError("请在 Windows 发行版中安装更新；当前运行方式仅支持下载。")
    arguments = [str(path), "/SP-", "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOCLOSEAPPLICATIONS", "/CURRENTUSER",
                 f"/UPDATEPID={os.getpid()}", "/RESTARTAPP=1"]
    # Inno Setup reuses the registered installation directory. A portable build
    # becomes an installed copy; do not overwrite the currently running portable tree.
    return subprocess.Popen(arguments, cwd=str(path.parent), close_fds=True,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


class UpdateInstaller(QObject):
    changed = pyqtSignal()
    install_ready = pyqtSignal()
    _progress = pyqtSignal(int, object, object, str)
    _finished = pyqtSignal(int, str, str, str)

    def __init__(self, parent=None, cache_dir=None):
        super().__init__(parent)
        base = Path(QStandardPaths.writableLocation(QStandardPaths.GenericCacheLocation))
        self.cache_dir = Path(cache_dir) if cache_dir else base / APP_ID / "updates"
        self.state = "idle"
        self.error = ""
        self.asset = None
        self.path = None
        self.received = self.total = 0
        self._sequence = 0
        self._pending = None
        self._cancel = threading.Event()
        self._worker = None
        self._progress.connect(self._on_progress, Qt.QueuedConnection)
        self._finished.connect(self._on_finished, Qt.QueuedConnection)

    @property
    def busy(self):
        return self._pending is not None

    def start(self, asset: InstallerAsset):
        if self.busy:
            return
        if Path(asset.name).name != asset.name or not asset.name.endswith("-setup.exe"):
            self.fail("安装包名称无效，请重新检查更新。")
            return
        self.asset = asset
        self.path = self.cache_dir / asset.sha256 / asset.name
        self._begin("download")

    def prepare_install(self):
        if self.busy or self.state != "ready" or self.asset is None or self.path is None:
            return
        self._begin("install")

    def _begin(self, purpose):
        self._sequence += 1
        self._pending = self._sequence
        self._cancel = threading.Event()
        self.error = ""
        self.total = self.asset.size
        self.received = 0 if purpose == "download" else self.total
        self.state = "downloading" if purpose == "download" else "verifying"
        self.changed.emit()
        request_id, asset, path, cancel = self._pending, self.asset, self.path, self._cancel
        progress_signal, result_signal = self._progress, self._finished

        def work():
            error, outcome = "", purpose
            try:
                if purpose == "install":
                    verify_installer(path, asset, cancel)
                else:
                    _download_installer(asset, path, cancel,
                        lambda received, total, phase: _emit(progress_signal, request_id, received, total, phase))
            except DownloadCancelled:
                outcome = "cancelled"
            except (OSError, ValueError, HTTPException) as exc:
                logging.info("Installer download/verification failed: %s", exc)
                if isinstance(exc, HTTPError):
                    error = f"安装包下载失败（HTTP {exc.code}），请重试或重新检查更新。"
                    exc.close()
                elif isinstance(exc, (URLError, TimeoutError)):
                    error = "下载连接失败或超时，请检查网络后重试。"
                else:
                    error = str(exc) or "更新下载失败，请重试。"
                outcome = "error"
            _emit(result_signal, request_id, outcome, str(path), error)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _on_progress(self, request_id, received, total, phase):
        if request_id == self._pending:
            self.received, self.total, self.state = received, total, phase
            self.changed.emit()

    def _on_finished(self, request_id, outcome, path, error):
        if request_id != self._pending:
            return
        self._pending = None
        self.error = error
        if outcome == "error":
            self.state = "error"
        elif outcome == "cancelled":
            self.state = "cancelled"
        else:
            self.state = "ready"
            self.path = Path(path)
            self.received = self.total
        self.changed.emit()
        if outcome == "install":
            self.install_ready.emit()

    def cancel(self):
        if self.busy:
            self._cancel.set()
            self._pending = None  # Late results cannot revive a cancelled download.
            self.state = "cancelled"
            self.changed.emit()

    def fail(self, message):
        self.error = message
        # Keep a successfully downloaded installer available for another install attempt.
        self.state = "ready" if self.path and self.path.is_file() else "error"
        self.changed.emit()

    def stop(self):
        self.cancel()
