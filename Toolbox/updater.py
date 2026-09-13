"""Asynchronous release checks; application data never leaves this machine."""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from http.client import HTTPException
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PyQt5.QtCore import QObject, QSettings, QTimer, Qt, pyqtSignal

from Toolbox.version import APP_ID, APP_VERSION, UPDATE_REPOSITORY


class _UpdateRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if request.full_url.startswith("https:") and not new_url.startswith("https:"):
            raise URLError("Refusing an insecure update redirect")
        return super().redirect_request(request, response, code, message, headers, new_url)


def _request_release(result_signal, request_id, api_url, current_version, timeout):
    """Do bounded, read-only networking without keeping the application alive."""
    payload, error = None, ""
    try:
        request = Request(api_url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"{APP_ID}/{current_version}",
        })
        with build_opener(_UpdateRedirectHandler()).open(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError("Unexpected update response")
            data = response.read(1_000_001)
        if len(data) > 1_000_000:
            raise ValueError("Release response is too large")
        payload = json.loads(data)
    except HTTPError as exc:
        if exc.code == 404:
            error = "暂时无法读取更新信息，发布地址可能尚未开放。可稍后重试或打开发布页面。"
        elif exc.code in (403, 429):
            error = "更新服务暂时无法访问，请稍后重试或打开发布页面。"
        else:
            error = "无法连接更新服务，请检查网络后重试。"
        exc.close()
    except TimeoutError:
        error = "检查更新超时，请检查网络后重试。"
    except URLError as exc:
        error = ("检查更新超时，请检查网络后重试。" if isinstance(exc.reason, TimeoutError)
                 else "无法连接更新服务，请检查网络后重试。")
    except (ValueError, TypeError):
        error = "更新信息暂时不可用，请稍后重试或打开发布页面。"
    except (OSError, HTTPException):
        error = "无法连接更新服务，请检查网络后重试。"
    try:
        result_signal.emit(request_id, payload, error)
    except RuntimeError:
        pass  # The owning Qt window may already have been destroyed.


def version_key(version: str) -> tuple[int, ...]:
    """Compare numeric release versions, including v prefixes and build metadata."""
    if not isinstance(version, str):
        raise ValueError("Invalid release version")
    match = re.fullmatch(r"[vV]?(\d+(?:\.\d+){0,3})(?:\+[0-9A-Za-z.-]+)?", version.strip())
    if not match:
        raise ValueError("Invalid stable release version")
    parts = tuple(int(part) for part in match.group(1).split("."))
    return parts + (0,) * (4 - len(parts))


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    url: str
    notes: str


def parse_release(payload: object, repository: str = UPDATE_REPOSITORY) -> ReleaseInfo:
    if not isinstance(payload, dict):
        raise ValueError("Expected a release object")
    if payload.get("draft", False) is not False or payload.get("prerelease", False) is not False:
        raise ValueError("Expected a published stable release")
    tag = payload.get("tag_name", "")
    version_key(tag)
    tag = tag.strip()
    notes = payload.get("body") or ""
    if not isinstance(notes, str):
        raise ValueError("Invalid release notes")
    # Build the destination from the configured repository, never from remote HTML.
    return ReleaseInfo(
        version=tag.lstrip("vV"),
        url=f"https://github.com/{repository}/releases/tag/{quote(tag, safe='')}",
        notes=notes[:30000],
    )


class UpdateChecker(QObject):
    changed = pyqtSignal()
    _result_ready = pyqtSignal(int, object, str)

    def __init__(self, parent=None, *, settings: QSettings | None = None,
                 current_version=APP_VERSION, repository=UPDATE_REPOSITORY,
                 api_url=None, timeout_ms=10000):
        super().__init__(parent)
        self.current_version = current_version
        self.repository = repository
        self.releases_url = f"https://github.com/{repository}/releases/latest"
        self.api_url = api_url or f"https://api.github.com/repos/{repository}/releases/latest"
        self.settings = settings
        self.state = "idle"
        self.error = ""
        self.release: ReleaseInfo | None = None
        self._cache_key = f"updates/{repository}/latest_release"
        self._pending_id = None
        self._request_sequence = 0
        self._result_ready.connect(self._finish, Qt.QueuedConnection)
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.setInterval(timeout_ms)
        self._timeout.timeout.connect(self._abort_timed_out_request)
        self._startup = QTimer(self)
        self._startup.setSingleShot(True)
        self._startup.timeout.connect(self.check)
        self._periodic = QTimer(self)
        self._periodic.setInterval(6 * 60 * 60 * 1000)
        self._periodic.timeout.connect(self.check)
        if self.settings is not None:
            cached = self.settings.value(self._cache_key, "")
            if cached:
                try:
                    self.release = parse_release(json.loads(cached), self.repository)
                    if self.has_update:
                        self.state = "available"
                except (ValueError, TypeError):
                    logging.debug("Ignoring invalid cached release information")

    @property
    def has_update(self) -> bool:
        return self.release is not None and version_key(self.release.version) > version_key(self.current_version)

    @property
    def checking(self) -> bool:
        return self._pending_id is not None

    def start_automatic_checks(self) -> None:
        self._startup.start(1200)
        self._periodic.start()

    def check(self) -> None:
        if self.checking:
            return
        self._startup.stop()
        self.error = ""
        self.state = "checking"
        self._request_sequence += 1
        self._pending_id = self._request_sequence
        self._timeout.start()
        self.changed.emit()
        # Daemon threads perform no writes and never hold up application exit.
        # Request IDs discard late results after timeout, retry, or window close.
        worker = threading.Thread(
            target=_request_release,
            args=(self._result_ready, self._pending_id, self.api_url,
                  self.current_version, self._timeout.interval() / 1000),
            daemon=True,
        )
        worker.start()

    def _abort_timed_out_request(self) -> None:
        if self._pending_id is not None:
            self._finish(self._pending_id, None, "检查更新超时，请检查网络后重试。")

    def _finish(self, request_id, payload, error) -> None:
        if request_id != self._pending_id:
            return
        self._pending_id = None
        self._timeout.stop()
        self.error = error
        if not self.error:
            try:
                release = parse_release(payload, self.repository)
            except (ValueError, TypeError):
                self.error = "更新信息暂时不可用，请稍后重试或打开发布页面。"
            else:
                self.release = release
                if self.settings is not None:
                    self.settings.setValue(self._cache_key, json.dumps({
                        "tag_name": payload["tag_name"], "body": release.notes,
                    }, ensure_ascii=False))
        if self.error:
            self.state = "error"
            logging.info("Update check failed: %s", self.error)
        else:
            self.state = "available" if self.has_update else "current"
        self.changed.emit()

    def stop(self) -> None:
        """Stop checking and discard pending results without delaying window close."""
        self._startup.stop()
        self._periodic.stop()
        self._timeout.stop()
        self._pending_id = None
        if self.state == "checking":
            self.state = "available" if self.has_update else "idle"
