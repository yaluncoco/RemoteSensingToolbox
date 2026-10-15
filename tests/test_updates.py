import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
import threading
import time
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QLabel

from Toolbox.main import ToolboxWindow
from Toolbox.updater import UpdateChecker, parse_release, version_key
from Toolbox.version import UPDATE_REPOSITORY


app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)


def wait_until(predicate, timeout=3000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    if not predicate():
        raise AssertionError("Update check timed out")


class ReleaseHandler(BaseHTTPRequestHandler):
    requests = Counter()

    def do_GET(self):
        self.requests[self.path] += 1
        if self.path == "/slow":
            time.sleep(.4)
        status = {"/missing": 404, "/limited": 429, "/denied": 403, "/broken": 500}.get(self.path, 200)
        version = {"/current": "v1.9.0", "/older": "v1.8.0"}.get(self.path, "v1.10.0")
        payload = {"tag_name": version, "body": "改进图像处理与点云工具。", "draft": False, "prerelease": False}
        payload["assets"] = [{"name": f"RemoteSensingToolbox-{version[1:]}-windows-x64-setup.exe",
                              "size": 500, "digest": "sha256:" + "a" * 64, "state": "uploaded"}]
        if self.path == "/preview":
            payload["prerelease"] = True
        body = b"invalid json" if self.path == "/invalid" else json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Cancellation deliberately disconnects slow requests.

    def log_message(self, *args):
        pass


class VersionTests(unittest.TestCase):
    def test_numeric_versions_do_not_sort_lexically(self):
        self.assertGreater(version_key("v1.10.0"), version_key("1.9.9"))
        self.assertGreater(version_key("2.0.0"), version_key("1.99.99"))
        self.assertEqual(version_key("v1.0"), version_key("1.0.0.0"))
        self.assertEqual(version_key("1.0.0+build.7"), version_key("1.0.0"))

    def test_invalid_and_unpublished_releases_are_not_updates(self):
        for payload in (None, [], {"tag_name": None}, {"tag_name": "latest"},
                        {"tag_name": "v1.1.0-rc.1"}, {"tag_name": "v1.1.0", "draft": True},
                        {"tag_name": "v1.1.0", "prerelease": True},
                        {"tag_name": "v1.1.0", "body": {"invalid": "notes"}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_release(payload)

    def test_download_destination_stays_in_the_configured_repository(self):
        release = parse_release({"tag_name": "v1.1.0", "html_url": "file:///untrusted", "body": "更新说明"})
        self.assertEqual(release.url, f"https://github.com/{UPDATE_REPOSITORY}/releases/tag/v1.1.0")
        self.assertEqual(release.notes, "更新说明")


class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ReleaseHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    def checker(self, path="/new", **kwargs):
        checker = UpdateChecker(current_version="1.9.0", api_url=self.base_url + path, **kwargs)
        self.addCleanup(checker.stop)
        return checker

    def complete_check(self, checker):
        checker.check()
        wait_until(lambda: not checker.checking)

    def window(self, automatic=False):
        window = ToolboxWindow(persist_settings=False, check_updates=automatic)
        self.addCleanup(window.close)
        window.update_checker.current_version = "1.9.0"
        window.update_checker.api_url = self.base_url + "/new"
        window.show()
        return window

    def test_new_current_and_older_releases(self):
        for path, expected in (("/new", True), ("/current", False), ("/older", False)):
            with self.subTest(path=path):
                checker = self.checker(path)
                self.complete_check(checker)
                self.assertEqual(checker.has_update, expected)
                self.assertEqual(checker.state, "available" if expected else "current")
                self.assertEqual(checker.error, "")

    def test_connection_and_response_failures_never_report_current(self):
        for path in ("/missing", "/limited", "/denied", "/broken", "/invalid", "/preview"):
            with self.subTest(path=path):
                checker = self.checker(path)
                self.complete_check(checker)
                self.assertEqual(checker.state, "error")
                self.assertFalse(checker.has_update)
                self.assertTrue(checker.error)

    def test_timeout_can_be_retried(self):
        checker = self.checker("/slow", timeout_ms=150)
        self.complete_check(checker)
        self.assertEqual(checker.state, "error")
        self.assertIn("超时", checker.error)
        checker.api_url = self.base_url + "/new"
        self.complete_check(checker)
        self.assertTrue(checker.has_update)
        self.assertEqual(checker.error, "")

    def test_repeated_clicks_share_one_request(self):
        before = ReleaseHandler.requests["/slow"]
        checker = self.checker("/slow")
        checker.check()
        checker.check()
        checker.check()
        wait_until(lambda: not checker.checking)
        self.assertEqual(ReleaseHandler.requests["/slow"] - before, 1)

    def test_late_result_after_timeout_does_not_overwrite_retry(self):
        release_late = threading.Event()
        late_sent = threading.Event()
        self.addCleanup(release_late.set)
        checker = self.checker("/current", timeout_ms=150)

        def delayed_result(signal, request_id, *args):
            release_late.wait(3)
            signal.emit(request_id, {"tag_name": "v99.0.0"}, "")
            late_sent.set()

        with patch("Toolbox.updater._request_release", new=delayed_result):
            self.complete_check(checker)
        self.assertIn("超时", checker.error)
        self.complete_check(checker)
        self.assertEqual(checker.state, "current")
        release_late.set()
        wait_until(late_sent.is_set)
        QTest.qWait(20)
        self.assertFalse(checker.has_update)
        self.assertEqual(checker.release.version, "1.9.0")

    def test_cached_new_version_survives_restart_and_network_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat)
            checker = self.checker(settings=settings)
            self.complete_check(checker)
            settings.sync()
            restored = self.checker("/missing", settings=QSettings(settings.fileName(), QSettings.IniFormat))
            self.assertTrue(restored.has_update)
            self.complete_check(restored)
            self.assertTrue(restored.has_update)
            self.assertEqual(restored.state, "error")
            upgraded = UpdateChecker(current_version="1.10.0", settings=settings)
            self.addCleanup(upgraded.stop)
            self.assertFalse(upgraded.has_update)

    def test_startup_checks_silently_and_leaves_only_the_update_icon_at_the_bottom(self):
        window = self.window(automatic=True)
        self.assertFalse(window.update_checker.has_update)
        wait_until(lambda: window.update_checker.has_update)
        self.assertIsNone(window.update_dialog)
        self.assertIn("1.10.0", window.update_button.toolTip())
        self.assertEqual(window.update_button.text(), "")
        self.assertIsNone(window.findChild(QLabel, "sidebarHint"))
        self.assertIsNone(window.findChild(QLabel, "sidebarStatus"))

    def test_click_opens_release_information_and_starts_in_app_download(self):
        window = self.window()
        QTest.mouseClick(window.update_button, Qt.LeftButton)
        self.assertTrue(window.update_dialog.isVisible())
        self.assertFalse(window.update_dialog.check_button.isEnabled())
        wait_until(lambda: not window.update_checker.checking)
        self.assertIn("新版本", window.update_dialog.heading.text())
        self.assertIn("1.10.0", window.update_dialog.versions.text())
        self.assertTrue(window.update_dialog.check_button.isEnabled())
        self.assertEqual(window.update_dialog.download_button.text(), "下载更新")
        with patch.object(window.update_installer, "start") as start:
            QTest.mouseClick(window.update_dialog.download_button, Qt.LeftButton)
        self.assertEqual(start.call_args.args[0], window.update_checker.release.installer)
        window.update_dialog.close()
        self.assertTrue(window.update_checker.has_update)

    def test_close_cancels_an_inflight_check_without_waiting(self):
        window = self.window()
        window.update_checker.api_url = self.base_url + "/slow"
        window.open_updates()
        QTest.qWait(30)
        before = time.monotonic()
        self.assertTrue(window.close())
        self.assertLess(time.monotonic() - before, .15)
        self.assertFalse(window.update_checker.checking)
        self.assertFalse(window.update_dialog.isVisible())
        QTest.qWait(450)
        self.assertIsNone(window.update_checker.release)


if __name__ == "__main__":
    unittest.main()
