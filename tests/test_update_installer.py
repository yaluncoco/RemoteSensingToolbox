import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.request import urlopen, Request

from PyQt5.QtCore import QSettings, QEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget

from Toolbox.main import ToolboxWindow
from Toolbox.updater import InstallerAsset, ReleaseInfo, UpdateChecker, parse_release, _UpdateRedirectHandler
from Toolbox.update_installer import UpdateInstaller, _download_installer, verify_installer, launch_installer

app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)
CONTENT = b"MZ" + b"installer test bytes" * 40000
DIGEST = hashlib.sha256(CONTENT).hexdigest()
NAME = "RemoteSensingToolbox-9.0.0-windows-x64-setup.exe"
ASSET = InstallerAsset(NAME, "https://github.com/owner/releases/download/v9.0.0/" + NAME, len(CONTENT), DIGEST)


def payload(**changes):
    asset = dict(name=NAME, state="uploaded", size=len(CONTENT), digest="sha256:" + DIGEST,
                 browser_download_url="file:///must-not-be-used")
    asset.update(changes)
    return dict(tag_name="v9.0.0", assets=[asset])


def wait_until(predicate, timeout=5000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    if not predicate():
        raise AssertionError("Installer operation timed out")


class DownloadHandler(BaseHTTPRequestHandler):
    requests = 0

    def do_GET(self):
        type(self).requests += 1
        if self.path == "/missing":
            self.send_error(404)
            return
        content = CONTENT if self.path != "/corrupt" else b"MZ" + b"x" * (len(CONTENT) - 2)
        self.send_response(200)
        if self.path != "/unknown-size":
            size = len(content) + 1 if self.path == "/wrong-size" else len(content)
            self.send_header("Content-Length", str(size))
        self.end_headers()
        try:
            if self.path == "/short":
                self.wfile.write(content[:30])
            elif self.path == "/slow":
                time.sleep(.35)
                self.wfile.write(content)
            else:
                self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass


class InstallerMetadataTests(unittest.TestCase):
    def test_only_exact_installer_and_usable_digest_are_selected(self):
        release = parse_release(payload(), "owner/distribution")
        self.assertEqual(release.installer.url, f"https://github.com/owner/distribution/releases/download/v9.0.0/{NAME}")
        self.assertEqual(release.installer.sha256, DIGEST)
        for changes in [dict(name="../setup.exe"), dict(name=NAME.replace("9.0.0", "8.0.0")),
                        dict(state="new"), dict(size=0), dict(size=True), dict(size=3_000_000_000),
                        dict(digest=None), dict(digest="sha256:wrong"), dict(digest="md5:" + DIGEST)]:
            with self.subTest(changes=changes):
                self.assertIsNone(parse_release(payload(**changes)).installer)
        duplicate = payload()
        duplicate["assets"] *= 2
        with self.assertRaises(ValueError):
            parse_release(duplicate)

    def test_cached_release_keeps_verifiable_installer(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(str(Path(directory) / "cache.ini"), QSettings.IniFormat)
            checker = UpdateChecker(settings=settings, current_version="1.0.0")
            checker._pending_id = 1
            checker._finish(1, payload(), "")
            restored = UpdateChecker(settings=settings, current_version="1.0.0")
            self.assertEqual(restored.release.installer, checker.release.installer)
            checker.stop()
            restored.stop()

    def test_https_downgrade_is_rejected(self):
        with self.assertRaises(URLError):
            _UpdateRedirectHandler().redirect_request(Request(ASSET.url), None, 302, "", {}, "http://example.com/setup.exe")


class InstallerDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(2)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name) / "中文 更新缓存"
        self.installer = UpdateInstaller(cache_dir=self.root)
        self.addCleanup(self.cleanup_worker)
        self.endpoint = "/ok"
        opener = Mock()
        # Only the transport is redirected to a local test server. Production still
        # validates and constructs HTTPS GitHub URLs and rejects insecure redirects.
        opener.open.side_effect = lambda request, timeout: urlopen(self.base_url + self.endpoint, timeout=timeout)
        patcher = patch("Toolbox.update_installer.build_opener", return_value=opener)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.opener = opener

    def cleanup_worker(self):
        self.installer.stop()
        if self.installer._worker:
            self.installer._worker.join(3)
        app.processEvents()

    def download(self, asset=ASSET):
        self.installer.start(asset)
        wait_until(lambda: not self.installer.busy)

    def test_download_verifies_and_reuses_cache_without_network(self):
        self.download()
        self.assertEqual(self.installer.state, "ready")
        self.assertEqual(self.installer.path.read_bytes(), CONTENT)
        self.assertEqual(list(self.root.rglob("*.part")), [])
        self.opener.open.reset_mock()
        self.download()
        self.opener.open.assert_not_called()
        self.assertEqual(self.installer.state, "ready")

    def test_incomplete_corrupt_and_http_failures_can_be_retried(self):
        for endpoint in ("/short", "/corrupt", "/missing", "/wrong-size"):
            with self.subTest(endpoint=endpoint):
                self.endpoint = endpoint
                self.download()
                self.assertEqual(self.installer.state, "error")
                self.assertFalse(self.installer.path.exists())
                self.assertEqual(list(self.root.rglob("*.part")), [])
                self.endpoint = "/ok"
                self.download()
                self.assertEqual(self.installer.state, "ready")
                self.installer.path.unlink()

    def test_repeated_clicks_and_cancel_discard_late_results(self):
        self.endpoint = "/slow"
        before = DownloadHandler.requests
        self.installer.start(ASSET)
        worker = self.installer._worker
        self.installer.start(ASSET)
        wait_until(lambda: DownloadHandler.requests > before)
        self.installer.cancel()
        self.assertEqual(self.installer.state, "cancelled")
        self.endpoint = "/ok"
        self.download()
        worker.join(3)
        QTest.qWait(30)
        self.assertEqual(self.installer.state, "ready")
        self.assertEqual(self.installer.path.read_bytes(), CONTENT)
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_progress_and_unknown_content_length(self):
        self.endpoint = "/unknown-size"
        progress = []
        self.installer.changed.connect(lambda: progress.append((self.installer.state, self.installer.received)))
        self.download()
        self.assertEqual(self.installer.state, "ready")
        self.assertTrue(any(state == "verifying" for state, size in progress))
        self.assertEqual(self.installer.received, len(CONTENT))

    def test_tampered_cache_is_redownloaded_and_rechecked_before_install(self):
        self.download()
        self.installer.path.write_bytes(b"MZinvalid")
        self.download()
        self.assertEqual(self.installer.path.read_bytes(), CONTENT)
        install = Mock()
        self.installer.install_ready.connect(install)
        self.installer.path.write_bytes(b"MZinvalid")
        self.installer.prepare_install()
        wait_until(lambda: not self.installer.busy)
        install.assert_not_called()
        self.assertEqual(self.installer.state, "error")

    def test_disk_space_and_non_executable_content_are_rejected(self):
        with patch("Toolbox.update_installer.shutil.disk_usage", return_value=Mock(free=0)):
            self.download()
        self.assertIn("空间不足", self.installer.error)
        self.opener.open.assert_not_called()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / NAME
        path.write_bytes(b"not an executable")
        asset = replace(ASSET, size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "Windows 安装程序"):
            verify_installer(path, asset, threading.Event())

    def test_network_timeout_is_retryable(self):
        self.opener.open.side_effect = TimeoutError()
        self.download()
        self.assertEqual(self.installer.state, "error")
        self.assertIn("超时", self.installer.error)


class InstallerUiTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.window = ToolboxWindow(persist_settings=False, check_updates=False)
        self.window.update_checker.release = ReleaseInfo("9.0.0", "", "更新说明", ASSET)
        self.installer = self.window.update_installer
        self.installer.cache_dir = Path(self.scratch.name)
        self.window.show()
        self.window.open_updates()
        self.addCleanup(self.window.close)

    def ready(self):
        self.installer.asset = ASSET
        self.installer.path = Path(self.scratch.name) / NAME
        self.installer.path.write_bytes(CONTENT)
        self.installer.state = "ready"
        self.installer.changed.emit()

    def test_missing_installer_disables_download_and_ready_changes_action(self):
        self.window.update_checker.release = ReleaseInfo("9.0.0", "", "")
        self.window.update_checker.changed.emit()
        self.assertFalse(self.window.update_dialog.download_button.isEnabled())
        self.window.update_checker.release = ReleaseInfo("9.0.0", "", "", ASSET)
        self.ready()
        self.assertEqual(self.window.update_dialog.download_button.text(), "安装并重启")

    def test_active_task_blocks_install_without_stopping_it(self):
        self.ready()
        with patch.object(self.window, "_update_tasks_running", return_value=True), patch("Toolbox.main.launch_installer") as launch:
            self.window.install_update()
        launch.assert_not_called()
        self.assertIn("任务", self.installer.error)
        self.assertTrue(self.window.isVisible())
        self.assertEqual(self.installer.state, "ready")

    def test_closed_tool_with_live_visualizer_still_blocks_install(self):
        release = threading.Event()
        thread = threading.Thread(target=lambda: release.wait(3), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(release.set)
        self.window.open_windows["closed-point-tool"] = SimpleNamespace(visualization_thread=thread)
        self.window._tool_closed("closed-point-tool", thread)
        self.assertTrue(self.window._update_tasks_running())
        release.set()
        thread.join(3)
        self.assertFalse(self.window._update_tasks_running())

    def test_tool_destruction_never_reads_a_deleted_qt_widget(self):
        spec = SimpleNamespace(id="temporary-tool", title="临时工具")
        self.window._tool_loaded(spec, QWidget, None)
        self.window.open_windows[spec.id].close()
        app.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertNotIn(spec.id, self.window.open_windows)

    def test_launch_failure_keeps_app_open_and_success_closes_it(self):
        self.ready()
        with patch("Toolbox.main.launch_installer", side_effect=OSError("launch failed")):
            self.window.install_update()
            wait_until(lambda: bool(self.installer.error))
        self.assertTrue(self.window.isVisible())
        self.assertEqual(self.installer.state, "ready")
        with patch("Toolbox.main.launch_installer") as launch:
            self.window.install_update()
            wait_until(lambda: not self.window.isVisible())
        launch.assert_called_once_with(self.installer.path)

    def test_dialog_close_keeps_download_but_application_close_cancels(self):
        self.installer._pending = 1
        self.installer.state = "downloading"
        self.window.update_dialog.close()
        self.assertTrue(self.installer.busy)
        self.window.close()
        self.assertFalse(self.installer.busy)
        self.assertTrue(self.installer._cancel.is_set())

    def test_installer_launch_is_shell_free_and_waits_for_parent(self):
        path = Path(self.scratch.name) / "中文 路径" / NAME
        with patch("Toolbox.update_installer.sys.platform", "win32"), \
             patch("Toolbox.update_installer.sys.frozen", True, create=True), \
             patch("Toolbox.update_installer.subprocess.CREATE_NEW_PROCESS_GROUP", 512, create=True), \
             patch("Toolbox.update_installer.subprocess.Popen") as launch:
            launch_installer(path)
        arguments = launch.call_args.args[0]
        self.assertEqual(arguments[0], str(path))
        self.assertIn(f"/UPDATEPID={os.getpid()}", arguments)
        self.assertIn("/RESTARTAPP=1", arguments)
        self.assertIn("/NOCLOSEAPPLICATIONS", arguments)
        self.assertFalse(launch.call_args.kwargs.get("shell", False))
