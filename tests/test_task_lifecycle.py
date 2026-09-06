import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from Toolbox.task_support import bind_task


app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)


class SlowWorker(QThread):
    finished_signal = pyqtSignal(str, str)

    def run(self):
        self.finished_signal.emit("finished", "info")
        self.msleep(350)


def wait_until(predicate, timeout=3000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    if not predicate():
        raise AssertionError("Qt operation timed out")


class TaskLifecycleTests(unittest.TestCase):
    def start_window(self, callback):
        window = QWidget()
        window.worker = SlowWorker()
        bind_task(window, "worker", callback)
        window.show()
        worker = window.worker
        worker.start()
        QTest.qWait(20)
        self.addCleanup(lambda: window.close())
        self.addCleanup(lambda: worker.wait() if window.worker is worker else None)
        return window, worker

    def test_result_is_delivered_only_after_thread_has_stopped(self):
        results = []
        window, worker = self.start_window(lambda *args: results.append((worker.isRunning(), args)))
        self.assertEqual(results, [])
        self.assertIs(window.worker, worker)
        wait_until(lambda: bool(results))
        self.assertEqual(results, [(False, ("finished", "info"))])
        self.assertIsNone(window.worker)

    def test_close_stays_responsive_until_worker_stops(self):
        results = []
        window, worker = self.start_window(lambda *args: results.append(args))
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            before = time.monotonic()
            accepted = window.close()
            elapsed = time.monotonic() - before
        self.assertFalse(accepted)
        self.assertLess(elapsed, 0.15)
        self.assertTrue(window.isVisible())
        self.assertTrue(worker.isInterruptionRequested())
        wait_until(lambda: not window.isVisible())
        self.assertIsNone(window.worker)
        self.assertEqual(results, [])

    def test_declining_close_leaves_task_running(self):
        results = []
        window, worker = self.start_window(lambda *args: results.append(args))
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.assertFalse(window.close())
        self.assertTrue(window.isVisible())
        self.assertFalse(worker.isInterruptionRequested())
        wait_until(lambda: bool(results))
        self.assertTrue(window.close())

    def test_worker_can_finish_while_close_confirmation_is_open(self):
        results = []
        window, _worker = self.start_window(lambda *args: results.append(args))

        def delayed_confirmation(*args):
            QTest.qWait(450)
            return QMessageBox.Yes

        with patch.object(QMessageBox, "question", side_effect=delayed_confirmation):
            self.assertFalse(window.close())
        wait_until(lambda: not window.isVisible())
        self.assertIsNone(window.worker)
        self.assertEqual(results, [])
