"""Keep QThreads alive until they stop and close task windows without blocking Qt."""
from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import QMessageBox


class TaskBinding(QObject):
    def __init__(self, window, attribute, callback):
        super().__init__(window)
        self.window = window
        self.attribute = attribute
        self.worker = getattr(window, attribute)
        self.callback = callback
        self.result = ()
        self.closing = False
        self.confirming = False
        self.finished_while_confirming = False
        self.completed = False
        self.worker.finished_signal.connect(self._remember_result, Qt.QueuedConnection)
        self.worker.finished.connect(self._finish, Qt.QueuedConnection)
        window.installEventFilter(self)
        window._task_binding = self

    def _remember_result(self, *args):
        self.result = args

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Close and self.worker.isRunning():
            event.ignore()
            if self.closing:
                return True
            self.confirming = True
            try:
                answer = QMessageBox.question(
                    self.window, "停止任务并关闭",
                    "任务正在运行，是否停止任务并在安全结束后关闭窗口？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
            finally:
                self.confirming = False
            if answer == QMessageBox.Yes:
                self.closing = True
                self.window.setProperty("closing_task", True)
                self.window.setWindowTitle(self.window.windowTitle() + " · 正在停止…")
                if self.worker.isRunning():
                    stop = getattr(self.worker, "stop", self.worker.requestInterruption)
                    stop()
            if self.finished_while_confirming:
                QTimer.singleShot(0, self._finish)
            return True
        return super().eventFilter(watched, event)

    @pyqtSlot()
    def _finish(self):
        if self.completed:
            return
        if self.confirming:
            self.finished_while_confirming = True
            return
        self.completed = True
        self.window.removeEventFilter(self)
        try:
            if not self.closing:
                self.callback(*self.result)
        finally:
            if getattr(self.window, self.attribute, None) is self.worker:
                setattr(self.window, self.attribute, None)
            if getattr(self.window, "_task_binding", None) is self:
                self.window._task_binding = None
            self.worker.deleteLater()
            if self.closing:
                QTimer.singleShot(0, self.window.close)
            self.deleteLater()


def bind_task(window, attribute: str, callback) -> TaskBinding:
    """Bind a worker's result to its actual QThread.finished event, before start()."""
    return TaskBinding(window, attribute, callback)
