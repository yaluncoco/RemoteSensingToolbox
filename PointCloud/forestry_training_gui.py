"""Background conversion of existing forestry output, without rerunning engines."""
from datetime import datetime
from pathlib import Path
import uuid

from PyQt5.QtCore import QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPushButton, QFileDialog, QMessageBox, QProgressBar)

from PointCloud.cloudcompare import SegmentationCancelled
from PointCloud.forestry_gui import TerrainInput, STAGES
from Toolbox.task_support import bind_task


class TrainingWorker(QThread):
    status_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(object, str)

    def __init__(self, source, target, format):
        super().__init__()
        self.source, self.target, self.format = source, target, format

    def stop(self):
        self.requestInterruption()

    def run(self):
        from PointCloud.forestry_training import export_training
        try:
            result = export_training(self.source, self.target, self.format,
                                     cancelled=self.isInterruptionRequested, notify=self.status_signal.emit)
            self.finished_signal.emit(result, "")
        except SegmentationCancelled as exc:
            self.finished_signal.emit({"cancelled": True}, str(exc))
        except Exception as exc:
            self.finished_signal.emit(None, str(exc))


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("点云训练数据转换")
        self.resize(740, 430)
        self.worker, self.result = None, None
        layout = QVBoxLayout(self)
        intro = QLabel("从完整单木分割结果导出训练数据，或将本工具的 TXT 与 NumPy 互转。"
                       "每行点属性、语义和实例标签保持对应；转换会逐点校验，不重新运行分割。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.source_edit, self.output_edit = QLineEdit(), QLineEdit()
        self.source_edit.setPlaceholderText("含 export.json 的分割目录，或含 manifest.json 的训练目录")
        for label, edit in (("已有结果", self.source_edit), ("输出位置", self.output_edit)):
            row = QHBoxLayout()
            row.addWidget(edit)
            button = QPushButton("选择文件夹")
            button.clicked.connect(lambda checked=False, target=edit: self._choose(target))
            row.addWidget(button)
            form.addRow(label, row)
        self.format = TerrainInput()
        for name, key in (("TXT", "txt"), ("NumPy", "numpy"), ("TXT ＋ NumPy", "both")):
            self.format.addItem(name, key)
        form.addRow("输出格式", self.format)
        layout.addLayout(form)
        note = QLabel("TXT 无表头，列顺序为 XYZ、强度、RGB、语义编号、实例编号、原始点索引、复核状态、"
                      "训练语义、训练实例、训练掩码。自动结果是伪标签；重点复核点默认忽略。"
                      "TXT 可能较大，NumPy 支持分块读取。每次转换创建新目录。")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.run_button, self.cancel_button, self.folder_button = (QPushButton(s) for s in ("开始转换", "取消", "打开结果"))
        self.cancel_button.setEnabled(False)
        self.folder_button.setEnabled(False)
        self.run_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        self.folder_button.clicked.connect(self._open)
        for button in (self.run_button, self.cancel_button, self.folder_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _choose(self, edit):
        value = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
        if value:
            edit.setText(value)

    def _start(self):
        if self.worker is not None:
            return
        source, parent = self.source_edit.text().strip(), self.output_edit.text().strip()
        if not source or not Path(source).is_dir() or not parent or not Path(parent).is_dir():
            QMessageBox.warning(self, "目录无效", "请选择已有结果目录和输出位置。")
            return
        target = Path(parent) / ("training_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        self.result = None
        self.folder_button.setEnabled(False)
        self.worker = TrainingWorker(source, target, self.format.currentData())
        self.worker.status_signal.connect(self._status)
        bind_task(self, "worker", self._finished)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("正在检查源数据完整性…")
        self.worker.start()

    def _status(self, data):
        self.status.setText(STAGES.get(data["stage"], data["stage"]) + f" {data['points']:,} / {data['total']:,}")

    def _cancel(self):
        if self.worker:
            self.worker.stop()
            self.cancel_button.setEnabled(False)

    def _finished(self, result=None, error=""):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0 if error else 1)
        if error:
            self.status.setText(error)
            if not (result and result.get("cancelled")):
                QMessageBox.warning(self, "转换未完成", error)
            return
        self.result = result
        self.folder_button.setEnabled(True)
        self.status.setText(f"已完成 {result['point_count']:,} 点，逐点对齐校验通过。\n" + result["directory"])

    def _open(self):
        if self.result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.result["directory"]))
