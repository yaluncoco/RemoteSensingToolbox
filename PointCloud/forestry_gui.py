"""Single-tree segmentation tool, with persistent external-engine settings."""
from pathlib import Path

from PyQt5.QtCore import QSettings, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from PointCloud import cloudcompare
from Toolbox.task_support import bind_task
from Toolbox.version import APP_ID


STAGES = {
    "checking_output_runtime": "正在检查 PNG / SVG 绘图组件…",
    "rendering_results": "正在生成编号总览图和单木示例图…",
    "reading_bin": "检测到 CloudCompare BIN 内容，正在读取工作副本…",
    "converting_bin": "正在转换 BIN，保留坐标与属性…",
    "validating_bin": "正在逐点校验 BIN 转换结果…",
    "checking_plugins": "正在检测 CSF / TreeIso 插件…", "starting": "正在准备任务…",
    "sampling": "正在生成体素采样…", "ground_filtering": "正在分离地面…",
    "vegetation_filtering": "正在筛选植被…", "treeiso": "正在进行 TreeIso 单木分割…",
    "numbering_candidates": "正在筛选候选并编号…", "segmentation_complete": "分割完成，准备回写原始点…",
    "writing_full_resolution": "正在写入完整标注点云…", "validating_full_resolution": "正在逐点校验导出…",
    "indexing_tree_points": "正在建立单木索引…", "extracting_individual_trees": "正在逐树导出…",
    "complete": "导出与校验完成。",
}


class ForestryWorker(QThread):
    status_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(object, str)

    def __init__(self, binary, parameters=None):
        super().__init__()
        self.binary, self.parameters = binary, parameters

    def stop(self):
        self.requestInterruption()

    def run(self):
        try:
            if self.parameters is None:
                result = cloudcompare.probe(self.binary, self.isInterruptionRequested)
            else:
                from PointCloud.forestry_workflow import run_workflow
                result = run_workflow(binary=self.binary, cancelled=self.isInterruptionRequested,
                                      notify=self.status_signal.emit, **self.parameters)
            self.finished_signal.emit(result, "")
        except cloudcompare.SegmentationCancelled as exc:
            self.finished_signal.emit({"cancelled": True}, str(exc))
        except Exception as exc:
            self.finished_signal.emit(None, str(exc))


class _NoWheelMixin:
    def wheelEvent(self, event):
        event.ignore()


class NumberInput(_NoWheelMixin, QDoubleSpinBox):
    pass


class IntegerInput(_NoWheelMixin, QSpinBox):
    pass


class TerrainInput(_NoWheelMixin, QComboBox):
    pass


class ForestryWindow(QWidget):
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle("单木分割")
        self.resize(800, 820)
        self.setMinimumSize(640, 480)
        self.settings = settings if settings is not None else QSettings(APP_ID, APP_ID)
        self.worker = None
        self.result = None
        self.output_directory = None
        self._probing = False
        shell = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        shell.addWidget(scroll)

        heading = QLabel("单木分割")
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(heading)
        intro = QLabel("选择林地点云，自动分离地面、提取单木候选并编号。支持 LAS/LAZ 和单点云 BIN，结果另存到新的任务文件夹。")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        engine = QGroupBox("CloudCompare")
        engine_layout = QVBoxLayout(engine)
        saved = str(self.settings.value("forestry/cloudcompare_path", ""))
        detected = cloudcompare.find_binary(saved)
        self.binary_edit = QLineEdit(saved or (str(detected) if detected else ""))
        self.binary_edit.setPlaceholderText("选择已安装的 CloudCompare.exe")
        engine_layout.addLayout(self._path_row(self.binary_edit, "选择程序", self._choose_binary))
        self.probe_button = QPushButton("检测插件")
        self.probe_button.clicked.connect(self._probe)
        engine_layout.addWidget(self.probe_button, alignment=Qt.AlignLeft)
        self.engine_status = QLabel("需要 CSF 和 TreeIso 插件；设置的程序路径会自动保存。")
        self.engine_status.setWordWrap(True)
        engine_layout.addWidget(self.engine_status)
        self.binary_edit.editingFinished.connect(self._save_binary)
        layout.addWidget(engine)

        self.inputs = QGroupBox("输入与参数")
        form = QFormLayout(self.inputs)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择 LAS / LAZ / BIN 文件，按内容识别格式")
        form.addRow("林地点云", self._path_row(self.source_edit, "选择文件", self._choose_source))
        self.output_edit = QLineEdit(str(self.settings.value("forestry/output_parent", "")))
        form.addRow("输出位置", self._path_row(self.output_edit, "选择文件夹", self._choose_output))
        self.resolution = NumberInput()
        self.resolution.setRange(.05, .5)
        self.resolution.setSingleStep(.05)
        self.resolution.setValue(.2)
        self.resolution.setSuffix(" 米")
        form.addRow("采样间距", self.resolution)
        self.terrain = TerrainInput()
        for label, value in (("平坦地形", "FLAT"), ("起伏地形", "RELIEF"), ("陡坡", "SLOPE")):
            self.terrain.addItem(label, value)
        form.addRow("地形", self.terrain)
        self.min_height = NumberInput()
        self.min_height.setRange(.5, 40.)
        self.min_height.setValue(2.)
        self.min_height.setSuffix(" 米")
        form.addRow("最低候选树高（P95）", self.min_height)
        self.min_points = IntegerInput()
        self.min_points.setRange(20, 1000000)
        self.min_points.setValue(100)
        form.addRow("最少采样点数", self.min_points)
        self.metres = QCheckBox("我确认 XYZ 坐标的单位均为米")
        form.addRow(self.metres)
        self.extract = QCheckBox("导出每棵候选树的独立 LAZ 文件")
        self.extract.setChecked(True)
        form.addRow(self.extract)
        layout.addWidget(self.inputs)

        note = QLabel("自动结果均需复核，编号数量是候选数。当前流程筛选离地 0.35～40 米的点；树冠交叠、近地树基和稀疏边缘可能存在遗漏或混分。")
        note.setWordWrap(True)
        layout.addWidget(note)
        actions = QHBoxLayout()
        self.run_button = QPushButton("开始分割")
        self.run_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        shell.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        shell.addWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        shell.addWidget(self.status_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(80)
        self.log.document().setMaximumBlockCount(250)
        shell.addWidget(self.log)
        outputs = QHBoxLayout()
        self.folder_button = QPushButton("打开结果文件夹")
        self.folder_button.clicked.connect(self._open_folder)
        self.folder_button.setEnabled(False)
        self.preview_button = QPushButton("用 CloudCompare 查看结果")
        self.preview_button.clicked.connect(self._open_preview)
        self.preview_button.setEnabled(False)
        outputs.addWidget(self.folder_button)
        outputs.addWidget(self.preview_button)
        shell.addLayout(outputs)
        self._engine_group = engine

    def _path_row(self, edit, text, callback):
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        button = QPushButton(text)
        button.clicked.connect(callback)
        row.addWidget(button)
        return row

    def _choose_binary(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CloudCompare", self.binary_edit.text(), "程序 (*.exe);;所有文件 (*)")
        if path:
            self.binary_edit.setText(path)
            self._save_binary()

    def _save_binary(self):
        self.settings.setValue("forestry/cloudcompare_path", self.binary_edit.text().strip())
        self.engine_status.setText("路径已保存；运行前会重新检测 CSF / TreeIso。")

    def _choose_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择林地点云", self.source_edit.text(), "点云 (*.las *.laz *.bin);;所有文件 (*)")
        if path:
            self.source_edit.setText(path)
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(Path(path).parent))

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出位置", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _probe(self):
        self._save_binary()
        self._launch(None)

    def _start(self):
        source, parent = self.source_edit.text().strip(), self.output_edit.text().strip()
        if not Path(source).is_file():
            QMessageBox.warning(self, "输入无效", "请选择存在的 LAS / LAZ / BIN 文件。")
            return
        if not parent or not Path(parent).is_dir():
            QMessageBox.warning(self, "输出无效", "请选择输出文件夹。")
            return
        if not self.metres.isChecked():
            QMessageBox.warning(self, "坐标单位", "请确认 XYZ 坐标的单位均为米。")
            return
        self._save_binary()
        self.settings.setValue("forestry/output_parent", parent)
        self.result, self.output_directory = None, None
        self.preview_button.setEnabled(False)
        self.folder_button.setEnabled(False)
        self.log.clear()
        self._launch(dict(source=source, output_parent=parent, coordinates_in_metres=True,
                          resolution=self.resolution.value(), terrain=self.terrain.currentData(),
                          min_sample_points=self.min_points.value(), min_height=self.min_height.value(),
                          extract_trees=self.extract.isChecked()))

    def _launch(self, parameters):
        if self.worker is not None:
            return
        self._probing = parameters is None
        self.worker = ForestryWorker(self.binary_edit.text().strip(), parameters)
        self.worker.status_signal.connect(self._on_status)
        bind_task(self, "worker", self._finished)
        self._set_busy(True)
        self.status_label.setText("正在检测 CloudCompare…")
        self.worker.start()

    def _set_busy(self, busy):
        self.inputs.setEnabled(not busy)
        self._engine_group.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.preview_button.setEnabled(not busy and self.result is not None)
        self.progress.setRange(0, 0 if busy else 1)

    def _cancel(self):
        if self.worker:
            self.worker.stop()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("正在停止，等待当前计算安全结束…")

    def _on_status(self, data):
        text = STAGES.get(data["stage"], data["stage"])
        count = data.get("points", data.get("completed"))
        if count is not None and "total" in data:
            text += f" {count:,} / {data['total']:,}"
        if "output_directory" in data:
            self.output_directory = data["output_directory"]
            self.log.append("输出目录：" + self.output_directory)
            self.folder_button.setEnabled(True)
        self.status_label.setText(text)
        self.log.append(text)

    def _finished(self, result=None, error=""):
        self._set_busy(False)
        self.progress.setValue(0 if error else 1)
        if error:
            self.status_label.setText("任务已取消" if result and result.get("cancelled") else "任务未完成")
            self.log.append(error)
            if not (result and result.get("cancelled")):
                QMessageBox.warning(self, "单木分割", error)
            return
        if self._probing:
            text = ("检测通过：CSF 和 TreeIso 均可用。" if result["available"] else
                    "缺少命令行插件：" + ", ".join(result["missing"]))
            self.engine_status.setText(text)
            self.status_label.setText(text)
            self.log.append(text)
            return
        self.result = result
        self.output_directory = result["output_directory"]
        summary = result["segmentation"]
        text = (f"已完成：{summary['candidate_tree_count']} 个单木候选，"
                f"其中 {summary['flagged_candidate_count']} 个需重点复核。所有候选均需人工核验。")
        self.status_label.setText(text)
        self.log.append(text)
        self.folder_button.setEnabled(True)
        self.preview_button.setEnabled(True)

    def _open_folder(self):
        if self.output_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_directory))

    def _open_preview(self):
        if self.result:
            try:
                cloudcompare.open_cloud(self.result["binary"], self.result["export"]["preview_cloud"])
            except Exception as exc:
                QMessageBox.warning(self, "打开结果失败", str(exc))
