"""Forestry measurements as a separate point-cloud tool."""
from pathlib import Path

from PyQt5.QtCore import QSettings, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QTabWidget, QVBoxLayout, QWidget)

from PointCloud import cloudcompare
from PointCloud.forest_metrics import MetricOptions
from PointCloud.forestry_gui import NumberInput, IntegerInput, TerrainInput
from Toolbox.task_support import bind_task
from Toolbox.soft_widgets import SoftScrollArea
from Toolbox.version import APP_ID


STAGES = {"reading":"读取点云与编号", "ground":"建立地形并计算离地高度",
          "stems":"检测树干并校验胸径", "grouping":"复用已有单木编号",
          "measuring":"逐树提取参数", "writing":"生成并校验 CSV、Excel 和分布图",
          "export":"导出并校验原始属性", "complete":"提取完成",
          "sampling":"准备地面过滤采样", "reading_bin":"读取 BIN 工作副本",
          "validating_bin":"校验 BIN 格式转换"}
STAGES["scene"] = "建立可交互单木对象与点云索引"

QUALITY_TEXT = {"ok":"通过", "insufficient_points":"切片点不足", "no_circle":"未拟合出圆",
    "insufficient_inliers":"有效内点不足", "weak_inliers":"内点支持不足", "partial_arc":"圆弧覆盖不足",
    "radius_at_limit":"半径触及边界", "no_low_stem_support":"低处树干点不足",
    "ground_support_incomplete":"部分点缺少地面支持", "ground_unsupported":"缺少地面支持",
    "sparse_tree":"单木点数偏少", "low_height":"树高偏低", "top_outliers_or_sparse_top":"顶部稀疏或有离群点",
    "possible_lean_or_branches":"可能倾斜或混入枝条", "crown_base_unresolved":"冠基部无法确定",
    "crown_slices_incomplete":"树冠切片覆盖不足"}


class MetricsWorker(QThread):
    status_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(object, str)

    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters

    def stop(self):
        self.requestInterruption()

    def run(self):
        try:
            from PointCloud.forest_metrics_workflow import run_metrics
            self.finished_signal.emit(run_metrics(**self.parameters, cancelled=self.isInterruptionRequested,
                                                   notify=self.status_signal.emit), "")
        except cloudcompare.SegmentationCancelled as exc:
            self.finished_signal.emit({"cancelled":True}, str(exc))
        except Exception as exc:
            self.finished_signal.emit(None, str(exc))


class ForestMetricsWindow(QWidget):
    continue_requested = pyqtSignal(str, object)

    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle("林业参数提取")
        self.resize(1440, 940)
        self.setMinimumSize(1000, 650)
        self.setObjectName("metricsWindow")
        self.settings = settings if settings is not None else QSettings(APP_ID, APP_ID)
        self.worker = self.result = self.output_directory = None
        self.scene_worker = None
        self._segmentation_result = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        header = QHBoxLayout()
        heading = QLabel("林业参数 · 实例工作台")
        heading.setObjectName("pageTitle")
        header.addWidget(heading)
        header.addStretch()
        self.new_button = QPushButton("新建参数分析")
        self.new_button.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.open_result_button = QPushButton("打开已有结果")
        self.open_result_button.setObjectName("secondaryButton")
        self.open_result_button.clicked.connect(self._open_result)
        header.addWidget(self.new_button)
        header.addWidget(self.open_result_button)
        layout.addLayout(header)
        self.tabs = QTabWidget()
        from PointCloud.forest_workspace import ForestWorkspace
        self.workspace = ForestWorkspace(self, QUALITY_TEXT)
        self.workspace.status_changed.connect(lambda text: self.status_label.setText(text))
        self.workspace.busy_changed.connect(self._scene_busy)
        self.workspace.loaded.connect(self._scene_loaded)
        self.tabs.addTab(self.workspace, "实例与参数")
        scroll = SoftScrollArea()
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidgetResizable(True)
        self.inputs = QWidget()
        body = QVBoxLayout(self.inputs)
        scroll.setWidget(self.inputs)
        self.tabs.addTab(scroll, "提取设置")
        layout.addWidget(self.tabs, 1)
        heading = QLabel("新建林业参数分析")
        heading.setStyleSheet("font-size:22px;font-weight:600")
        body.addWidget(heading)
        intro = QLabel("提取树高、胸径、冠幅、冠面积、冠体积及估算材积。可直接复用单木分割的编号，结果保存到新的任务文件夹。")
        intro.setWordWrap(True)
        body.addWidget(intro)
        group = QGroupBox("输入与处理方式")
        form = QFormLayout(group)
        self.source_edit, self.output_edit = QLineEdit(), QLineEdit(str(self.settings.value("metrics/output_parent", "")))
        self.source_edit.setPlaceholderText("LAS / LAZ / BIN，或单木分割结果文件夹")
        row = self._path_row(self.source_edit, "点云文件", self._choose_source)
        folder = QPushButton("分割结果目录")
        folder.clicked.connect(self._choose_result)
        row.addWidget(folder)
        form.addRow("输入点云", row)
        form.addRow("输出位置", self._path_row(self.output_edit, "选择文件夹", self._choose_output))
        self.mode = TerrainInput()
        for text, value in (("自动：优先复用编号，否则检测树干", "auto"),
                             ("仅复用已有单木编号", "reuse"), ("从原始点云检测树干", "stems")):
            self.mode.addItem(text, value)
        form.addRow("处理方式", self.mode)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("留空自动识别 instance_id / TreeID；0 为未分配")
        form.addRow("编号字段", self.label_edit)
        self.ground_mode = TerrainInput()
        self.ground_mode.addItem("自动：使用地面分类，否则调用 CSF", "auto")
        self.ground_mode.addItem("Z 已是离地高度（已完成地形归一化）", "normalized")
        form.addRow("地面处理", self.ground_mode)
        note = QLabel("仅做格式转换的 normalized.laz 仍可能是绝对高程，应选择自动地面处理。")
        note.setWordWrap(True)
        form.addRow(note)
        saved = str(self.settings.value("forestry/cloudcompare_path", ""))
        detected = cloudcompare.find_binary(saved)
        self.binary_edit = QLineEdit(saved or (str(detected) if detected else ""))
        form.addRow("CloudCompare", self._path_row(self.binary_edit, "选择程序", self._choose_binary))
        self.binary_edit.editingFinished.connect(self._save_binary)
        engine_note = QLabel("与单木分割共享程序路径。已有编号且含地面分类或已归一化的 LAS/LAZ 可独立处理；BIN 转换和无地面分类输入需要 CloudCompare。")
        engine_note.setWordWrap(True)
        form.addRow(engine_note)
        self.metres = QCheckBox("我确认 XYZ 单位为米")
        self.export_cloud = QCheckBox("导出参数标注 LAZ，保留原坐标、属性和编号")
        self.export_cloud.setChecked(True)
        form.addRow(self.metres)
        form.addRow(self.export_cloud)
        body.addWidget(group)
        advanced = QGroupBox("参数设置")
        form = QFormLayout(advanced)
        self.controls = {}
        specs = [
            ("breast_height", "胸径测量高度", .1, 5., " 米"),
            ("dbh_band", "胸径切片总厚度", .04, 1., " 米"),
            ("min_radius", "最小树干半径", .005, 1., " 米"),
            ("max_radius", "最大树干半径", .05, 2., " 米"),
            ("fit_tolerance", "圆拟合径向容差", .002, .1, " 米"),
            ("min_arc", "最少圆弧覆盖", 90., 300., " °"),
            ("ground_support", "最远地面支持距离", .5, 20., " 米"),
            ("max_assign_distance", "自动分组最远树干距离", 1., 20., " 米"),
            ("min_height", "最低参考树高", .5, 50., " 米"),
            ("form_factor", "材积形数（按树种调整）", .1, 1., ""),
        ]
        defaults = MetricOptions()
        for name, title, low, high, suffix in specs:
            edit = NumberInput()
            edit.setDecimals(3)
            edit.setRange(low, high)
            edit.setValue(getattr(defaults, name))
            edit.setSuffix(suffix)
            edit.setSingleStep({"breast_height": .05, "fit_tolerance": .002, "min_arc": 5.,
                                "ground_support": .25, "max_assign_distance": .25,
                                "min_height": .25, "form_factor": .05}.get(name, .01))
            self.controls[name] = edit
            form.addRow(title, edit)
        self.terrain = TerrainInput()
        for title, value in (("平坦地形", "FLAT"), ("起伏地形", "RELIEF"), ("陡坡", "SLOPE")):
            self.terrain.addItem(title, value)
        form.addRow("CSF 地形", self.terrain)
        body.addWidget(advanced)
        note = QLabel("树冠基部是几何估计，不能直接判定活枝；材积依赖所设形数。胸径质量不足时留空并说明原因，自动分组与参数均需复核。")
        note.setWordWrap(True)
        body.addWidget(note)
        actions = QHBoxLayout()
        self.run_button, self.cancel_button = QPushButton("开始提取"), QPushButton("取消")
        self.run_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.preview = self.workspace.table
        self.folder_button = QPushButton("打开结果文件夹")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_directory)))
        self.folder_button.setObjectName("secondaryButton")
        actions.addWidget(self.folder_button)
        self.training_button = QPushButton("03 · 使用同一分割结果导出训练数据")
        self.training_button.setEnabled(False)
        self.training_button.clicked.connect(self._continue_training)
        self.training_button.setObjectName("secondaryButton")
        actions.addWidget(self.training_button)
        layout.addLayout(actions)
        self.source_edit.textChanged.connect(self._source_changed)

    def _scene_busy(self, busy):
        self.inputs.setEnabled(not busy)
        self.open_result_button.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.training_button.setEnabled(not busy and self._segmentation_result is not None)
        self.progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress.setValue(1)

    def _scene_loaded(self, scene):
        self.output_directory = scene["root"]
        self.result = {"output_directory": scene["root"], "summary": scene["summary"]}
        self.folder_button.setEnabled(True)

    def _open_result(self):
        if self.worker is not None or self.scene_worker is not None:
            return
        path = QFileDialog.getExistingDirectory(self, "打开已完成的参数结果目录", self.output_directory or "")
        if path:
            self.open_result_directory(path)

    def open_result_directory(self, path):
        if self.worker is not None or self.scene_worker is not None:
            return False
        self._segmentation_result = None
        self.result = self.output_directory = None
        self.folder_button.setEnabled(False)
        self.training_button.setEnabled(False)
        self.tabs.setCurrentIndex(0)
        self.workspace.load(path)
        return True

    def _source_changed(self):
        self._segmentation_result = None
        self.training_button.setEnabled(False)

    def use_segmentation_result(self, result):
        """Explicit navigation reuses IDs and cannot replace an active job."""
        if self.worker is not None or self.scene_worker is not None:
            return False
        self.source_edit.setText(result["output_directory"])
        self.output_edit.setText(result["output_directory"])
        self.mode.setCurrentIndex(self.mode.findData("reuse"))
        self.label_edit.setText("instance_id")
        self.ground_mode.setCurrentIndex(self.ground_mode.findData("auto"))
        self.binary_edit.setText(result.get("binary", ""))
        self.metres.setChecked(True)  # Successful segmentation requires metre coordinates.
        self.result = self.output_directory = None
        self.workspace.clear()
        self.tabs.setCurrentIndex(1)
        self.folder_button.setEnabled(False)
        self.progress.setValue(0)
        self._segmentation_result = result
        self.training_button.setEnabled(True)
        self.status_label.setText("已带入单木分割结果，使用原实例编号和地面分类。检查参数后点击“开始提取”。")
        return True

    def _continue_training(self):
        if self._segmentation_result is not None and self.worker is None and self.scene_worker is None:
            self.continue_requested.emit("forest-training", self._segmentation_result)

    @staticmethod
    def _path_row(edit, title, callback):
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        button = QPushButton(title)
        button.clicked.connect(callback)
        row.addWidget(button)
        return row

    def _save_binary(self):
        self.settings.setValue("forestry/cloudcompare_path", self.binary_edit.text().strip())

    def _choose_binary(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CloudCompare", self.binary_edit.text(), "程序 (*.exe);;所有文件 (*)")
        if path:
            self.binary_edit.setText(path)
            self._save_binary()

    def _choose_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择林地点云", self.source_edit.text(), "点云 (*.las *.laz *.bin);;所有文件 (*)")
        if path:
            self.source_edit.setText(path)
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(Path(path).parent))

    def _choose_result(self):
        path = QFileDialog.getExistingDirectory(self, "选择包含 forest_instances.laz 的分割结果目录")
        if path:
            self.source_edit.setText(path)
            if not self.output_edit.text().strip():
                self.output_edit.setText(path)

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出位置", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _start(self):
        if self.worker is not None or self.scene_worker is not None:
            return
        try:
            source, output = self.source_edit.text().strip(), self.output_edit.text().strip()
            if not source or not Path(source).exists():
                raise ValueError("请选择存在的点云或分割结果目录。")
            if not output or not Path(output).is_dir():
                raise ValueError("请选择输出文件夹。")
            if not self.metres.isChecked():
                raise ValueError("请确认 XYZ 坐标单位为米。")
            options = MetricOptions(**{name:edit.value() for name, edit in self.controls.items()}).validate()
        except ValueError as exc:
            QMessageBox.warning(self, "参数检查", str(exc))
            return
        self._save_binary()
        self.settings.setValue("metrics/output_parent", output)
        self.result = self.output_directory = None
        self.folder_button.setEnabled(False)
        self.workspace.clear()
        self.open_result_button.setEnabled(False)
        self.new_button.setEnabled(False)
        self.worker = MetricsWorker(dict(source=source, output_parent=output, binary=self.binary_edit.text().strip(),
            coordinates_in_metres=True, mode=self.mode.currentData(), label_field=self.label_edit.text().strip(),
            ground_mode=self.ground_mode.currentData(), terrain=self.terrain.currentData(),
            export_cloud=self.export_cloud.isChecked(), options=options))
        self.worker.status_signal.connect(self._status)
        bind_task(self, "worker", self._finished)
        self.inputs.setEnabled(False)
        self.training_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("正在读取输入…")
        self.worker.start()

    def _status(self, data):
        if "output_directory" in data:
            self.output_directory = data["output_directory"]
        text = STAGES.get(data.get("stage"), "正在处理点云")
        done = data.get("trees", data.get("points"))
        if done is not None:
            text += f"：{done:,} / {data.get('total', done):,}"
        self.status_label.setText(text)

    def _cancel(self):
        active = self.worker or self.scene_worker
        if active:
            active.stop()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("正在停止并关闭输出文件…")

    def _finished(self, result=None, error=""):
        self.inputs.setEnabled(True)
        self.training_button.setEnabled(self._segmentation_result is not None)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.open_result_button.setEnabled(True)
        self.new_button.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.folder_button.setEnabled(bool(self.output_directory))
        if error:
            self.status_label.setText(("已取消：" if result and result.get("cancelled") else "提取失败：")+error)
            return
        self.result = result
        self.output_directory = result["output_directory"]
        self.folder_button.setEnabled(True)
        summary = result["summary"]
        self.status_label.setText(f"完成：{summary['tree_count']} 个单木编号；{summary['accepted_dbh_count']} 个胸径通过检查；"
                                  f"{summary['flagged_count']} 个结果有质量提示。用时 {summary['seconds']:.1f} 秒。")
        self.progress.setValue(1)
        self.tabs.setCurrentIndex(0)
        self.workspace.load(self.output_directory)
