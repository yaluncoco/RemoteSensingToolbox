"""Forestry measurements as a separate point-cloud tool."""
import csv
from pathlib import Path

from PyQt5.QtCore import QSettings, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from PointCloud import cloudcompare
from PointCloud.forest_metrics import MetricOptions
from PointCloud.forestry_gui import NumberInput, IntegerInput, TerrainInput
from Toolbox.task_support import bind_task
from Toolbox.version import APP_ID


STAGES = {"reading":"读取点云与编号", "ground":"建立地形并计算离地高度",
          "stems":"检测树干并校验胸径", "grouping":"复用已有单木编号",
          "measuring":"逐树提取参数", "writing":"生成并校验 CSV、Excel 和分布图",
          "export":"导出并校验原始属性", "complete":"提取完成",
          "sampling":"准备地面过滤采样", "reading_bin":"读取 BIN 工作副本",
          "validating_bin":"校验 BIN 格式转换"}

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
    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle("林业参数提取")
        self.resize(880, 850)
        self.setMinimumSize(650, 500)
        self.settings = settings if settings is not None else QSettings(APP_ID, APP_ID)
        self.worker = self.result = self.output_directory = None
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.inputs = QWidget()
        body = QVBoxLayout(self.inputs)
        scroll.setWidget(self.inputs)
        layout.addWidget(scroll)
        heading = QLabel("林业参数提取")
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
        layout.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.preview = QTableWidget(0, 0)
        self.preview.setMaximumHeight(170)
        self.preview.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.preview)
        self.folder_button = QPushButton("打开结果文件夹")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_directory)))
        layout.addWidget(self.folder_button)

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
        if self.worker is not None:
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
        self.preview.setRowCount(0)
        self.worker = MetricsWorker(dict(source=source, output_parent=output, binary=self.binary_edit.text().strip(),
            coordinates_in_metres=True, mode=self.mode.currentData(), label_field=self.label_edit.text().strip(),
            ground_mode=self.ground_mode.currentData(), terrain=self.terrain.currentData(),
            export_cloud=self.export_cloud.isChecked(), options=options))
        self.worker.status_signal.connect(self._status)
        bind_task(self, "worker", self._finished)
        self.inputs.setEnabled(False)
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
        if self.worker:
            self.worker.stop()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("正在停止并关闭输出文件…")

    def _finished(self, result=None, error=""):
        self.inputs.setEnabled(True)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
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
        with (Path(self.output_directory)/"林业参数.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))[:200]
        columns = [("TreeID", "编号"), ("Height_m", "树高(m)"), ("DBH_cm", "胸径(cm)"),
                   ("CrownDiameter_m", "冠幅(m)"), ("DBH_status", "胸径状态"), ("QualityFlags", "质量提示")]
        self.preview.setColumnCount(len(columns))
        self.preview.setHorizontalHeaderLabels([title for _, title in columns])
        self.preview.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (name, _) in enumerate(columns):
                value = row[name]
                if name == "DBH_status":
                    value = QUALITY_TEXT.get(value, value)
                elif name == "QualityFlags":
                    value = "；".join(QUALITY_TEXT.get(flag.removeprefix("dbh_"), flag) for flag in value.split("|") if flag)
                if name not in ("TreeID", "DBH_status", "QualityFlags") and value:
                    value = f"{float(value):.2f}"
                self.preview.setItem(i, j, QTableWidgetItem(value or "—"))
        self.preview.resizeColumnsToContents()
