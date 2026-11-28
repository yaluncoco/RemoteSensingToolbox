"""Interactive tree inventory, point-cloud canvas and full parameter inspector."""
from pathlib import Path
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from PointCloud.forest_viewer import ForestCanvas
from Toolbox.task_support import bind_task
from Toolbox.soft_widgets import SoftComboBox, SoftScrollArea


FIELDS = [
    ("位置与高度", [("X", "树干 X", "m"), ("Y", "树干 Y", "m"),
      ("Height_m", "树高 · P99", "m"), ("HeightMax_m", "最高点高度", "m"),
      ("NPoints", "完整实例点数", "点")]),
    ("树干与胸径", [("DBH_cm", "胸径", "cm"), ("DBH_status", "胸径状态", ""),
      ("DBH_points", "胸径切片点数", "点"), ("DBH_fit_inlier", "拟合内点率", "%"),
      ("DBH_fit_rmse_m", "拟合误差 RMSE", "m"), ("DBH_arc_deg", "圆弧覆盖", "°"),
      ("StemVolume_m3", "估算材积", "m³")]),
    ("树冠结构", [("ClearBole_m", "几何枝下高", "m"), ("CrownBase_m", "几何冠基部", "m"),
      ("CrownLength_m", "冠长", "m"), ("CrownRatio", "冠长率", "%"),
      ("CrownEW_m", "X 方向冠幅", "m"), ("CrownNS_m", "Y 方向冠幅", "m"),
      ("CrownDiameter_m", "平均冠幅", "m"), ("CrownArea_m2", "冠投影面积", "m²"),
      ("CrownVolume_m3", "树冠体积", "m³"), ("CrownSliceCoverage", "有效切片比例", "%")]),
    ("地面支持", [("GroundSupportedFraction", "地面支持比例", "%"),
      ("GroundDistanceP95_m", "地面支持距离 P95", "m")]),
]


class SceneTask(QThread):
    status_signal = pyqtSignal(object)
    finished_signal = pyqtSignal(object, str)

    def __init__(self, function, *args):
        super().__init__()
        self.function, self.args = function, args

    def stop(self):
        self.requestInterruption()

    def run(self):
        try:
            result = self.function(*self.args, cancelled=self.isInterruptionRequested, notify=self.status_signal.emit)
            self.finished_signal.emit(result, "")
        except Exception as exc:
            self.finished_signal.emit(None, str(exc))


class NumericItem(QTableWidgetItem):
    def __lt__(self, other):
        return self.data(Qt.UserRole) < other.data(Qt.UserRole)


class ForestWorkspace(QWidget):
    busy_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    loaded = pyqtSignal(object)

    def __init__(self, host, quality_text):
        super().__init__(host)
        self.host, self.quality_text = host, quality_text
        self.scene = None
        self.rows = {}
        self.selected_id = 0
        self.last_export = None
        self._detail_cache = {}
        self._detail_requested_id = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        self.summary = QLabel("点云与参数联动 · 选择一棵树，查看完整属性")
        self.summary.setObjectName("summaryLabel")
        layout.addWidget(self.summary)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 8, 0)
        tools = QHBoxLayout()
        self.color = SoftComboBox()
        for label, data in (("实例颜色", "instance"), ("离地高度", "height"), ("原始 RGB", "rgb")):
            self.color.addItem(label, data)
        self.color.setToolTip("切换颜色只影响显示，不修改点云或标签")
        tools.addWidget(self.color)
        for label, direction in (("三维", "iso"), ("俯视", "top"), ("正视", "front")):
            button = QPushButton(label)
            button.setObjectName("secondaryButton")
            button.clicked.connect(lambda _, d=direction: self.canvas.reset_view(d))
            tools.addWidget(button)
        fit = QPushButton("适应视图")
        fit.setObjectName("secondaryButton")
        fit.clicked.connect(lambda: self.canvas.reset_view())
        tools.addWidget(fit)
        tools.addStretch()
        snapshot = QPushButton("保存视图")
        snapshot.setObjectName("secondaryButton")
        snapshot.clicked.connect(self._snapshot)
        tools.addWidget(snapshot)
        center_layout.addLayout(tools)
        options = QHBoxLayout()
        self.ground = QCheckBox("背景点")
        self.ground.setChecked(True)
        self.labels = QCheckBox("树木编号")
        self.labels.setChecked(True)
        self.normalized = QCheckBox("离地高度视图")
        for control in (self.ground, self.labels, self.normalized):
            options.addWidget(control)
        options.addStretch()
        options.addWidget(QLabel("点大小"))
        self.point_size = QSlider(Qt.Horizontal)
        self.point_size.setRange(1, 3)
        self.point_size.setValue(2)
        self.point_size.setMaximumWidth(80)
        options.addWidget(self.point_size)
        center_layout.addLayout(options)
        vertical = QSplitter(Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        self.canvas = ForestCanvas()
        vertical.addWidget(self.canvas)
        listing = QWidget()
        list_layout = QVBoxLayout(listing)
        list_layout.setContentsMargins(0, 8, 0, 0)
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("查找单木编号…")
        self.review_only = QCheckBox("仅看质量提示")
        search_row.addWidget(self.search, 1)
        search_row.addWidget(self.review_only)
        list_layout.addLayout(search_row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["编号", "树高 m", "胸径 cm", "冠幅 m", "完整点数", "质量提示"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setMinimumHeight(120)
        list_layout.addWidget(self.table)
        vertical.addWidget(listing)
        vertical.setSizes([540, 200])
        center_layout.addWidget(vertical, 1)
        split.addWidget(center)
        details = SoftScrollArea()
        details.setObjectName("inspector")
        details.setWidgetResizable(True)
        details.setMinimumWidth(275)
        details.setFrameShape(QFrame.NoFrame)
        self.detail_content = QWidget()
        self.detail_content.setObjectName("inspectorBody")
        right = QVBoxLayout(self.detail_content)
        right.setContentsMargins(18, 16, 18, 18)
        right.setSpacing(14)
        self.tree_title = QLabel("选择一棵树")
        self.tree_title.setObjectName("pageTitle")
        right.addWidget(self.tree_title)
        self.tree_subtitle = QLabel("点击点云或下方实例列表")
        self.tree_subtitle.setWordWrap(True)
        self.tree_subtitle.setObjectName("summaryLabel")
        right.addWidget(self.tree_subtitle)
        highlights = QHBoxLayout()
        self.highlights = {}
        for name, title in (("Height_m", "树高 / m"), ("DBH_cm", "胸径 / cm"), ("CrownDiameter_m", "冠幅 / m")):
            tile = QFrame()
            tile.setObjectName("metricTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(8, 10, 8, 10)
            value = QLabel("—")
            value.setObjectName("metricValue")
            label = QLabel(title)
            label.setObjectName("summaryLabel")
            tile_layout.addWidget(value)
            tile_layout.addWidget(label)
            highlights.addWidget(tile)
            self.highlights[name] = value
        right.addLayout(highlights)
        focus_row = QHBoxLayout()
        self.isolate_button = QPushButton("独立查看")
        self.overview_button = QPushButton("回到全景")
        self.overview_button.setObjectName("secondaryButton")
        self.isolate_button.clicked.connect(self.show_instance)
        self.overview_button.clicked.connect(self.show_overview)
        focus_row.addWidget(self.isolate_button)
        focus_row.addWidget(self.overview_button)
        right.addLayout(focus_row)
        self.export_button = QPushButton("导出此树 · 完整 LAZ")
        self.export_button.clicked.connect(lambda: self._choose_export(False))
        self.export_all_button = QPushButton("导出全部单木与参数表")
        self.export_all_button.setObjectName("secondaryButton")
        self.export_all_button.clicked.connect(lambda: self._choose_export(True))
        right.addWidget(self.export_button)
        right.addWidget(self.export_all_button)
        self.quality = QLabel("尚未选择实例")
        self.quality.setWordWrap(True)
        self.quality.setObjectName("qualityNote")
        right.addWidget(self.quality)
        self.detail_labels = {}
        for title, fields in FIELDS:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setSpacing(9)
            for name, label, unit in fields:
                value = QLabel("—")
                value.setTextInteractionFlags(Qt.TextSelectableByMouse)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setToolTip(name)
                form.addRow(label, value)
                self.detail_labels[name] = value
            right.addWidget(group)
        method = QLabel("树高使用 P99；冠幅使用 2%–98% 分位宽度。冠基部为几何估计，材积按形数估算。预览经过抽样，实例导出保留完整点与原属性。")
        method.setWordWrap(True)
        method.setObjectName("summaryLabel")
        right.addWidget(method)
        right.addStretch()
        details.setWidget(self.detail_content)
        split.addWidget(details)
        split.setSizes([1000, 340])
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        self.color.currentIndexChanged.connect(self._display_options)
        self.ground.toggled.connect(self._display_options)
        self.labels.toggled.connect(self._display_options)
        self.point_size.valueChanged.connect(self._display_options)
        self.normalized.toggled.connect(self._height_view)
        self.search.textChanged.connect(self._filter)
        self.review_only.toggled.connect(self._filter)
        self.canvas.tree_selected.connect(self.select_tree)
        self.canvas.isolation_requested.connect(self.show_instance)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.clear()

    def clear(self):
        self.scene = None
        self.rows = {}
        self.selected_id = 0
        self._detail_cache = {}
        self._detail_requested_id = None
        self.table.setRowCount(0)
        self.search.clear()
        self.review_only.setChecked(False)
        self.canvas.set_scene(None)
        self.tree_title.setText("选择一棵树")
        self.tree_subtitle.setText("点击点云或下方实例列表")
        self.quality.setText("尚未选择实例")
        self.summary.setText("点云与参数联动 · 选择一棵树，查看完整属性")
        for label in self.detail_labels.values():
            label.setText("—")
        for label in self.highlights.values():
            label.setText("—")
        self._set_actions(False)

    def _set_actions(self, enabled):
        self.export_button.setEnabled(enabled and bool(self.selected_id))
        self.isolate_button.setEnabled(enabled and bool(self.selected_id))
        self.export_all_button.setEnabled(enabled and bool(self.scene))

    def _task(self, function, args, callback):
        if self.host.scene_worker is not None:
            return False
        self.host.scene_worker = SceneTask(function, *args)
        self.host.scene_worker.status_signal.connect(lambda data: self.status_changed.emit(
            f"{'加载点云' if data.get('stage') == 'scene' else '导出完整实例'}：{data.get('points', 0):,} / {data.get('total', 0):,}"))
        bind_task(self.host, "scene_worker", callback)
        self._set_actions(False)
        self.busy_changed.emit(True)
        self.host.scene_worker.start()
        return True

    def load(self, directory):
        from PointCloud.forest_scene import load_scene
        if self.host.scene_worker is not None:
            return
        self.clear()
        self.status_changed.emit("正在加载实例点云与完整参数…")
        self._task(load_scene, (directory,), self._loaded)

    def _loaded(self, scene, error):
        self.busy_changed.emit(False)
        if error:
            self.status_changed.emit("点云加载未完成：" + error)
            return
        self.scene = scene
        self.rows = {int(row["TreeID"]): row for row in scene["rows"]}
        self.canvas.set_scene(scene)
        self.color.model().item(2).setEnabled(bool(scene["has_rgb"]))
        if not bool(scene["has_rgb"]) and self.color.currentData() == "rgb":
            self.color.setCurrentIndex(0)
        self._display_options()
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(scene["rows"]))
        for i, row in enumerate(scene["rows"]):
            for j, name in enumerate(("TreeID", "Height_m", "DBH_cm", "CrownDiameter_m", "NPoints", "ReviewRequired")):
                raw = row.get(name, "")
                if j == 5:
                    text = "需复核" if raw in (True, "True", "1") else "未检出异常"
                    item = QTableWidgetItem(text)
                else:
                    value = float(raw) if raw not in ("", None) else -1.
                    text = (str(int(raw)) if j in (0, 4) else f"{value:.2f}") if value >= 0 else "—"
                    item = NumericItem(text)
                    item.setData(Qt.UserRole, int(raw) if j in (0, 4) else value)
                self.table.setItem(i, j, item)
        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.AscendingOrder)
        self.summary.setText(f"{len(self.rows):,} 个实例  ·  完整点云 {scene['summary']['source_points']:,} 点  ·  显示抽样 {len(scene['ids']):,} 点")
        self._filter()
        self._set_actions(True)
        self.status_changed.emit("实例工作台已就绪。点击树木可联动查看参数，双击独立查看。")
        self.loaded.emit(scene)

    def select_tree(self, tree_id):
        tree_id = int(tree_id)
        if tree_id not in self.rows:
            return
        if self.canvas.scene is not self.scene and tree_id != self.selected_id:
            self.show_overview()
        self.selected_id = tree_id
        row = self.rows[tree_id]
        self.canvas.select_tree(tree_id)
        self.tree_title.setText(f"单木 {tree_id}")
        self.tree_subtitle.setText(f"完整点数 {int(row['NPoints']):,} · 编号与结果表一致")
        for name, label in self.highlights.items():
            raw = row.get(name)
            label.setText(f"{float(raw):.2f}" if raw not in (None, "") else "—")
        flags = row.get("QualityFlags", "")
        translated = [self.quality_text.get(v.removeprefix("dbh_"), v) for v in flags.split("|") if v]
        self.quality.setText("；".join(translated) if translated else "未检出预设质量异常 · 自动估计仍需复核")
        for _, fields in FIELDS:
            for name, _, unit in fields:
                value = row.get(name, "")
                if value in ("", None):
                    text = "—"
                elif name == "DBH_status":
                    text = self.quality_text.get(value, value)
                elif unit == "点":
                    text = f"{int(value):,}"
                else:
                    number = float(value) * (100 if unit == "%" else 1)
                    text = f"{number:,.{3 if name in ('X', 'Y', 'DBH_fit_rmse_m', 'StemVolume_m3') else 2}f} {unit}"
                self.detail_labels[name].setText(text)
        self.table.blockSignals(True)
        for i in range(self.table.rowCount()):
            if int(self.table.item(i, 0).text()) == tree_id:
                if not self.table.isRowHidden(i):
                    self.table.selectRow(i)
                    self.table.scrollToItem(self.table.item(i, 0))
                break
        self.table.blockSignals(False)
        self._set_actions(self.host.scene_worker is None)

    def show_instance(self):
        if not self.scene or not self.selected_id or self.host.scene_worker is not None:
            return
        self.canvas.isolate(True)
        self._detail_requested_id = self.selected_id
        if self.selected_id in self._detail_cache:
            self._instance_loaded(self._detail_cache[self.selected_id], "")
            return
        from PointCloud.forest_scene import load_instance
        self.status_changed.emit("正在从完整点云加载所选单木细节…")
        self._task(load_instance, (self.scene, self.selected_id), self._instance_loaded)

    def _instance_loaded(self, detail, error):
        self.busy_changed.emit(False)
        self._set_actions(True)
        if error:
            self.status_changed.emit("单木细节未加载：" + error)
            return
        tree_id = detail["detail_tree_id"]
        if len(self._detail_cache) >= 3:
            self._detail_cache.pop(next(iter(self._detail_cache)))
        self._detail_cache[tree_id] = detail
        if self.selected_id != tree_id or self._detail_requested_id != tree_id:
            return
        self.canvas.set_scene(detail)
        self.canvas.select_tree(tree_id)
        self.canvas.isolate(True)
        heights_available = bool(np.isfinite(detail['height']).any())
        self.normalized.setEnabled(heights_available)
        if not heights_available:
            self.normalized.setChecked(False)
        self.status_changed.emit(f"单木 {tree_id}：完整 {detail['detail_count']:,} 点，当前显示 {len(detail['ids']):,} 点。")

    def show_overview(self):
        if self.scene is None:
            return
        self._detail_requested_id = None
        self.normalized.setEnabled(True)
        self.canvas.set_scene(self.scene)
        self.canvas.select_tree(self.selected_id)
        self.canvas.isolate(False)

    def _table_selected(self):
        selected = self.table.selectionModel().selectedRows()
        if selected:
            self.select_tree(int(self.table.item(selected[0].row(), 0).text()))

    def _filter(self):
        needle = self.search.text().strip()
        for i in range(self.table.rowCount()):
            tree_id = self.table.item(i, 0).text()
            flagged = self.rows[int(tree_id)].get("ReviewRequired") in (True, "True", "1")
            self.table.setRowHidden(i, needle not in tree_id or (self.review_only.isChecked() and not flagged))

    def _display_options(self, *_):
        self.canvas.color_mode = self.color.currentData()
        self.canvas.show_ground = self.ground.isChecked()
        self.canvas.show_labels = self.labels.isChecked()
        self.canvas.point_size = self.point_size.value()
        self.canvas.invalidate()

    def _height_view(self):
        self.canvas.normalized = self.normalized.isChecked()
        self.canvas.reset_view()

    def _snapshot(self):
        if not self.scene:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存当前点云视图", str(Path(self.scene["root"]) / "实例视图.png"), "PNG (*.png)")
        if path:
            if not self.canvas.grab().save(path):
                QMessageBox.warning(self, "保存失败", "无法写入所选图片文件。")
            else:
                self.status_changed.emit("已保存当前点云视图：" + path)

    def _choose_export(self, all_trees):
        if not self.scene or self.host.scene_worker is not None:
            return
        parent = QFileDialog.getExistingDirectory(self, "选择实例导出位置（自动创建新目录）", self.scene["root"])
        if parent:
            self.export_to(parent, list(self.rows) if all_trees else [self.selected_id])

    def export_to(self, parent, tree_ids):
        from PointCloud.forest_scene import export_instances
        if self.scene:
            self.status_changed.emit("正在读取完整实例并保留原坐标与属性…")
            self._task(export_instances, (self.scene, tree_ids, parent), self._exported)

    def _exported(self, result, error):
        self.busy_changed.emit(False)
        self._set_actions(True)
        if error:
            self.status_changed.emit("实例导出未完成：" + error)
            return
        self.last_export = result
        self.status_changed.emit(f"已导出 {len(result['files'])} 个完整单木，点数与原属性读回校验通过：{result['output_directory']}")
