from PointCloud.file_io import read_point_cloud, write_point_cloud
from Toolbox.task_support import bind_task
import sys
import os
import threading
import numpy as np
import open3d as o3d
import traceback
import math  # For math.isclose

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QWidget, QFileDialog, QLabel,
                             QGroupBox, QStackedWidget, QFrame, QCheckBox,
                             QProgressBar, QMessageBox, QTabWidget, QSizePolicy,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QTextEdit, QLineEdit, QComboBox)

from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import QScrollArea
from Toolbox.version import APP_VERSION
from PyQt5.QtGui import QFont, QIcon
from PointCloud.visualization_options import VisualizationFormatWidget

# Import worker classes and visualization functions
from PointCloud.workers import (PepperProcessWorker, GenericConvertWorker,
                     ConvertScanNetWorker, CheckShapeWorker, MergePredictionWorker, DownsampleWorker)
from PointCloud.visualization import (_run_visualization, _run_sequential_visualization,
                           prepare_point_cloud_for_visualization, _run_custom_color_visualization)


class _VisualizationSignals(QObject):
    # No window parent: a renderer may outlive a closed tool window. Connections to
    # destroyed widgets disconnect automatically, while the thread keeps this alive.
    status = pyqtSignal(str)
    finished = pyqtSignal()


class PointCloudProcessor(QMainWindow):

    def __init__(self):
        super().__init__()
        self.point_cloud = None
        self.visualization_thread = None
        self.custom_visualization_signals = _VisualizationSignals()

        # Workers
        self.pepper_worker = None
        self.generic_convert_worker = None
        self.scannet_convert_worker = None
        self.check_shape_worker = None
        self.merge_prediction_worker = None
        self.downsample_worker = None  # ADD THIS
        # No S3DIS worker anymore

        self.processed_point_cloud_for_comparison = None
        self.original_file_path_for_comparison = None
        self.current_compare_method_name = ""

        # Paths for Pepper - Pipeline, SOR, ROR (dynamic attributes used later)

        # Paths for Generic conversion
        self.generic_single_input_path = None
        self.generic_single_output_dir = None
        self.generic_batch_input_dir = None
        self.generic_batch_output_dir = None

        # Paths for ScanNet conversion
        self.scannet_input_dir = None
        self.scannet_output_dir = None

        # Path for NPY Shape Check
        self.npy_shape_input_path = None

        # Paths for Prediction Analysis
        self.merge_orig_txt_path = None
        self.merge_pred_txt_path = None
        self.merge_output_txt_path = None  # Stores the result of merging
        self.vis_custom_input_txt_path = None  # For custom visualization, can be filled by merge result
        self.vis_custom_screenshot_path = None

        # Paths for Downsampling (ADD THIS BLOCK)
        self.downsample_single_input_path = None
        self.downsample_single_output_path = None
        self.downsample_batch_input_dir = None
        self.downsample_batch_output_dir = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('点云处理工作台')
        self.setMinimumSize(840, 580)
        self.setGeometry(100, 100, 1120, 760)
        self.setStyleSheet("""
            QMainWindow { background-color: #F4F6F8; }
            QWidget { font-family: "Microsoft YaHei", "Segoe UI", Arial; font-size: 10pt; color: #1F2933; }
            QLabel { color: #1F2933; }
        """)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(22, 20, 22, 18)
        main_layout.setSpacing(14)

        title_label = QLabel("点云处理工作台")
        title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        title_label.setStyleSheet("color: #17202A; margin-bottom: 2px;")
        main_layout.addWidget(title_label)

        load_group = QGroupBox("当前点云")
        load_group.setStyleSheet(self.get_groupbox_style())
        load_layout = QVBoxLayout(load_group);
        load_layout.setSpacing(12)
        load_info_layout = QHBoxLayout()
        load_button = QPushButton(QIcon.fromTheme("document-open", QIcon(
            ":/qt-project.org/styles/commonstyle/images/standardbutton-open-32.png")), " 打开并预览")
        load_button.clicked.connect(self.load_and_visualize_point_cloud)
        load_button.setMinimumHeight(42);
        load_button.setMinimumWidth(145)
        load_button.setStyleSheet(self.get_button_style("info"))
        load_info_layout.addWidget(load_button, 0)
        info_widget = QWidget();
        info_vlayout = QVBoxLayout(info_widget)
        info_vlayout.setContentsMargins(10, 0, 0, 0)
        self.file_label = QLabel("当前文件：未打开");
        self.file_label.setStyleSheet("padding: 2px; color: #334155;");
        self.file_label.setWordWrap(True)
        info_vlayout.addWidget(self.file_label)
        self.points_info = QLabel("点云信息：等待加载");
        self.points_info.setStyleSheet("padding: 2px; color: #64748B;")
        info_vlayout.addWidget(self.points_info)
        load_info_layout.addWidget(info_widget, 1)
        load_layout.addLayout(load_info_layout)
        main_layout.addWidget(load_group)

        modules_group = QGroupBox("常用工作")
        modules_group.setStyleSheet(self.get_groupbox_style())
        modules_layout = QGridLayout(modules_group)  # Use QGridLayout for more flexible button arrangement
        modules_layout.setSpacing(12)

        pepper_button = QPushButton("点云处理 / 去噪")
        pepper_button.clicked.connect(lambda: self.switch_page(1))
        pepper_button.setMinimumHeight(76);
        pepper_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pepper_button.setStyleSheet(self.get_button_style("primary", font_size="12pt"))
        modules_layout.addWidget(pepper_button, 0, 0)

        convert_button = QPushButton("格式转换 / NPY检查")
        convert_button.clicked.connect(lambda: self.switch_page(2))
        convert_button.setMinimumHeight(76);
        convert_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        convert_button.setStyleSheet(self.get_button_style("info", font_size="12pt"))
        modules_layout.addWidget(convert_button, 0, 1)

        analysis_button = QPushButton("预测标签分析")
        analysis_button.clicked.connect(lambda: self.switch_page(3))  # New Page Index
        analysis_button.setMinimumHeight(76);
        analysis_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        analysis_button.setStyleSheet(self.get_button_style("warning", font_size="12pt"))
        modules_layout.addWidget(analysis_button, 1, 0)  # NO LONGER SPANS

        downsample_button = QPushButton("TXT 点云下采样")
        downsample_button.clicked.connect(lambda: self.switch_page(4))
        downsample_button.setMinimumHeight(76)
        downsample_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        downsample_button.setStyleSheet(self.get_button_style("secondary", font_size="12pt"))
        modules_layout.addWidget(downsample_button, 1, 1)

        main_layout.addWidget(modules_group)

        status_group = QGroupBox("运行状态")
        status_group.setStyleSheet(self.get_groupbox_style())
        status_layout = QVBoxLayout(status_group)
        self.dashboard_status_label = QLabel("就绪")
        self.dashboard_status_label.setStyleSheet(self.get_status_label_style())
        self.dashboard_status_label.setWordWrap(True)
        status_layout.addWidget(self.dashboard_status_label)
        main_layout.addWidget(status_group)
        main_layout.addStretch(1)

        footer_label = QLabel(f"遥感与点云综合工具箱 {APP_VERSION}")
        footer_label.setAlignment(Qt.AlignRight)
        footer_label.setStyleSheet("color: #94A3B8; margin-top: 8px; font-size: 9pt;")
        main_layout.addWidget(footer_label)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.scroll_page(main_widget))  # Index 0: Main Page

        pepper_page = self.create_pepper_page()  # Index 1
        self.stacked_widget.addWidget(self.scroll_page(pepper_page))
        convert_page = self.create_convert_page()  # Index 2
        self.stacked_widget.addWidget(self.scroll_page(convert_page))
        analysis_page = self.create_prediction_analysis_page()  # Index 3
        self.stacked_widget.addWidget(self.scroll_page(analysis_page))
        downsample_page = self.create_downsample_page()  # Index 4
        self.stacked_widget.addWidget(self.scroll_page(downsample_page))

        shell_widget = QWidget()
        shell_layout = QHBoxLayout(shell_widget)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self.create_sidebar(), 0)
        shell_layout.addWidget(self.stacked_widget, 1)

        self.setCentralWidget(shell_widget)
        self.statusBar().showMessage("就绪")

    @staticmethod
    def scroll_page(widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet("""
            QFrame#sidebar { background-color: #111827; border: none; }
            QLabel { color: #E5E7EB; }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(10)

        title = QLabel("点云工具包")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(title)

        version = QLabel(f"v{APP_VERSION}")
        version.setStyleSheet("color: #9CA3AF; margin-bottom: 14px;")
        layout.addWidget(version)

        self.nav_buttons = []
        nav_items = [
            ("工作台", 0),
            ("点云处理", 1),
            ("格式转换", 2),
            ("预测分析", 3),
            ("下采样", 4),
        ]
        for text, index in nav_items:
            button = QPushButton(text)
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, page_index=index: self.switch_page(page_index))
            button.setStyleSheet(self.get_nav_button_style())
            layout.addWidget(button)
            self.nav_buttons.append(button)

        layout.addStretch(1)

        self.sidebar_status_label = QLabel("状态：就绪")
        self.sidebar_status_label.setWordWrap(True)
        self.sidebar_status_label.setStyleSheet(
            "color: #CBD5E1; background-color: #1F2937; border-radius: 6px; padding: 10px;"
        )
        layout.addWidget(self.sidebar_status_label)
        self._set_nav_active(0)
        return sidebar

    def switch_page(self, index):
        if self.is_worker_running():
            QMessageBox.warning(self, "操作受限", "请等待当前后台任务完成后再切换页面。")
            return
        self.stacked_widget.setCurrentIndex(index)
        self._set_nav_active(index)
        page_titles = ["主页", "辣椒植株处理", "格式转换与工具", "预测分析工具", "点云下采样"]
        if 0 <= index < len(page_titles):
            self.statusBar().showMessage(f"已切换到: {page_titles[index]}", 3000)

    def _set_nav_active(self, index):
        if not hasattr(self, "nav_buttons"):
            return
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)

    def create_pepper_page(self):
        pepper_widget = QWidget()
        main_layout = QVBoxLayout(pepper_widget);
        main_layout.setSpacing(10)
        header_layout = QHBoxLayout()
        title_label = QLabel("辣椒植株点云处理");
        title_label.setFont(QFont("Arial", 16, QFont.Bold));
        title_label.setStyleSheet("color: #2C3E50; margin-bottom: 5px;")
        header_layout.addWidget(title_label, 1)
        back_button = QPushButton(QIcon.fromTheme("go-home", QIcon(
            ":/qt-project.org/styles/commonstyle/images/standardbutton-go-home-32.png")), " 返回主页")
        back_button.clicked.connect(lambda: self.switch_page(0));
        back_button.setStyleSheet(self.get_button_style("danger"));
        back_button.setMinimumWidth(120)
        header_layout.addWidget(back_button, 0)
        main_layout.addLayout(header_layout)
        line = QFrame();
        line.setFrameShape(QFrame.HLine);
        line.setFrameShadow(QFrame.Sunken);
        line.setStyleSheet("background-color: #BDC3C7; margin: 5px 0px;")
        main_layout.addWidget(line)
        method_tab_widget = QTabWidget();
        method_tab_widget.setStyleSheet(self.get_tab_style())
        pipeline_page = self.create_pepper_method_subpage("pipeline", "植株流程化去噪")
        sor_page = self.create_pepper_method_subpage("sor", "统计离群值滤波 (SOR)")
        ror_page = self.create_pepper_method_subpage("ror", "半径离群值滤波 (ROR)")
        method_tab_widget.addTab(pipeline_page, "流程化去噪");
        method_tab_widget.addTab(sor_page, "统计滤波 (SOR)");
        method_tab_widget.addTab(ror_page, "半径滤波 (ROR)")
        main_layout.addWidget(method_tab_widget)
        return pepper_widget

    def create_pepper_method_subpage(self, method_type,
                                     method_title_unused):  # method_title_unused as it's set by tab now
        method_widget = QWidget()
        method_layout = QVBoxLayout(method_widget);
        method_layout.setContentsMargins(5, 10, 5, 5)
        sub_tab_widget = QTabWidget()
        single_file_tab = self.create_pepper_single_file_tab(method_type)  # Pass only method_type
        batch_file_tab = self.create_pepper_batch_tab(method_type)  # Pass only method_type
        sub_tab_widget.addTab(single_file_tab, "单个文件处理")
        sub_tab_widget.addTab(batch_file_tab, "批量处理")
        method_layout.addWidget(sub_tab_widget)
        return method_widget

    def create_pepper_single_file_tab(self, method_type):
        single_tab_widget = QWidget()
        single_file_layout = QVBoxLayout(single_tab_widget);
        single_file_layout.setSpacing(15)
        file_select_group = QGroupBox("文件选择");
        file_select_group.setStyleSheet(self.get_groupbox_style())
        file_select_layout = QVBoxLayout(file_select_group);
        file_select_layout.setSpacing(10)
        file_input_layout = QHBoxLayout()
        select_file_button = QPushButton("选择输入点云文件");
        select_file_button.clicked.connect(lambda: self.select_pepper_single_input_file(method_type))
        select_file_button.setStyleSheet(self.get_button_style("secondary"));
        select_file_button.setMinimumWidth(150)
        file_input_layout.addWidget(select_file_button)
        input_label = QLabel("输入文件: 未选择");
        input_label.setWordWrap(True);
        input_label.setStyleSheet("padding: 5px; color: #555;")
        file_input_layout.addWidget(input_label, 1)
        file_select_layout.addLayout(file_input_layout)
        file_output_layout = QHBoxLayout()
        select_output_button = QPushButton("选择结果输出目录");
        select_output_button.clicked.connect(lambda: self.select_pepper_single_output_dir(method_type))
        select_output_button.setStyleSheet(self.get_button_style("secondary"));
        select_output_button.setMinimumWidth(150)
        file_output_layout.addWidget(select_output_button)
        output_label = QLabel("输出目录: 未选择");
        output_label.setWordWrap(True);
        output_label.setStyleSheet("padding: 5px; color: #555;")
        file_output_layout.addWidget(output_label, 1)
        file_select_layout.addLayout(file_output_layout);
        single_file_layout.addWidget(file_select_group)
        param_group = QGroupBox("参数设置");
        param_group.setStyleSheet(self.get_groupbox_style())
        param_layout = QFormLayout(param_group);
        param_layout.setSpacing(10);
        param_layout.setContentsMargins(15, 10, 15, 10)
        if method_type == "pipeline":
            intermediate_checkbox = QCheckBox("保存流程中间点云");
            intermediate_checkbox.setChecked(False);
            intermediate_checkbox.setToolTip("保存包围盒过滤、DBSCAN过滤后的点云")
            param_layout.addRow(intermediate_checkbox);
            setattr(self, f"{method_type}_single_intermediate_checkbox", intermediate_checkbox)
        elif method_type == "sor":
            sor_neighbors_spin = QSpinBox();
            sor_neighbors_spin.setRange(1, 1000);
            sor_neighbors_spin.setValue(20);
            sor_neighbors_spin.setToolTip("统计分析时考虑的邻近点数量 K")
            param_layout.addRow("邻近点数 (K):", sor_neighbors_spin);
            setattr(self, f"{method_type}_single_neighbors", sor_neighbors_spin)
            sor_std_spin = QDoubleSpinBox();
            sor_std_spin.setRange(0.1, 10.0);
            sor_std_spin.setValue(1.0);
            sor_std_spin.setDecimals(2);
            sor_std_spin.setSingleStep(0.1);
            sor_std_spin.setToolTip("标准差倍数阈值。\n值越小，滤除点越多。")
            param_layout.addRow("标准差倍数:", sor_std_spin);
            setattr(self, f"{method_type}_single_std_ratio", sor_std_spin)
        elif method_type == "ror":
            ror_points_spin = QSpinBox();
            ror_points_spin.setRange(1, 1000);
            ror_points_spin.setValue(16);
            ror_points_spin.setToolTip("半径内所需的最少邻近点数量")
            param_layout.addRow("邻近点数:", ror_points_spin);
            setattr(self, f"{method_type}_single_nb_points", ror_points_spin)
            ror_radius_spin = QDoubleSpinBox();
            ror_radius_spin.setRange(0.001, 1.0);
            ror_radius_spin.setValue(0.05);
            ror_radius_spin.setDecimals(3);
            ror_radius_spin.setSingleStep(0.01);
            ror_radius_spin.setToolTip("用于查询邻近点的半径")
            param_layout.addRow("搜索半径:", ror_radius_spin);
            setattr(self, f"{method_type}_single_radius", ror_radius_spin)
        single_file_layout.addWidget(param_group)
        action_group = QGroupBox("操作");
        action_group.setStyleSheet(self.get_groupbox_style())
        action_buttons_layout = QHBoxLayout(action_group);
        action_buttons_layout.setSpacing(10)
        process_button = QPushButton(f"开始处理 ({method_type.upper()})");
        process_button.clicked.connect(lambda: self.process_single_pepper_file(method_type))
        process_button.setMinimumHeight(40);
        process_button.setStyleSheet(self.get_button_style("primary"))
        action_buttons_layout.addWidget(process_button)
        stop_button = QPushButton("停止处理");
        stop_button.clicked.connect(self.stop_current_worker)
        stop_button.setMinimumHeight(40);
        stop_button.setStyleSheet(self.get_button_style("danger"));
        stop_button.setEnabled(False)
        action_buttons_layout.addWidget(stop_button)
        compare_button = QPushButton("查看处理前后对比");
        compare_button.clicked.connect(lambda: self.visualize_before_after_comparison(method_type))
        compare_button.setMinimumHeight(40);
        compare_button.setStyleSheet(self.get_button_style("info"));
        compare_button.setEnabled(False)
        action_buttons_layout.addWidget(compare_button);
        single_file_layout.addWidget(action_group)
        status_group = QGroupBox("处理状态");
        status_group.setStyleSheet(self.get_groupbox_style())
        status_layout = QVBoxLayout(status_group)
        status_label = QLabel("准备就绪");
        status_label.setStyleSheet(self.get_status_label_style());
        status_label.setWordWrap(True);
        status_label.setAlignment(Qt.AlignCenter);
        status_label.setMinimumHeight(40)
        status_layout.addWidget(status_label)
        progress_bar = QProgressBar();
        progress_bar.setStyleSheet(self.get_progressbar_style());
        progress_bar.setMaximumHeight(18);
        progress_bar.setValue(0)
        status_layout.addWidget(progress_bar);
        single_file_layout.addWidget(status_group);
        single_file_layout.addStretch(1)
        setattr(self, f"{method_type}_single_input_label", input_label);
        setattr(self, f"{method_type}_single_output_label", output_label)
        setattr(self, f"{method_type}_single_process_button", process_button);
        setattr(self, f"{method_type}_single_stop_button", stop_button)
        setattr(self, f"{method_type}_single_compare_button", compare_button);
        setattr(self, f"{method_type}_single_status_label", status_label)
        setattr(self, f"{method_type}_single_progress_bar", progress_bar)
        return single_tab_widget

    def create_pepper_batch_tab(self, method_type):
        batch_tab_widget = QWidget()
        batch_layout = QVBoxLayout(batch_tab_widget);
        batch_layout.setSpacing(15)
        batch_group = QGroupBox("文件夹选择");
        batch_group.setStyleSheet(self.get_groupbox_style())
        batch_select_layout = QVBoxLayout(batch_group);
        batch_select_layout.setSpacing(10)
        batch_input_layout = QHBoxLayout()
        select_batch_input_button = QPushButton("选择输入文件夹");
        select_batch_input_button.clicked.connect(lambda: self.select_pepper_batch_input_folder(method_type))
        select_batch_input_button.setStyleSheet(self.get_button_style("secondary"));
        select_batch_input_button.setMinimumWidth(150)
        batch_input_layout.addWidget(select_batch_input_button)
        input_label = QLabel("输入文件夹: 未选择");
        input_label.setWordWrap(True);
        input_label.setStyleSheet("padding: 5px; color: #555;")
        batch_input_layout.addWidget(input_label, 1);
        batch_select_layout.addLayout(batch_input_layout)
        batch_output_layout = QHBoxLayout()
        select_batch_output_button = QPushButton("选择输出文件夹");
        select_batch_output_button.clicked.connect(lambda: self.select_pepper_batch_output_folder(method_type))
        select_batch_output_button.setStyleSheet(self.get_button_style("secondary"));
        select_batch_output_button.setMinimumWidth(150)
        batch_output_layout.addWidget(select_batch_output_button)
        output_label = QLabel("输出文件夹: 未选择");
        output_label.setWordWrap(True);
        output_label.setStyleSheet("padding: 5px; color: #555;")
        batch_output_layout.addWidget(output_label, 1);
        batch_select_layout.addLayout(batch_output_layout);
        batch_layout.addWidget(batch_group)
        param_group = QGroupBox("参数设置");
        param_group.setStyleSheet(self.get_groupbox_style())
        param_layout = QFormLayout(param_group);
        param_layout.setSpacing(10);
        param_layout.setContentsMargins(15, 10, 15, 10)
        if method_type == "pipeline":
            intermediate_checkbox = QCheckBox("为每个文件保存流程中间点云");
            intermediate_checkbox.setChecked(False);
            intermediate_checkbox.setToolTip("在输出目录为每个文件创建子目录并保存中间步骤")
            param_layout.addRow(intermediate_checkbox);
            setattr(self, f"{method_type}_batch_intermediate_checkbox", intermediate_checkbox)
        elif method_type == "sor":
            sor_neighbors_spin = QSpinBox();
            sor_neighbors_spin.setRange(1, 1000);
            sor_neighbors_spin.setValue(20);
            sor_neighbors_spin.setToolTip("统计分析时考虑的邻近点数量 K")
            param_layout.addRow("邻近点数 (K):", sor_neighbors_spin);
            setattr(self, f"{method_type}_batch_neighbors", sor_neighbors_spin)
            sor_std_spin = QDoubleSpinBox();
            sor_std_spin.setRange(0.1, 10.0);
            sor_std_spin.setValue(1.0);
            sor_std_spin.setDecimals(2);
            sor_std_spin.setSingleStep(0.1);
            sor_std_spin.setToolTip("标准差倍数阈值。\n值越小，滤除点越多。")
            param_layout.addRow("标准差倍数:", sor_std_spin);
            setattr(self, f"{method_type}_batch_std_ratio", sor_std_spin)
        elif method_type == "ror":
            ror_points_spin = QSpinBox();
            ror_points_spin.setRange(1, 1000);
            ror_points_spin.setValue(16);
            ror_points_spin.setToolTip("半径内所需的最少邻近点数量")
            param_layout.addRow("邻近点数:", ror_points_spin);
            setattr(self, f"{method_type}_batch_nb_points", ror_points_spin)
            ror_radius_spin = QDoubleSpinBox();
            ror_radius_spin.setRange(0.001, 1.0);
            ror_radius_spin.setValue(0.05);
            ror_radius_spin.setDecimals(3);
            ror_radius_spin.setSingleStep(0.01);
            ror_radius_spin.setToolTip("用于查询邻近点的半径")
            param_layout.addRow("搜索半径:", ror_radius_spin);
            setattr(self, f"{method_type}_batch_radius", ror_radius_spin)
        batch_layout.addWidget(param_group)
        action_group = QGroupBox("操作");
        action_group.setStyleSheet(self.get_groupbox_style())
        batch_process_layout = QHBoxLayout(action_group)
        process_button = QPushButton(f"开始批量处理 ({method_type.upper()})");
        process_button.clicked.connect(lambda: self.process_batch_pepper_files(method_type))
        process_button.setMinimumHeight(40);
        process_button.setStyleSheet(self.get_button_style("primary"))
        batch_process_layout.addWidget(process_button)
        stop_button = QPushButton("停止处理");
        stop_button.clicked.connect(self.stop_current_worker)
        stop_button.setMinimumHeight(40);
        stop_button.setStyleSheet(self.get_button_style("danger"));
        stop_button.setEnabled(False)
        batch_process_layout.addWidget(stop_button);
        batch_layout.addWidget(action_group)
        batch_status_group = QGroupBox("处理状态");
        batch_status_group.setStyleSheet(self.get_groupbox_style())
        batch_status_layout = QVBoxLayout(batch_status_group)
        status_label = QLabel("准备就绪");
        status_label.setStyleSheet(self.get_status_label_style());
        status_label.setWordWrap(True);
        status_label.setAlignment(Qt.AlignCenter);
        status_label.setMinimumHeight(40)
        batch_status_layout.addWidget(status_label)
        progress_bar = QProgressBar();
        progress_bar.setStyleSheet(self.get_progressbar_style());
        progress_bar.setMaximumHeight(18);
        progress_bar.setValue(0)
        batch_status_layout.addWidget(progress_bar);
        batch_layout.addWidget(batch_status_group);
        batch_layout.addStretch(1)
        setattr(self, f"{method_type}_batch_input_label", input_label);
        setattr(self, f"{method_type}_batch_output_label", output_label)
        setattr(self, f"{method_type}_batch_process_button", process_button);
        setattr(self, f"{method_type}_batch_stop_button", stop_button)
        setattr(self, f"{method_type}_batch_status_label", status_label);
        setattr(self, f"{method_type}_batch_progress_bar", progress_bar)
        return batch_tab_widget

    def create_convert_page(self):
        convert_widget = QWidget()
        layout = QVBoxLayout(convert_widget);
        layout.setSpacing(10)
        header_layout = QHBoxLayout()
        title_label = QLabel("格式转换与工具");
        title_label.setFont(QFont("Arial", 16, QFont.Bold));
        title_label.setStyleSheet("color: #2C3E50; margin-bottom: 5px;")
        header_layout.addWidget(title_label, 1)
        back_button = QPushButton(QIcon.fromTheme("go-home", QIcon(
            ":/qt-project.org/styles/commonstyle/images/standardbutton-go-home-32.png")), " 返回主页")
        back_button.clicked.connect(lambda: self.switch_page(0));
        back_button.setStyleSheet(self.get_button_style("danger"));
        back_button.setMinimumWidth(120)
        header_layout.addWidget(back_button, 0);
        layout.addLayout(header_layout)
        line = QFrame();
        line.setFrameShape(QFrame.HLine);
        line.setFrameShadow(QFrame.Sunken);
        line.setStyleSheet("background-color: #BDC3C7; margin: 5px 0px;");
        layout.addWidget(line)

        tab_widget = QTabWidget();
        tab_widget.setStyleSheet(self.get_tab_style())

        # == Generic Single File Conversion Tab ==
        single_conv_tab = QWidget();
        single_conv_layout = QVBoxLayout(single_conv_tab);
        single_conv_layout.setSpacing(15)
        # ... (UI for generic single file conversion - largely unchanged) ...
        single_conv_io_group = QGroupBox("文件选择");
        single_conv_io_group.setStyleSheet(self.get_groupbox_style())
        single_conv_io_layout = QVBoxLayout(single_conv_io_group);
        single_conv_io_layout.setSpacing(10)
        s_conv_input_layout = QHBoxLayout()
        select_s_conv_input_button = QPushButton("选择输入点云文件");
        select_s_conv_input_button.clicked.connect(self.select_generic_single_input_file)
        select_s_conv_input_button.setStyleSheet(self.get_button_style("secondary"));
        select_s_conv_input_button.setMinimumWidth(150)
        s_conv_input_layout.addWidget(select_s_conv_input_button)
        self.generic_single_input_label = QLabel("输入文件: 未选择");
        self.generic_single_input_label.setWordWrap(True);
        self.generic_single_input_label.setStyleSheet("padding: 5px; color: #555;")
        s_conv_input_layout.addWidget(self.generic_single_input_label, 1);
        single_conv_io_layout.addLayout(s_conv_input_layout)
        s_conv_output_layout = QHBoxLayout()
        select_s_conv_output_button = QPushButton("选择输出目录");
        select_s_conv_output_button.clicked.connect(self.select_generic_single_output_dir)
        select_s_conv_output_button.setStyleSheet(self.get_button_style("secondary"));
        select_s_conv_output_button.setMinimumWidth(150)
        s_conv_output_layout.addWidget(select_s_conv_output_button)
        self.generic_single_output_label = QLabel("输出目录: 未选择");
        self.generic_single_output_label.setWordWrap(True);
        self.generic_single_output_label.setStyleSheet("padding: 5px; color: #555;")
        s_conv_output_layout.addWidget(self.generic_single_output_label, 1);
        single_conv_io_layout.addLayout(s_conv_output_layout);
        single_conv_layout.addWidget(single_conv_io_group)
        single_conv_action_group = QGroupBox("执行转换");
        single_conv_action_group.setStyleSheet(self.get_groupbox_style())
        single_conv_action_layout_v = QVBoxLayout(single_conv_action_group)
        single_conv_action_layout_h = QHBoxLayout();
        single_conv_action_layout_h.setSpacing(10)
        self.s_conv_pcd_button = QPushButton("转换为 PCD");
        self.s_conv_pcd_button.clicked.connect(lambda: self.start_generic_single_conversion("pcd"));
        self.s_conv_pcd_button.setStyleSheet(self.get_button_style("primary"));
        self.s_conv_pcd_button.setMinimumHeight(35);
        single_conv_action_layout_h.addWidget(self.s_conv_pcd_button)
        self.s_conv_ply_button = QPushButton("转换为 PLY");
        self.s_conv_ply_button.clicked.connect(lambda: self.start_generic_single_conversion("ply"));
        self.s_conv_ply_button.setStyleSheet(self.get_button_style("primary"));
        self.s_conv_ply_button.setMinimumHeight(35);
        single_conv_action_layout_h.addWidget(self.s_conv_ply_button)
        self.s_conv_xyz_button = QPushButton("转换为 XYZ");
        self.s_conv_xyz_button.clicked.connect(lambda: self.start_generic_single_conversion("xyz"));
        self.s_conv_xyz_button.setStyleSheet(self.get_button_style("primary"));
        self.s_conv_xyz_button.setMinimumHeight(35);
        single_conv_action_layout_h.addWidget(self.s_conv_xyz_button)
        single_conv_action_layout_v.addLayout(single_conv_action_layout_h)
        self.stop_generic_single_button = QPushButton("停止转换");
        self.stop_generic_single_button.clicked.connect(self.stop_current_worker);
        self.stop_generic_single_button.setMinimumHeight(35);
        self.stop_generic_single_button.setStyleSheet(self.get_button_style("danger"));
        self.stop_generic_single_button.setEnabled(False);
        single_conv_action_layout_v.addWidget(self.stop_generic_single_button);
        single_conv_layout.addWidget(single_conv_action_group)
        s_conv_status_group = QGroupBox("转换状态");
        s_conv_status_group.setStyleSheet(self.get_groupbox_style());
        s_conv_status_layout = QVBoxLayout(s_conv_status_group)
        self.generic_single_status = QLabel("准备就绪");
        self.generic_single_status.setStyleSheet(self.get_status_label_style());
        self.generic_single_status.setWordWrap(True);
        self.generic_single_status.setAlignment(Qt.AlignCenter);
        self.generic_single_status.setMinimumHeight(40);
        s_conv_status_layout.addWidget(self.generic_single_status)
        self.generic_single_progress = QProgressBar();
        self.generic_single_progress.setStyleSheet(self.get_progressbar_style());
        self.generic_single_progress.setMaximumHeight(18);
        self.generic_single_progress.setValue(0);
        s_conv_status_layout.addWidget(self.generic_single_progress);
        single_conv_layout.addWidget(s_conv_status_group);
        single_conv_layout.addStretch(1)
        tab_widget.addTab(single_conv_tab, "单文件格式转换")

        # == Generic Batch Conversion Tab ==
        batch_conv_tab = QWidget();
        batch_conv_layout = QVBoxLayout(batch_conv_tab);
        batch_conv_layout.setSpacing(15)
        # ... (UI for generic batch conversion - largely unchanged) ...
        batch_conv_io_group = QGroupBox("文件夹选择");
        batch_conv_io_group.setStyleSheet(self.get_groupbox_style());
        batch_conv_io_layout = QVBoxLayout(batch_conv_io_group);
        batch_conv_io_layout.setSpacing(10)
        b_conv_input_layout = QHBoxLayout();
        select_b_conv_input_button = QPushButton("选择输入文件夹");
        select_b_conv_input_button.clicked.connect(self.select_generic_batch_input_dir);
        select_b_conv_input_button.setStyleSheet(self.get_button_style("secondary"));
        select_b_conv_input_button.setMinimumWidth(150);
        b_conv_input_layout.addWidget(select_b_conv_input_button)
        self.generic_batch_input_label = QLabel("输入文件夹: 未选择");
        self.generic_batch_input_label.setWordWrap(True);
        self.generic_batch_input_label.setStyleSheet("padding: 5px; color: #555;");
        b_conv_input_layout.addWidget(self.generic_batch_input_label, 1);
        batch_conv_io_layout.addLayout(b_conv_input_layout)
        b_conv_output_layout = QHBoxLayout();
        select_b_conv_output_button = QPushButton("选择输出文件夹");
        select_b_conv_output_button.clicked.connect(self.select_generic_batch_output_dir);
        select_b_conv_output_button.setStyleSheet(self.get_button_style("secondary"));
        select_b_conv_output_button.setMinimumWidth(150);
        b_conv_output_layout.addWidget(select_b_conv_output_button)
        self.generic_batch_output_label = QLabel("输出文件夹: 未选择");
        self.generic_batch_output_label.setWordWrap(True);
        self.generic_batch_output_label.setStyleSheet("padding: 5px; color: #555;");
        b_conv_output_layout.addWidget(self.generic_batch_output_label, 1);
        batch_conv_io_layout.addLayout(b_conv_output_layout);
        batch_conv_layout.addWidget(batch_conv_io_group)
        batch_conv_action_group = QGroupBox("执行批量转换");
        batch_conv_action_group.setStyleSheet(self.get_groupbox_style());
        batch_conv_action_layout_v = QVBoxLayout(batch_conv_action_group)
        batch_conv_action_layout_h = QHBoxLayout();
        batch_conv_action_layout_h.setSpacing(10)
        self.b_conv_pcd_button = QPushButton("全部转换为 PCD");
        self.b_conv_pcd_button.clicked.connect(lambda: self.start_generic_batch_conversion("pcd"));
        self.b_conv_pcd_button.setStyleSheet(self.get_button_style("primary"));
        self.b_conv_pcd_button.setMinimumHeight(35);
        batch_conv_action_layout_h.addWidget(self.b_conv_pcd_button)
        self.b_conv_ply_button = QPushButton("全部转换为 PLY");
        self.b_conv_ply_button.clicked.connect(lambda: self.start_generic_batch_conversion("ply"));
        self.b_conv_ply_button.setStyleSheet(self.get_button_style("primary"));
        self.b_conv_ply_button.setMinimumHeight(35);
        batch_conv_action_layout_h.addWidget(self.b_conv_ply_button)
        self.b_conv_xyz_button = QPushButton("全部转换为 XYZ");
        self.b_conv_xyz_button.clicked.connect(lambda: self.start_generic_batch_conversion("xyz"));
        self.b_conv_xyz_button.setStyleSheet(self.get_button_style("primary"));
        self.b_conv_xyz_button.setMinimumHeight(35);
        batch_conv_action_layout_h.addWidget(self.b_conv_xyz_button)
        batch_conv_action_layout_v.addLayout(batch_conv_action_layout_h)
        self.stop_generic_batch_button = QPushButton("停止转换");
        self.stop_generic_batch_button.clicked.connect(self.stop_current_worker);
        self.stop_generic_batch_button.setMinimumHeight(35);
        self.stop_generic_batch_button.setStyleSheet(self.get_button_style("danger"));
        self.stop_generic_batch_button.setEnabled(False);
        batch_conv_action_layout_v.addWidget(self.stop_generic_batch_button);
        batch_conv_layout.addWidget(batch_conv_action_group)
        b_conv_status_group = QGroupBox("转换状态");
        b_conv_status_group.setStyleSheet(self.get_groupbox_style());
        b_conv_status_layout = QVBoxLayout(b_conv_status_group)
        self.generic_batch_status = QLabel("准备就绪");
        self.generic_batch_status.setStyleSheet(self.get_status_label_style());
        self.generic_batch_status.setWordWrap(True);
        self.generic_batch_status.setAlignment(Qt.AlignCenter);
        self.generic_batch_status.setMinimumHeight(40);
        b_conv_status_layout.addWidget(self.generic_batch_status)
        self.generic_batch_progress = QProgressBar();
        self.generic_batch_progress.setStyleSheet(self.get_progressbar_style());
        self.generic_batch_progress.setMaximumHeight(18);
        self.generic_batch_progress.setValue(0);
        b_conv_status_layout.addWidget(self.generic_batch_progress);
        batch_conv_layout.addWidget(b_conv_status_group);
        batch_conv_layout.addStretch(1)
        tab_widget.addTab(batch_conv_tab, "批量格式转换")

        # == ScanNet Conversion Tab (New) ==
        scannet_tab = QWidget()
        scannet_layout_main = QVBoxLayout(scannet_tab);
        scannet_layout_main.setSpacing(15)
        scannet_group = QGroupBox("转换为 ScanNet 格式");
        scannet_group.setStyleSheet(self.get_groupbox_style())
        scannet_form_layout = QFormLayout(scannet_group);
        scannet_form_layout.setSpacing(10)

        scannet_info = QLabel("将输入文件夹中的 TXT 文件 (格式: X Y Z R G B Label NormX NormY NormZ - 共10列) "
                              "批量转换为 ScanNet 格式的 .npy 文件，并按比例分割为训练/验证/测试集。\n"
                              "输出目录将包含 scannetv2_train.txt, scannetv2_val.txt, scannetv2_test.txt "
                              "以及每个场景ID命名的子文件夹（如 scene0000_00），内含 coord.npy, color.npy 等文件。")
        scannet_info.setWordWrap(True);
        scannet_info.setStyleSheet(
            "margin-bottom: 10px; color: #555; background-color: #f8f9f9; border: 1px solid #e1e5e8; border-radius: 3px; padding: 8px;")
        scannet_form_layout.addRow(scannet_info)

        # Input Dir
        self.scannet_input_label = QLabel("输入文件夹: 未选择");
        self.scannet_input_label.setWordWrap(True)
        btn_select_scannet_input = QPushButton("选择TXT输入文件夹");
        btn_select_scannet_input.clicked.connect(self.select_scannet_input_folder)
        btn_select_scannet_input.setStyleSheet(self.get_button_style("secondary"))
        scannet_form_layout.addRow(btn_select_scannet_input, self.scannet_input_label)

        # Output Dir
        self.scannet_output_label = QLabel("ScanNet输出文件夹: 未选择");
        self.scannet_output_label.setWordWrap(True)
        btn_select_scannet_output = QPushButton("选择ScanNet输出文件夹");
        btn_select_scannet_output.clicked.connect(self.select_scannet_output_folder)
        btn_select_scannet_output.setStyleSheet(self.get_button_style("secondary"))
        scannet_form_layout.addRow(btn_select_scannet_output, self.scannet_output_label)

        # Split Ratio
        self.scannet_split_ratio_edit = QLineEdit("0.7, 0.2, 0.1")
        self.scannet_split_ratio_edit.setToolTip("训练、验证、测试集的分割比例，以逗号分隔，总和应为1.0")
        scannet_form_layout.addRow("分割比例 (训练,验证,测试):", self.scannet_split_ratio_edit)

        # Seed
        self.scannet_seed_edit = QLineEdit("42")
        self.scannet_seed_edit.setToolTip("用于打乱文件顺序的随机种子 (整数)")
        scannet_form_layout.addRow("随机种子:", self.scannet_seed_edit)

        scannet_action_layout = QHBoxLayout()
        self.scannet_convert_button = QPushButton("开始转换为 ScanNet 格式");
        self.scannet_convert_button.clicked.connect(self.start_scannet_conversion)
        self.scannet_convert_button.setMinimumHeight(40);
        self.scannet_convert_button.setStyleSheet(self.get_button_style("primary", font_size="11pt"))
        scannet_action_layout.addWidget(self.scannet_convert_button)
        self.stop_scannet_button = QPushButton("停止转换");
        self.stop_scannet_button.clicked.connect(self.stop_current_worker)
        self.stop_scannet_button.setMinimumHeight(40);
        self.stop_scannet_button.setStyleSheet(self.get_button_style("danger"));
        self.stop_scannet_button.setEnabled(False)
        scannet_action_layout.addWidget(self.stop_scannet_button)
        scannet_form_layout.addRow(scannet_action_layout)
        scannet_layout_main.addWidget(scannet_group)

        scannet_status_group = QGroupBox("转换状态");
        scannet_status_group.setStyleSheet(self.get_groupbox_style())
        scannet_status_layout = QVBoxLayout(scannet_status_group)
        self.scannet_status_label = QLabel("准备就绪");
        self.scannet_status_label.setStyleSheet(self.get_status_label_style());
        self.scannet_status_label.setWordWrap(True);
        self.scannet_status_label.setAlignment(Qt.AlignCenter);
        self.scannet_status_label.setMinimumHeight(40)
        scannet_status_layout.addWidget(self.scannet_status_label)
        self.scannet_progress_bar = QProgressBar();
        self.scannet_progress_bar.setStyleSheet(self.get_progressbar_style());
        self.scannet_progress_bar.setMaximumHeight(18);
        self.scannet_progress_bar.setValue(0)
        scannet_status_layout.addWidget(self.scannet_progress_bar)
        scannet_layout_main.addWidget(scannet_status_group)
        scannet_layout_main.addStretch(1)
        tab_widget.addTab(scannet_tab, "ScanNet 格式转换")

        # == Check NPY Shape Tab (New) ==
        npy_shape_tab = QWidget()
        npy_shape_layout = QVBoxLayout(npy_shape_tab);
        npy_shape_layout.setSpacing(15)
        npy_shape_group = QGroupBox("检查 .npy 文件形状");
        npy_shape_group.setStyleSheet(self.get_groupbox_style())
        npy_shape_form = QFormLayout(npy_shape_group);
        npy_shape_form.setSpacing(10)

        self.npy_shape_input_label = QLabel("NPY文件: 未选择");
        self.npy_shape_input_label.setWordWrap(True)
        btn_select_npy_shape_input = QPushButton("选择 .npy 文件");
        btn_select_npy_shape_input.clicked.connect(self.select_npy_shape_input_file)
        btn_select_npy_shape_input.setStyleSheet(self.get_button_style("secondary"))
        npy_shape_form.addRow(btn_select_npy_shape_input, self.npy_shape_input_label)

        self.npy_shape_check_button = QPushButton("检查形状");
        self.npy_shape_check_button.clicked.connect(self.start_npy_shape_check)
        self.npy_shape_check_button.setMinimumHeight(35);
        self.npy_shape_check_button.setStyleSheet(self.get_button_style("info"))
        npy_shape_form.addRow(self.npy_shape_check_button)

        self.npy_shape_result_textedit = QTextEdit();
        self.npy_shape_result_textedit.setReadOnly(True)
        self.npy_shape_result_textedit.setFont(QFont("Courier New", 10));
        self.npy_shape_result_textedit.setMinimumHeight(100)
        npy_shape_form.addRow(QLabel("结果:"), self.npy_shape_result_textedit)

        npy_shape_layout.addWidget(npy_shape_group)

        npy_shape_status_group = QGroupBox("检查状态");
        npy_shape_status_group.setStyleSheet(self.get_groupbox_style())
        npy_shape_status_layout_v = QVBoxLayout(npy_shape_status_group)
        self.npy_shape_status_label = QLabel("准备就绪");
        self.npy_shape_status_label.setStyleSheet(self.get_status_label_style());
        self.npy_shape_status_label.setWordWrap(True);
        self.npy_shape_status_label.setAlignment(Qt.AlignCenter)
        npy_shape_status_layout_v.addWidget(self.npy_shape_status_label)
        npy_shape_layout.addWidget(npy_shape_status_group)
        npy_shape_layout.addStretch(1)
        tab_widget.addTab(npy_shape_tab, ".npy 文件检查")

        layout.addWidget(tab_widget)
        return convert_widget

    def create_prediction_analysis_page(self):
        analysis_widget = QWidget()
        main_layout = QVBoxLayout(analysis_widget);
        main_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        title_label = QLabel("预测分析工具");
        title_label.setFont(QFont("Arial", 16, QFont.Bold));
        title_label.setStyleSheet("color: #2C3E50; margin-bottom: 5px;")
        header_layout.addWidget(title_label, 1)
        back_button = QPushButton(QIcon.fromTheme("go-home", QIcon(
            ":/qt-project.org/styles/commonstyle/images/standardbutton-go-home-32.png")), " 返回主页")
        back_button.clicked.connect(lambda: self.switch_page(0));
        back_button.setStyleSheet(self.get_button_style("danger"));
        back_button.setMinimumWidth(120)
        header_layout.addWidget(back_button, 0)
        main_layout.addLayout(header_layout)
        line = QFrame();
        line.setFrameShape(QFrame.HLine);
        line.setFrameShadow(QFrame.Sunken);
        line.setStyleSheet("background-color: #BDC3C7; margin: 5px 0px;")
        main_layout.addWidget(line)

        # --- Merge Predictions Group ---
        merge_group = QGroupBox("合并预测标签到点云");
        merge_group.setStyleSheet(self.get_groupbox_style())
        merge_layout = QFormLayout(merge_group);
        merge_layout.setSpacing(10)

        merge_info = QLabel("将预测的标签（每行一个整数标签）合并到原始点云TXT文件的第7列（索引6）。\n"
                            "原始点云文件格式应为 (N, 10): X Y Z R G B OrigLabel NormX NormY NormZ。\n"
                            "预测标签文件格式为 (N, 1): PredLabel。若标签从1开始，会自动减1处理。")
        merge_info.setWordWrap(True);
        merge_info.setStyleSheet(self.get_info_label_style())
        merge_layout.addRow(merge_info)

        self.merge_orig_label = QLabel("原始点云TXT: 未选择");
        self.merge_orig_label.setWordWrap(True)
        btn_sel_merge_orig = QPushButton("选择原始点云文件");
        btn_sel_merge_orig.clicked.connect(self.select_merge_original_txt)
        btn_sel_merge_orig.setStyleSheet(self.get_button_style("secondary"))
        merge_layout.addRow(btn_sel_merge_orig, self.merge_orig_label)

        self.merge_pred_label = QLabel("预测标签TXT: 未选择");
        self.merge_pred_label.setWordWrap(True)
        btn_sel_merge_pred = QPushButton("选择预测标签文件");
        btn_sel_merge_pred.clicked.connect(self.select_merge_prediction_txt)
        btn_sel_merge_pred.setStyleSheet(self.get_button_style("secondary"))
        merge_layout.addRow(btn_sel_merge_pred, self.merge_pred_label)

        self.merge_output_label = QLabel("合并后输出TXT: 未选择");
        self.merge_output_label.setWordWrap(True)
        btn_sel_merge_output = QPushButton("选择合并后保存路径");
        btn_sel_merge_output.clicked.connect(self.select_merge_output_txt)
        btn_sel_merge_output.setStyleSheet(self.get_button_style("secondary"))
        merge_layout.addRow(btn_sel_merge_output, self.merge_output_label)

        merge_action_layout = QHBoxLayout()
        self.merge_start_button = QPushButton("开始合并");
        self.merge_start_button.clicked.connect(self.start_merge_predictions)
        self.merge_start_button.setMinimumHeight(35);
        self.merge_start_button.setStyleSheet(self.get_button_style("primary"))
        merge_action_layout.addWidget(self.merge_start_button)
        self.merge_stop_button = QPushButton("停止合并");
        self.merge_stop_button.clicked.connect(self.stop_current_worker)
        self.merge_stop_button.setMinimumHeight(35);
        self.merge_stop_button.setStyleSheet(self.get_button_style("danger"));
        self.merge_stop_button.setEnabled(False)
        merge_action_layout.addWidget(self.merge_stop_button)
        merge_layout.addRow(merge_action_layout)

        self.merge_status_label = QLabel("准备就绪");
        self.merge_status_label.setStyleSheet(self.get_status_label_style());
        self.merge_status_label.setWordWrap(True);
        self.merge_status_label.setAlignment(Qt.AlignCenter)
        merge_layout.addRow(QLabel("状态:"), self.merge_status_label)  # Progress bar might be too much for quick merge
        main_layout.addWidget(merge_group)

        # --- Visualize with Custom Colors Group ---
        vis_group = QGroupBox("自定义颜色可视化");
        vis_group.setStyleSheet(self.get_groupbox_style())
        vis_layout = QFormLayout(vis_group);
        vis_layout.setSpacing(10)
        vis_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        vis_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        vis_info = QLabel("选择文件与数据排列，再按标签、原始 RGB 或单色显示。截图为可选项。")
        vis_info.setWordWrap(True);
        vis_info.setStyleSheet(self.get_info_label_style())
        vis_layout.addRow(vis_info)

        self.vis_custom_input_label = QLabel("待可视化文件: 未选择");
        self.vis_custom_input_label.setWordWrap(True)
        self.vis_custom_input_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        btn_sel_vis_input = QPushButton("选择点云文件");
        btn_sel_vis_input.clicked.connect(self.select_vis_custom_input_txt)
        btn_sel_vis_input.setStyleSheet(self.get_button_style("secondary"))
        vis_layout.addRow(btn_sel_vis_input, self.vis_custom_input_label)

        self.vis_format_widget = VisualizationFormatWidget()
        vis_layout.addRow(self.vis_format_widget)

        self.vis_color_map_widget = QWidget()
        color_layout = QVBoxLayout(self.vis_color_map_widget)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_help = QLabel("标签颜色（每行：标签号: [R, G, B]；未指定时为灰色）")
        color_help.setWordWrap(True)
        color_layout.addWidget(color_help)
        self.vis_custom_color_map_edit = QTextEdit()
        self.vis_custom_color_map_edit.setFont(QFont("Courier New", 10))
        self.vis_custom_color_map_edit.setPlainText(
            "0: [0, 0, 255]    # 标签0：蓝色\n1: [255, 0, 0]    # 标签1：红色\n2: [255, 165, 0]  # 标签2：橙色\n3: [0, 180, 0]    # 标签3：绿色")
        self.vis_custom_color_map_edit.setFixedHeight(110)
        color_layout.addWidget(self.vis_custom_color_map_edit)
        vis_layout.addRow(self.vis_color_map_widget)
        self.vis_format_widget.color_mode_changed.connect(
            lambda mode: self.vis_color_map_widget.setVisible(mode == "label"))

        self.vis_custom_screenshot_label = QLabel("截图保存路径: 未选择 (可选)");
        self.vis_custom_screenshot_label.setWordWrap(True)
        btn_sel_screenshot = QPushButton("选择截图保存路径");
        btn_sel_screenshot.clicked.connect(self.select_vis_custom_screenshot_path)
        btn_sel_screenshot.setStyleSheet(self.get_button_style("secondary"))
        screenshot_buttons = QHBoxLayout()
        screenshot_buttons.addWidget(btn_sel_screenshot)
        clear_screenshot = QPushButton("不保存截图")
        clear_screenshot.setStyleSheet(self.get_button_style("secondary"))
        clear_screenshot.clicked.connect(self.clear_vis_custom_screenshot)
        screenshot_buttons.addWidget(clear_screenshot)
        screenshot_buttons.addStretch()
        vis_layout.addRow(screenshot_buttons)
        vis_layout.addRow(self.vis_custom_screenshot_label)

        self.vis_custom_start_button = QPushButton("开始可视化");
        self.vis_custom_start_button.clicked.connect(self.start_custom_color_visualization)
        self.vis_custom_start_button.setMinimumHeight(35);
        self.vis_custom_start_button.setStyleSheet(self.get_button_style("primary"))
        vis_layout.addRow(self.vis_custom_start_button)

        self.vis_custom_status_label = QLabel("准备就绪");
        self.vis_custom_status_label.setStyleSheet(self.get_status_label_style());
        self.vis_custom_status_label.setWordWrap(True);
        self.vis_custom_status_label.setAlignment(Qt.AlignCenter)
        vis_layout.addRow(QLabel("状态:"), self.vis_custom_status_label)
        self.custom_visualization_signals.status.connect(self.vis_custom_status_label.setText)
        self.custom_visualization_signals.finished.connect(self.on_custom_visualization_finished)
        main_layout.addWidget(vis_group)
        main_layout.addStretch(1)
        return analysis_widget

    def create_downsample_page(self):
        page_widget = QWidget()
        main_layout = QVBoxLayout(page_widget);
        main_layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("点云下采样工具");
        title_label.setFont(QFont("Arial", 16, QFont.Bold));
        title_label.setStyleSheet("color: #2C3E50; margin-bottom: 5px;")
        header_layout.addWidget(title_label, 1)
        back_button = QPushButton(QIcon.fromTheme("go-home"), " 返回主页")
        back_button.clicked.connect(lambda: self.switch_page(0));
        back_button.setStyleSheet(self.get_button_style("danger"));
        back_button.setMinimumWidth(120)
        header_layout.addWidget(back_button, 0)
        main_layout.addLayout(header_layout)
        line = QFrame();
        line.setFrameShape(QFrame.HLine);
        line.setFrameShadow(QFrame.Sunken);
        line.setStyleSheet("background-color: #BDC3C7; margin: 5px 0px;");
        main_layout.addWidget(line)

        # Tabs for Single/Batch
        tab_widget = QTabWidget();
        tab_widget.setStyleSheet(self.get_tab_style())

        # We can reuse a generic creator for single and batch tabs
        single_tab = self._create_downsample_tab(is_batch=False)
        batch_tab = self._create_downsample_tab(is_batch=True)

        tab_widget.addTab(single_tab, "单个文件处理")
        tab_widget.addTab(batch_tab, "批量处理")

        main_layout.addWidget(tab_widget)
        return page_widget

    def _create_downsample_tab(self, is_batch):
        """Helper to create the content for single or batch downsampling tabs."""
        mode = "batch" if is_batch else "single"
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget);
        layout.setSpacing(15)

        # --- I/O Group ---
        io_group = QGroupBox("文件/文件夹选择" if is_batch else "文件选择");
        io_group.setStyleSheet(self.get_groupbox_style())
        io_layout = QFormLayout(io_group)

        if is_batch:
            # Batch Input
            input_label = QLabel("输入文件夹: 未选择");
            input_label.setWordWrap(True)
            btn_select_input = QPushButton("选择输入文件夹");
            btn_select_input.setStyleSheet(self.get_button_style("secondary"))
            btn_select_input.clicked.connect(self.select_downsample_batch_input_dir)
            io_layout.addRow(btn_select_input, input_label)
            # Batch Output
            output_label = QLabel("输出文件夹: 未选择");
            output_label.setWordWrap(True)
            btn_select_output = QPushButton("选择输出文件夹");
            btn_select_output.setStyleSheet(self.get_button_style("secondary"))
            btn_select_output.clicked.connect(self.select_downsample_batch_output_dir)
            io_layout.addRow(btn_select_output, output_label)
        else:
            # Single Input
            input_label = QLabel("输入TXT文件: 未选择");
            input_label.setWordWrap(True)
            btn_select_input = QPushButton("选择输入文件");
            btn_select_input.setStyleSheet(self.get_button_style("secondary"))
            btn_select_input.clicked.connect(self.select_downsample_single_input_file)
            io_layout.addRow(btn_select_input, input_label)
            # Single Output
            output_label = QLabel("输出TXT文件: 未选择");
            output_label.setWordWrap(True)
            btn_select_output = QPushButton("选择保存路径");
            btn_select_output.setStyleSheet(self.get_button_style("secondary"))
            btn_select_output.clicked.connect(self.select_downsample_single_output_file)
            io_layout.addRow(btn_select_output, output_label)
        layout.addWidget(io_group)

        # --- Parameters Group ---
        param_group = QGroupBox("参数设置");
        param_group.setStyleSheet(self.get_groupbox_style())
        param_layout = QFormLayout(param_group)

        method_combo = QComboBox()
        method_combo.addItems(["随机采样 (Random)", "体素网格 (Voxel)", "均匀采样 (Uniform)", "最远点采样 (Farthest)"])
        method_combo.setToolTip(
            "随机/均匀采样：速度快，按目标点数取样。\n最远点采样：分布均匀，大文件耗时较长。\n体素网格：每个体素保留一个原始点，保留全部颜色、标签和特征列。")
        param_layout.addRow("采样方法:", method_combo)

        # Parameter stack
        param_stack = QStackedWidget()
        # Widget 0: Target Points (for random, uniform, farthest)
        target_points_widget = QWidget()
        target_points_layout = QFormLayout(target_points_widget);
        target_points_layout.setContentsMargins(0, 0, 0, 0)
        target_points_spin = QSpinBox();
        target_points_spin.setRange(100, 500000);
        target_points_spin.setValue(10000);
        target_points_spin.setSingleStep(1000)
        target_points_spin.setToolTip("目标点数。采样后点云将包含大约这个数量的点。")
        target_points_layout.addRow("目标点数:", target_points_spin)
        param_stack.addWidget(target_points_widget)

        # Widget 1: Voxel Size
        voxel_size_widget = QWidget()
        voxel_size_layout = QFormLayout(voxel_size_widget);
        voxel_size_layout.setContentsMargins(0, 0, 0, 0)
        voxel_size_spin = QDoubleSpinBox();
        voxel_size_spin.setRange(0.001, 5.0);
        voxel_size_spin.setValue(0.05);
        voxel_size_spin.setDecimals(4);
        voxel_size_spin.setSingleStep(0.01)
        voxel_size_spin.setToolTip("体素立方体的边长。\n值越小，保留的点越多，细节越丰富。")
        voxel_size_layout.addRow("体素大小:", voxel_size_spin)
        param_stack.addWidget(voxel_size_widget)

        param_layout.addRow(param_stack)
        method_combo.currentIndexChanged.connect(
            lambda index, stack=param_stack: stack.setCurrentIndex(1 if index == 1 else 0))
        layout.addWidget(param_group)

        # --- Action & Status ---
        action_group = QGroupBox("操作");
        action_group.setStyleSheet(self.get_groupbox_style())
        action_layout = QHBoxLayout(action_group)
        start_button = QPushButton("开始处理");
        start_button.setStyleSheet(self.get_button_style("primary"));
        start_button.setMinimumHeight(40)
        start_button.clicked.connect(lambda: self.start_downsample(is_batch))
        stop_button = QPushButton("停止处理");
        stop_button.setStyleSheet(self.get_button_style("danger"));
        stop_button.setMinimumHeight(40);
        stop_button.setEnabled(False)
        stop_button.clicked.connect(self.stop_current_worker)
        action_layout.addWidget(start_button);
        action_layout.addWidget(stop_button)
        layout.addWidget(action_group)

        status_group = QGroupBox("处理状态");
        status_group.setStyleSheet(self.get_groupbox_style())
        status_layout = QVBoxLayout(status_group)
        status_label = QLabel("准备就绪");
        status_label.setStyleSheet(self.get_status_label_style());
        status_label.setWordWrap(True);
        status_label.setAlignment(Qt.AlignCenter);
        status_label.setMinimumHeight(40)
        progress_bar = QProgressBar();
        progress_bar.setStyleSheet(self.get_progressbar_style());
        progress_bar.setMaximumHeight(18);
        progress_bar.setValue(0)
        status_layout.addWidget(status_label);
        status_layout.addWidget(progress_bar)
        layout.addWidget(status_group)

        layout.addStretch(1)

        # Store widgets for later access
        setattr(self, f"downsample_{mode}_input_label", input_label)
        setattr(self, f"downsample_{mode}_output_label", output_label)
        setattr(self, f"downsample_{mode}_method_combo", method_combo)
        setattr(self, f"downsample_{mode}_target_points_spin", target_points_spin)
        setattr(self, f"downsample_{mode}_voxel_size_spin", voxel_size_spin)
        setattr(self, f"downsample_{mode}_start_button", start_button)
        setattr(self, f"downsample_{mode}_stop_button", stop_button)
        setattr(self, f"downsample_{mode}_status_label", status_label)
        setattr(self, f"downsample_{mode}_progress_bar", progress_bar)

        return tab_widget
    # --- Helper methods for styling ---
    def get_groupbox_style(self):
        return (
            "QGroupBox { background-color: #FFFFFF; font-weight: bold; border: 1px solid #D8DEE6; "
            "border-radius: 6px; margin-top: 12px; padding-top: 16px; } "
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; "
            "padding: 0 8px; background-color: #FFFFFF; color: #243447; }")

    def get_tab_style(self):
        return """
            QTabWidget::pane { border: 1px solid #D8DEE6; margin-top: -1px; background-color: #FFFFFF; border-radius: 0 6px 6px 6px; }
            QTabBar::tab { padding: 9px 18px; background-color: #EEF2F6; border: 1px solid #D8DEE6; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; min-width: 96px; margin-right: 2px; color: #425466; }
            QTabBar::tab:selected { background-color: #FFFFFF; border-bottom: 1px solid #FFFFFF; margin-bottom: -1px; font-weight: bold; color: #17202A; }
            QTabBar::tab:hover { background-color: #E2E8F0; border-color: #CBD5E1; }
            QTabBar::tab:!selected { margin-top: 2px; }
            QTabBar { alignment: left; }
        """

    def get_button_style(self, type="primary", font_size="10pt"):
        styles = {
            "primary": ("#2563EB", "#1D4ED8", "#1E40AF", "white"),
            "secondary": ("#E2E8F0", "#CBD5E1", "#94A3B8", "#1F2937"),
            "info": ("#0F766E", "#0D9488", "#115E59", "white"),
            "warning": ("#C2410C", "#EA580C", "#9A3412", "white"),
            "danger": ("#DC2626", "#EF4444", "#991B1B", "white"),
        }
        bg_color, hover_color, pressed_color, text_color = styles.get(type, styles["secondary"])
        return (
            f""" QPushButton {{ background-color: {bg_color}; color: {text_color}; border: 1px solid transparent; border-radius: 5px; padding: 8px 14px; font-weight: bold; font-size: {font_size}; outline: none; min-width: 80px; }} QPushButton:hover {{ background-color: {hover_color}; }} QPushButton:pressed {{ background-color: {pressed_color}; }} QPushButton:disabled {{ background-color: #E5E7EB; color: #94A3B8; border: 1px solid #D1D5DB; }} """)

    def get_nav_button_style(self):
        return """
            QPushButton {
                background-color: transparent;
                color: #CBD5E1;
                border: none;
                border-radius: 6px;
                padding: 9px 12px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1F2937;
                color: #FFFFFF;
            }
            QPushButton:checked {
                background-color: #2563EB;
                color: #FFFFFF;
            }
        """

    def get_progressbar_style(self):
        return (
            "QProgressBar { border: 1px solid #D8DEE6; border-radius: 5px; text-align: center; padding: 1px; background-color: #FFFFFF; color: #243447; }"
            "QProgressBar::chunk { background-color: #2563EB; border-radius: 4px; margin: 0.5px; }")

    def get_status_label_style(self):
        return "margin: 4px; padding: 9px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; color: #334155; font-size: 10pt;"

    def get_info_label_style(self):
        return "margin-bottom: 10px; color: #475569; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px; padding: 8px;"

    # --- Worker Management ---
    def is_worker_running(self):
        return ((self.pepper_worker and self.pepper_worker.isRunning()) or
                (self.generic_convert_worker and self.generic_convert_worker.isRunning()) or
                (self.scannet_convert_worker and self.scannet_convert_worker.isRunning()) or
                (self.check_shape_worker and self.check_shape_worker.isRunning()) or
                (self.merge_prediction_worker and self.merge_prediction_worker.isRunning()) or
                (self.downsample_worker and self.downsample_worker.isRunning()) # ADD THIS LINE
                )


    def get_current_worker(self):
        if self.pepper_worker and self.pepper_worker.isRunning(): return self.pepper_worker
        if self.generic_convert_worker and self.generic_convert_worker.isRunning(): return self.generic_convert_worker
        if self.scannet_convert_worker and self.scannet_convert_worker.isRunning(): return self.scannet_convert_worker
        if self.check_shape_worker and self.check_shape_worker.isRunning(): return self.check_shape_worker
        if self.merge_prediction_worker and self.merge_prediction_worker.isRunning(): return self.merge_prediction_worker
        if self.downsample_worker and self.downsample_worker.isRunning(): return self.downsample_worker # ADD THIS LINE
        return None

    def _set_global_status(self, message):
        if hasattr(self, "sidebar_status_label"):
            self.sidebar_status_label.setText(f"状态：{message}")
        if hasattr(self, "dashboard_status_label"):
            self.dashboard_status_label.setText(message)
        self.statusBar().showMessage(message, 3000)

    def set_ui_processing_state(self, is_processing):
        current_worker = self.get_current_worker()
        active_task_type = ""  # e.g. "pepper_single", "pepper_batch", "generic_single", "scannet_convert", etc.

        if is_processing and current_worker:
            if isinstance(current_worker, PepperProcessWorker):
                active_task_type = f"pepper_{'single' if current_worker.single_file else 'batch'}_{current_worker.processing_type}"
            elif isinstance(current_worker, GenericConvertWorker):
                active_task_type = f"generic_{'single' if not current_worker.is_batch else 'batch'}"
            elif isinstance(current_worker, ConvertScanNetWorker):
                active_task_type = "scannet_convert"
            elif isinstance(current_worker, CheckShapeWorker):
                active_task_type = "npy_shape_check"  # Though this is very fast
            elif isinstance(current_worker, MergePredictionWorker):
                active_task_type = "merge_prediction"
            elif isinstance(current_worker, DownsampleWorker):
                active_task_type = f"downsample_{'batch' if current_worker.is_batch else 'single'}"

        task_labels = {
            "scannet_convert": "ScanNet 格式转换中",
            "npy_shape_check": "NPY 文件检查中",
            "merge_prediction": "预测标签合并中",
            "generic_single": "单文件格式转换中",
            "generic_batch": "批量格式转换中",
            "downsample_single": "单文件下采样中",
            "downsample_batch": "批量下采样中",
        }
        for method in ["pipeline", "sor", "ror"]:
            task_labels[f"pepper_single_{method}"] = f"{method.upper()} 单文件处理中"
            task_labels[f"pepper_batch_{method}"] = f"{method.upper()} 批量处理中"
        self._set_global_status(task_labels.get(active_task_type, "任务运行中") if is_processing else "就绪")

        # Pepper Page (disable all process buttons if any worker is active, enable specific stop)
        for method in ["pipeline", "sor", "ror"]:
            getattr(self, f"{method}_single_process_button").setEnabled(not is_processing)
            getattr(self, f"{method}_batch_process_button").setEnabled(not is_processing)

            is_this_pepper_single = active_task_type == f"pepper_single_{method}"
            is_this_pepper_batch = active_task_type == f"pepper_batch_{method}"

            getattr(self, f"{method}_single_stop_button").setEnabled(is_this_pepper_single)
            getattr(self, f"{method}_batch_stop_button").setEnabled(is_this_pepper_batch)

            if hasattr(self, f"{method}_single_intermediate_checkbox"): getattr(self,
                                                                                f"{method}_single_intermediate_checkbox").setEnabled(
                not is_this_pepper_single and not is_processing)
            if hasattr(self, f"{method}_single_neighbors"): getattr(self, f"{method}_single_neighbors").setEnabled(
                not is_this_pepper_single and not is_processing); getattr(self,
                                                                          f"{method}_single_std_ratio").setEnabled(
                not is_this_pepper_single and not is_processing)
            if hasattr(self, f"{method}_single_nb_points"): getattr(self, f"{method}_single_nb_points").setEnabled(
                not is_this_pepper_single and not is_processing); getattr(self, f"{method}_single_radius").setEnabled(
                not is_this_pepper_single and not is_processing)

            if hasattr(self, f"{method}_batch_intermediate_checkbox"): getattr(self,
                                                                               f"{method}_batch_intermediate_checkbox").setEnabled(
                not is_this_pepper_batch and not is_processing)
            if hasattr(self, f"{method}_batch_neighbors"): getattr(self, f"{method}_batch_neighbors").setEnabled(
                not is_this_pepper_batch and not is_processing); getattr(self, f"{method}_batch_std_ratio").setEnabled(
                not is_this_pepper_batch and not is_processing)
            if hasattr(self, f"{method}_batch_nb_points"): getattr(self, f"{method}_batch_nb_points").setEnabled(
                not is_this_pepper_batch and not is_processing); getattr(self, f"{method}_batch_radius").setEnabled(
                not is_this_pepper_batch and not is_processing)

            # Compare button enabled only on successful completion, not during processing
            if not is_processing:
                getattr(self, f"{method}_single_compare_button").setEnabled(
                    getattr(self, f"{method}_single_compare_button").isEnabled())
            else:
                getattr(self, f"{method}_single_compare_button").setEnabled(False)

        # Generic Conversion
        is_gen_single_active = active_task_type == "generic_single"
        is_gen_batch_active = active_task_type == "generic_batch"
        self.s_conv_pcd_button.setEnabled(not is_processing);
        self.s_conv_ply_button.setEnabled(not is_processing);
        self.s_conv_xyz_button.setEnabled(not is_processing)
        self.stop_generic_single_button.setEnabled(is_gen_single_active)
        self.b_conv_pcd_button.setEnabled(not is_processing);
        self.b_conv_ply_button.setEnabled(not is_processing);
        self.b_conv_xyz_button.setEnabled(not is_processing)
        self.stop_generic_batch_button.setEnabled(is_gen_batch_active)

        # ScanNet Conversion
        is_scannet_active = active_task_type == "scannet_convert"
        self.scannet_convert_button.setEnabled(not is_processing)
        self.stop_scannet_button.setEnabled(is_scannet_active)
        self.scannet_split_ratio_edit.setEnabled(not is_scannet_active and not is_processing)
        self.scannet_seed_edit.setEnabled(not is_scannet_active and not is_processing)

        # NPY Shape Check
        is_npy_check_active = active_task_type == "npy_shape_check"
        self.npy_shape_check_button.setEnabled(not is_processing)  # No stop button for this quick task

        # Merge Prediction
        is_merge_active = active_task_type == "merge_prediction"
        self.merge_start_button.setEnabled(not is_processing)
        self.merge_stop_button.setEnabled(is_merge_active)

        # Visualize Custom (not a worker, but disable button if other workers run)
        self.vis_custom_start_button.setEnabled(
            not is_processing and not (self.visualization_thread and self.visualization_thread.is_alive()))

        is_ds_single_active = active_task_type == "downsample_single"
        is_ds_batch_active = active_task_type == "downsample_batch"
        self.downsample_single_start_button.setEnabled(not is_processing)
        self.downsample_batch_start_button.setEnabled(not is_processing)
        self.downsample_single_stop_button.setEnabled(is_ds_single_active)
        self.downsample_batch_stop_button.setEnabled(is_ds_batch_active)
        # Also disable param widgets
        self.downsample_single_method_combo.setEnabled(not is_ds_single_active and not is_processing)
        self.downsample_single_target_points_spin.setEnabled(not is_ds_single_active and not is_processing)
        self.downsample_single_voxel_size_spin.setEnabled(not is_ds_single_active and not is_processing)
        self.downsample_batch_method_combo.setEnabled(not is_ds_batch_active and not is_processing)
        self.downsample_batch_target_points_spin.setEnabled(not is_ds_batch_active and not is_processing)
        self.downsample_batch_voxel_size_spin.setEnabled(not is_ds_batch_active and not is_processing)
        if not is_processing:  # Reset progress bars and stop button texts
            for method in ["pipeline", "sor", "ror"]:
                getattr(self, f"{method}_single_progress_bar").setValue(0);
                getattr(self, f"{method}_single_stop_button").setText("停止处理")
                getattr(self, f"{method}_batch_progress_bar").setValue(0);
                getattr(self, f"{method}_batch_stop_button").setText("停止处理")
            self.generic_single_progress.setValue(0);
            self.stop_generic_single_button.setText("停止转换")
            self.generic_batch_progress.setValue(0);
            self.stop_generic_batch_button.setText("停止转换")
            self.scannet_progress_bar.setValue(0);
            self.stop_scannet_button.setText("停止转换")
            # No progress for NPY check
            self.merge_stop_button.setText("停止合并")
            # No progress for custom vis start
            self.downsample_single_progress_bar.setValue(0)
            self.downsample_batch_progress_bar.setValue(0)
            self.downsample_single_stop_button.setText("停止处理")
            self.downsample_batch_stop_button.setText("停止处理")

    def stop_current_worker(self):
        worker = self.get_current_worker()
        if worker:
            worker.stop()
            sender_button = self.sender()  # The button that triggered stop
            if sender_button and isinstance(sender_button, QPushButton):
                sender_button.setEnabled(False)
                sender_button.setText("正在停止...")
            self._set_global_status("正在停止当前任务")
            self.statusBar().showMessage("正在尝试停止当前任务...", 3000)
        else:
            self._set_global_status("就绪")
            self.statusBar().showMessage("当前没有正在运行的任务。", 3000)

    # --- Path Selection Helpers ---
    def _select_file(self, title, current_path_attr, label_widget, file_filter, reset_comparison_for_method=None):
        if self.is_worker_running(): return
        current_path = getattr(self, current_path_attr, None)
        default_dir = os.path.dirname(current_path) if current_path else ""
        file_path, _ = QFileDialog.getOpenFileName(self, title, default_dir, file_filter)
        if file_path:
            setattr(self, current_path_attr, file_path)
            if label_widget:
                label_widget.setText(f"{title.split(' ')[0]}: ...{os.sep}{os.path.basename(file_path)}")
                label_widget.setToolTip(file_path)
            if reset_comparison_for_method:  # For pepper single input
                self.processed_point_cloud_for_comparison = None
                self.original_file_path_for_comparison = None
                compare_button = getattr(self, f"{reset_comparison_for_method}_single_compare_button", None)
                if compare_button: compare_button.setEnabled(False)
        return file_path

    def _select_directory(self, title, current_path_attr, label_widget):
        if self.is_worker_running():
            return

        current_path = getattr(self, current_path_attr, None)
        dir_path = QFileDialog.getExistingDirectory(self, title, current_path or "")
        if dir_path:
            setattr(self, current_path_attr, dir_path)
            if label_widget:
                display_prefix = title.replace("选择", "").replace("文件夹", "").strip()
                if not display_prefix:
                    display_prefix = title.split(' ')[0]
                label_widget.setText(f"{display_prefix}: ...{os.sep}{os.path.basename(dir_path)}")
                label_widget.setToolTip(dir_path)
        return dir_path

    def _select_save_file(self, title, current_path_attr, label_widget, file_filter, default_suffix=""):
        if self.is_worker_running():
            return

        current_path = getattr(self, current_path_attr, None)
        default_dir = os.path.dirname(current_path) if current_path and os.path.isabs(current_path) else ""

        if current_path_attr == "merge_output_txt_path" and self.merge_orig_txt_path:
            base, _ = os.path.splitext(os.path.basename(self.merge_orig_txt_path))
            suggested_name = os.path.join(default_dir, f"{base}_merged{default_suffix}")
        elif current_path_attr == "vis_custom_screenshot_path" and self.vis_custom_input_txt_path:
            base, _ = os.path.splitext(os.path.basename(self.vis_custom_input_txt_path))
            suggested_name = os.path.join(default_dir, f"{base}_custom_view{default_suffix}")
        else:
            suggested_name = default_dir

        if not suggested_name.strip() and default_suffix:
            suggested_name = f"output{default_suffix}"
        elif not os.path.basename(suggested_name) and default_suffix:
            suggested_name = os.path.join(suggested_name, f"output{default_suffix}")

        file_path, _ = QFileDialog.getSaveFileName(self, title, suggested_name, file_filter)
        if file_path:
            if default_suffix and not file_path.lower().endswith(default_suffix.lower()):
                file_path += default_suffix

            setattr(self, current_path_attr, file_path)
            if label_widget:
                display_prefix = title.replace("选择", "").replace("路径", "").strip()
                if not display_prefix:
                    display_prefix = title.split(' ')[0]
                label_widget.setText(f"{display_prefix}: ...{os.sep}{os.path.basename(file_path)}")
                label_widget.setToolTip(file_path)
        return file_path

    # --- Pepper Processing Slots ---
    def get_pepper_path_attr(self, method_type, mode, type):  # type is 'input' or 'output'
        return f"{method_type}_{mode}_{type}_path"

    def select_pepper_single_input_file(self, method_type):
        attr_name = self.get_pepper_path_attr(method_type, 'single', 'input')
        label_widget = getattr(self, f"{method_type}_single_input_label", None)
        self._select_file(f"选择输入点云文件 ({method_type.upper()})", attr_name, label_widget,
                          "Point Cloud Files (*.pcd *.ply *.xyz *.pts);;All Files (*)",
                          reset_comparison_for_method=method_type)

    def select_pepper_single_output_dir(self, method_type):
        attr_name = self.get_pepper_path_attr(method_type, 'single', 'output')
        label_widget = getattr(self, f"{method_type}_single_output_label", None)
        self._select_directory(f"选择输出目录 ({method_type.upper()})", attr_name, label_widget)

    def select_pepper_batch_input_folder(self, method_type):
        attr_name = self.get_pepper_path_attr(method_type, 'batch', 'input')
        label_widget = getattr(self, f"{method_type}_batch_input_label", None)
        self._select_directory(f"选择输入文件夹 ({method_type.upper()})", attr_name, label_widget)

    def select_pepper_batch_output_folder(self, method_type):
        attr_name = self.get_pepper_path_attr(method_type, 'batch', 'output')
        label_widget = getattr(self, f"{method_type}_batch_output_label", None)
        self._select_directory(f"选择输出文件夹 ({method_type.upper()})", attr_name, label_widget)

    def process_single_pepper_file(self, method_type):
        input_path_attr = self.get_pepper_path_attr(method_type, 'single', 'input')
        output_path_attr = self.get_pepper_path_attr(method_type, 'single', 'output')
        input_path = getattr(self, input_path_attr, None);
        output_path = getattr(self, output_path_attr, None)
        if not input_path or not output_path: QMessageBox.warning(self, "选择文件和目录",
                                                                  "请先选择输入文件和输出目录"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "另一个处理任务正在进行中，请稍候。"); return
        params = {};
        enable_intermediate = False
        if method_type == "pipeline":
            enable_intermediate = getattr(self, f"{method_type}_single_intermediate_checkbox", QCheckBox()).isChecked()
        elif method_type == "sor":
            params['nb_neighbors'] = getattr(self, f"{method_type}_single_neighbors").value(); params[
                'std_ratio'] = getattr(self, f"{method_type}_single_std_ratio").value()
        elif method_type == "ror":
            params['nb_points'] = getattr(self, f"{method_type}_single_nb_points").value(); params['radius'] = getattr(
                self, f"{method_type}_single_radius").value()
        self.processed_point_cloud_for_comparison = None;
        self.original_file_path_for_comparison = None
        compare_button = getattr(self, f"{method_type}_single_compare_button", None)
        if compare_button: compare_button.setEnabled(False)
        status_label = getattr(self, f"{method_type}_single_status_label");
        progress_bar = getattr(self, f"{method_type}_single_progress_bar")
        self.pepper_worker = PepperProcessWorker(processing_type=method_type, params=params, single_file=input_path,
                                                 output_dir=output_path,
                                                 enable_intermediate_results=enable_intermediate)
        self.clear_worker_connections(self.pepper_worker)
        self.pepper_worker.progress_signal.connect(lambda value, pb=progress_bar: pb.setValue(value))
        self.pepper_worker.status_signal.connect(lambda msg, sl=status_label: sl.setText(msg))
        bind_task(self, "pepper_worker", lambda mt=method_type: self.on_pepper_single_finished(mt))
        self.pepper_worker.result_signal.connect(self.on_pepper_process_result)
        progress_bar.setValue(0);
        status_label.setText(f"正在初始化 ({method_type.upper()})...");
        self.set_ui_processing_state(True);
        self.pepper_worker.start()

    def process_batch_pepper_files(self, method_type):
        input_path_attr = self.get_pepper_path_attr(method_type, 'batch', 'input')
        output_path_attr = self.get_pepper_path_attr(method_type, 'batch', 'output')
        input_path = getattr(self, input_path_attr, None);
        output_path = getattr(self, output_path_attr, None)
        if not input_path or not output_path: QMessageBox.warning(self, "选择文件夹",
                                                                  "请先选择输入和输出文件夹"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "另一个处理任务正在进行中，请稍候。"); return
        params = {};
        enable_intermediate = False
        if method_type == "pipeline":
            enable_intermediate = getattr(self, f"{method_type}_batch_intermediate_checkbox", QCheckBox()).isChecked()
        elif method_type == "sor":
            params['nb_neighbors'] = getattr(self, f"{method_type}_batch_neighbors").value(); params[
                'std_ratio'] = getattr(self, f"{method_type}_batch_std_ratio").value()
        elif method_type == "ror":
            params['nb_points'] = getattr(self, f"{method_type}_batch_nb_points").value(); params['radius'] = getattr(
                self, f"{method_type}_batch_radius").value()
        status_label = getattr(self, f"{method_type}_batch_status_label");
        progress_bar = getattr(self, f"{method_type}_batch_progress_bar")
        self.pepper_worker = PepperProcessWorker(processing_type=method_type, params=params, input_dir=input_path,
                                                 output_dir=output_path,
                                                 enable_intermediate_results=enable_intermediate)
        self.clear_worker_connections(self.pepper_worker)
        self.pepper_worker.progress_signal.connect(lambda value, pb=progress_bar: pb.setValue(value))
        self.pepper_worker.status_signal.connect(lambda msg, sl=status_label: sl.setText(msg))
        bind_task(self, "pepper_worker", lambda mt=method_type: self.on_pepper_batch_finished(mt))
        progress_bar.setValue(0);
        status_label.setText(f"正在初始化批量处理 ({method_type.upper()})...");
        self.set_ui_processing_state(True);
        self.pepper_worker.start()

    def clear_worker_connections(self, worker):
        if not worker: return
        signals_to_try = ["progress_signal", "status_signal", "finished_signal", "result_signal"]
        for sig_name in signals_to_try:
            if hasattr(worker, sig_name):
                signal_instance = getattr(worker, sig_name)
                try:
                    signal_instance.disconnect()
                except TypeError:
                    pass  # No connections
                except Exception as e:
                    print(f"Error disconnecting signal {sig_name}: {e}")

    def on_pepper_single_finished(self, method_type):
        self.set_ui_processing_state(False)
        status_label = getattr(self, f"{method_type}_single_status_label", None)
        compare_button = getattr(self, f"{method_type}_single_compare_button", None)
        final_status = status_label.text() if status_label else "状态未知"
        if "处理完成" in final_status and "出错" not in final_status and "中止" not in final_status:
            if self.processed_point_cloud_for_comparison is not None and self.processed_point_cloud_for_comparison.has_points():
                if status_label: status_label.setText(f"{method_type.upper()} 处理完成！可进行对比。")
                if compare_button: compare_button.setEnabled(True)
                QMessageBox.information(self, "处理完成", f"{method_type.upper()} 点云处理已完成。")
            else:
                if status_label: status_label.setText(f"{method_type.upper()} 处理完成，但未生成有效结果用于对比。")
                QMessageBox.warning(self, "处理完成", f"{method_type.upper()} 点云处理已完成，但未生成有效结果用于对比。")
        elif "中止" in final_status:
            QMessageBox.warning(self, "处理中止", f"{method_type.upper()} 点云处理已中止。")
        elif "出错" in final_status or "错误" in final_status:
            QMessageBox.critical(self, "处理出错", f"{method_type.upper()} 点云处理过程中发生错误:\n{final_status}")
        self.pepper_worker = None

    def on_pepper_batch_finished(self, method_type):
        self.set_ui_processing_state(False)
        status_label = getattr(self, f"{method_type}_batch_status_label", None)
        final_status = status_label.text() if status_label else "状态未知"
        if "完成" in final_status and "出错" not in final_status and "中止" not in final_status:
            QMessageBox.information(self, "处理完成", f"{method_type.upper()} 点云批量处理已完成。\n({final_status})")
        elif "中止" in final_status:
            QMessageBox.warning(self, "处理中止", f"{method_type.upper()} 点云批量处理已中止。")
        elif "出错" in final_status or "错误" in final_status:
            QMessageBox.critical(self, "处理出错",
                                 f"{method_type.upper()} 点云批量处理过程中发生错误或警告:\n{final_status}")
        self.pepper_worker = None

    def on_pepper_process_result(self, processed_pcd_obj, original_file_path):
        if processed_pcd_obj is None or not isinstance(processed_pcd_obj,
                                                       o3d.geometry.PointCloud) or not processed_pcd_obj.has_points():
            self.processed_point_cloud_for_comparison = None;
            self.original_file_path_for_comparison = None;
            self.current_compare_method_name = "";
            print("Warning: Received empty or invalid processed point cloud result.");
            return
        try:
            self.processed_point_cloud_for_comparison = prepare_point_cloud_for_visualization(processed_pcd_obj)
            self.original_file_path_for_comparison = original_file_path
            worker = self.get_current_worker()  # Might be None if worker finished very fast
            self.current_compare_method_name = worker.processing_type.upper() if worker and isinstance(worker,
                                                                                                       PepperProcessWorker) else "处理后"
            print(
                f"Stored processed result ({self.current_compare_method_name}) for comparison: {len(self.processed_point_cloud_for_comparison.points)} points")
        except Exception as e:
            error_msg = f"保存处理结果供对比时出错: {str(e)}";
            worker = self.get_current_worker()
            status_label_widget = None
            if worker and isinstance(worker, PepperProcessWorker) and worker.single_file: status_label_widget = getattr(
                self, f"{worker.processing_type}_single_status_label", None)
            if status_label_widget: QTimer.singleShot(0, lambda sl=status_label_widget, msg=error_msg: sl.setText(msg))
            print(error_msg);
            traceback.print_exc();
            self.processed_point_cloud_for_comparison = None;
            self.original_file_path_for_comparison = None;
            self.current_compare_method_name = ""

    def visualize_before_after_comparison(self, method_type):
        status_label = getattr(self, f"{method_type}_single_status_label", None)
        update_status_func = (
            lambda msg, sl=status_label: QTimer.singleShot(0, lambda: sl.setText(msg))) if status_label else None
        if self.processed_point_cloud_for_comparison is None or not self.original_file_path_for_comparison: QMessageBox.warning(
            self, "数据不足", "请先成功处理一个点云文件。"); return
        if not self.processed_point_cloud_for_comparison.has_points(): QMessageBox.warning(self, "数据无效",
                                                                                           "处理后的点云为空。"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "请等待当前任务完成后再进行可视化。"); return
        if self.visualization_thread and self.visualization_thread.is_alive(): QMessageBox.warning(self, "可视化进行中",
                                                                                                   "另一个可视化窗口似乎仍在运行。"); return
        try:
            if update_status_func: update_status_func("准备对比: 加载原始点云...")
            QApplication.processEvents()
            original_pcd = read_point_cloud(self.original_file_path_for_comparison)
            if not original_pcd.has_points():
                if self.original_file_path_for_comparison.lower().endswith('.xyz'):
                    data = np.loadtxt(self.original_file_path_for_comparison)
                    if data.shape[1] < 3: raise ValueError("< 3 columns")
                    original_pcd = o3d.geometry.PointCloud();
                    original_pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                    if data.shape[1] >= 6: colors = data[:, 3:6]; original_pcd.colors = o3d.utility.Vector3dVector(
                        colors / 255.0 if np.max(colors) > 1 else colors)
                    if not original_pcd.has_points(): raise ValueError("Empty after numpy")
                else:
                    raise ValueError("无法加载原始点云或文件为空")
            original_pcd_copy = prepare_point_cloud_for_visualization(original_pcd, [1.0, 0.7, 0.0])
            processed_pcd_copy = prepare_point_cloud_for_visualization(self.processed_point_cloud_for_comparison,
                                                                       [0.0, 0.7, 1.0])
            if not original_pcd_copy.has_points() or not processed_pcd_copy.has_points(): raise ValueError(
                "准备可视化时点云变空。")
            if update_status_func: update_status_func("正在启动对比可视化线程...")
            QApplication.processEvents()
            self.visualization_thread = threading.Thread(target=_run_sequential_visualization, args=(
            original_pcd_copy, processed_pcd_copy, self.original_file_path_for_comparison, update_status_func))
            self.visualization_thread.daemon = True;
            self.visualization_thread.start()
            if update_status_func: QTimer.singleShot(50,
                                                     lambda: update_status_func("对比可视化已启动 (请查看Open3D窗口)"))
        except Exception as e:
            error_msg = f"启动对比可视化出错: {str(e)}"; (
                update_status_func(error_msg) if update_status_func else None); QMessageBox.critical(self, "可视化错误",
                                                                                                     error_msg); traceback.print_exc()

    # --- Generic Conversion Slots ---
    def select_generic_single_input_file(self):
        self._select_file("选择输入点云文件", "generic_single_input_path", self.generic_single_input_label,
                          "Point Cloud Files (*.pcd *.ply *.xyz *.pts);;All Files (*)")

    def select_generic_single_output_dir(self):
        self._select_directory("选择输出目录", "generic_single_output_dir", self.generic_single_output_label)

    def start_generic_single_conversion(self, target_format):
        if not self.generic_single_input_path or not self.generic_single_output_dir: QMessageBox.warning(self,
                                                                                                         "选择文件和目录",
                                                                                                         "请先选择输入文件和输出目录"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "其他任务进行中，请稍候。"); return
        self.generic_convert_worker = GenericConvertWorker(self.generic_single_input_path,
                                                           self.generic_single_output_dir, target_format, False)
        self.clear_worker_connections(self.generic_convert_worker)
        self.generic_convert_worker.progress_signal.connect(self.generic_single_progress.setValue)
        self.generic_convert_worker.status_signal.connect(self.generic_single_status.setText)
        bind_task(self, "generic_convert_worker", self.on_generic_single_finished)
        self.generic_single_progress.setValue(0);
        self.generic_single_status.setText(f"准备转换为 {target_format.upper()}...");
        self.set_ui_processing_state(True);
        self.generic_convert_worker.start()

    def on_generic_single_finished(self):
        self.set_ui_processing_state(False);
        final_status = self.generic_single_status.text()
        if "完成" in final_status and "出错" not in final_status and "中止" not in final_status and "跳过" not in final_status:
            QMessageBox.information(self, "转换完成", "单文件格式转换已完成。")
        elif "跳过" in final_status:
            QMessageBox.information(self, "操作跳过", "文件已是目标格式或无法加载，已跳过。")
        elif "中止" in final_status:
            QMessageBox.warning(self, "转换中止", "单文件格式转换已中止。")
        elif "出错" in final_status or "错误" in final_status:
            QMessageBox.critical(self, "转换出错", f"单文件转换出错:\n{final_status}")
        self.generic_convert_worker = None

    def select_generic_batch_input_dir(self):
        self._select_directory("选择输入文件夹", "generic_batch_input_dir", self.generic_batch_input_label)

    def select_generic_batch_output_dir(self):
        self._select_directory("选择输出文件夹", "generic_batch_output_dir", self.generic_batch_output_label)

    def start_generic_batch_conversion(self, target_format):
        if not self.generic_batch_input_dir or not self.generic_batch_output_dir: QMessageBox.warning(self,
                                                                                                      "选择文件夹",
                                                                                                      "请先选择输入和输出文件夹"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "其他任务进行中，请稍候。"); return
        self.generic_convert_worker = GenericConvertWorker(self.generic_batch_input_dir, self.generic_batch_output_dir,
                                                           target_format, True)
        self.clear_worker_connections(self.generic_convert_worker)
        self.generic_convert_worker.progress_signal.connect(self.generic_batch_progress.setValue)
        self.generic_convert_worker.status_signal.connect(self.generic_batch_status.setText)
        bind_task(self, "generic_convert_worker", self.on_generic_batch_finished)
        self.generic_batch_progress.setValue(0);
        self.generic_batch_status.setText(f"准备批量转换为 {target_format.upper()}...");
        self.set_ui_processing_state(True);
        self.generic_convert_worker.start()

    def on_generic_batch_finished(self):
        self.set_ui_processing_state(False);
        final_status = self.generic_batch_status.text()
        if "批量转换完成" in final_status and "出错" not in final_status and "中止" not in final_status:
            QMessageBox.information(self, "转换完成", f"批量格式转换已完成。\n({final_status})")
        elif "未找到需要转换" in final_status:
            QMessageBox.information(self, "无需转换", final_status)
        elif "中止" in final_status:
            QMessageBox.warning(self, "转换中止", "批量格式转换已中止。")
        elif "出错" in final_status or "错误" in final_status:
            QMessageBox.warning(self, "转换有误", f"批量转换出错或有警告:\n{final_status}")
        self.generic_convert_worker = None

    # --- ScanNet Conversion Slots (New) ---
    def select_scannet_input_folder(self):
        self._select_directory("选择TXT输入文件夹 (ScanNet)", "scannet_input_dir", self.scannet_input_label)

    def select_scannet_output_folder(self):
        self._select_directory("选择ScanNet输出文件夹", "scannet_output_dir", self.scannet_output_label)

    def start_scannet_conversion(self):
        if not self.scannet_input_dir or not self.scannet_output_dir: QMessageBox.warning(self, "选择文件夹",
                                                                                          "请先选择ScanNet转换的输入和输出文件夹"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "其他任务进行中，请稍候。"); return
        split_ratio_str = self.scannet_split_ratio_edit.text()
        seed_str = self.scannet_seed_edit.text()
        try:  # Basic validation before starting worker
            ratios = [float(r.strip()) for r in split_ratio_str.split(',')]
            if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-5): raise ValueError(
                "分割比例必须是3个数字，总和为1.0")
            int(seed_str)  # Check if seed is integer
        except ValueError as e:
            QMessageBox.warning(self, "参数错误", f"ScanNet参数无效: {e}"); return

        self.scannet_convert_worker = ConvertScanNetWorker(self.scannet_input_dir, self.scannet_output_dir,
                                                           split_ratio_str, seed_str)
        self.clear_worker_connections(self.scannet_convert_worker)
        self.scannet_convert_worker.progress_signal.connect(self.scannet_progress_bar.setValue)
        self.scannet_convert_worker.status_signal.connect(self.scannet_status_label.setText)
        bind_task(self, "scannet_convert_worker", self.on_scannet_conversion_finished)
        self.scannet_progress_bar.setValue(0);
        self.scannet_status_label.setText("正在初始化 ScanNet 转换...");
        self.set_ui_processing_state(True);
        self.scannet_convert_worker.start()

    def on_scannet_conversion_finished(self):
        self.set_ui_processing_state(False);
        final_status = self.scannet_status_label.text()
        if "完成" in final_status and "出错" not in final_status and "中止" not in final_status:
            QMessageBox.information(self, "转换完成", f"ScanNet格式转换已完成。\n({final_status})")
        elif "未找到 .txt 文件" in final_status:
            QMessageBox.warning(self, "未找到文件", final_status)
        elif "中止" in final_status:
            QMessageBox.warning(self, "转换中止", "ScanNet格式转换已中止。")
        elif "出错" in final_status or "错误" in final_status:
            QMessageBox.critical(self, "转换出错", f"ScanNet格式转换出错:\n{final_status}")
        self.scannet_convert_worker = None

    # --- NPY Shape Check Slots (New) ---
    def select_npy_shape_input_file(self):
        self._select_file("选择 .npy 文件", "npy_shape_input_path", self.npy_shape_input_label,
                          "Numpy Array Files (*.npy);;All Files (*)")

    def start_npy_shape_check(self):
        if not self.npy_shape_input_path: QMessageBox.warning(self, "选择文件", "请先选择一个.npy文件"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理",
                                                         "其他任务进行中，请稍候。"); return  # Should be very quick, but for consistency
        self.check_shape_worker = CheckShapeWorker(self.npy_shape_input_path)
        self.clear_worker_connections(self.check_shape_worker)
        self.check_shape_worker.result_signal.connect(self.npy_shape_result_textedit.setText)
        self.check_shape_worker.status_signal.connect(self.npy_shape_status_label.setText)
        bind_task(self, "check_shape_worker", self.on_npy_shape_check_finished)
        self.npy_shape_result_textedit.clear();
        self.npy_shape_status_label.setText("正在检查...");
        self.set_ui_processing_state(True);
        self.check_shape_worker.start()

    def on_npy_shape_check_finished(self):
        self.set_ui_processing_state(False);
        self.check_shape_worker = None
        # Status label already updated by worker's status_signal

    # --- Prediction Analysis Slots (New) ---
    # Merge
    def select_merge_original_txt(self):
        self._select_file("选择原始点云TXT", "merge_orig_txt_path", self.merge_orig_label,
                          "Text Files (*.txt);;All Files (*)")

    def select_merge_prediction_txt(self):
        self._select_file("选择预测标签TXT", "merge_pred_txt_path", self.merge_pred_label,
                          "Text Files (*.txt);;All Files (*)")

    def select_merge_output_txt(self):
        path = self._select_save_file(
            "选择合并后保存路径",
            "merge_output_txt_path",
            self.merge_output_label,
            "Text Files (*.txt);;All Files (*)",
            default_suffix=".txt"
        )
        # Only successful merge output is forwarded to visualization.

    def start_merge_predictions(self):
        if not self.merge_orig_txt_path or not self.merge_pred_txt_path or not self.merge_output_txt_path: QMessageBox.warning(
            self, "选择文件", "请选择原始点云、预测标签和输出文件路径。"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "其他任务进行中，请稍候。"); return
        self.merge_prediction_worker = MergePredictionWorker(self.merge_orig_txt_path, self.merge_pred_txt_path,
                                                             self.merge_output_txt_path)
        self.clear_worker_connections(self.merge_prediction_worker)
        self.merge_prediction_worker.status_signal.connect(self.merge_status_label.setText)
        self.merge_prediction_worker.result_signal.connect(self.on_merge_prediction_result)  # To update vis input path
        bind_task(self, "merge_prediction_worker", self.on_merge_prediction_finished)
        self.merge_status_label.setText("开始合并...");
        self.set_ui_processing_state(True);
        self.merge_prediction_worker.start()

    def on_merge_prediction_result(self, merged_file_path):
        self.vis_custom_input_txt_path = merged_file_path  # Store for visualization tab
        self.vis_custom_input_label.setText(f"待可视化TXT: ...{os.sep}{os.path.basename(merged_file_path)}")
        self.vis_custom_input_label.setToolTip(merged_file_path)
        self.vis_format_widget.set_path(merged_file_path, merged=True)
        self.merge_status_label.setText(f"合并完成! 输出: {os.path.basename(merged_file_path)}")

    def on_merge_prediction_finished(self):
        self.set_ui_processing_state(False);
        final_status = self.merge_status_label.text()
        if "完成" in final_status and "出错" not in final_status and "中止" not in final_status:
            QMessageBox.information(self, "合并完成", f"预测标签合并已完成。\n({final_status})")
        elif "中止" in final_status:
            QMessageBox.warning(self, "合并中止", "预测标签合并已中止。")
        elif "出错" in final_status or "错误" in final_status:
            QMessageBox.critical(self, "合并出错", f"预测标签合并出错:\n{final_status}")
        self.merge_prediction_worker = None

    # Visualize Custom
    def select_vis_custom_input_txt(self):
        path = self._select_file("选择待可视化文件", "vis_custom_input_txt_path", self.vis_custom_input_label,
                                "点云文本 (*.txt *.xyz *.csv *.pts);;所有文件 (*)")
        if path:
            self.vis_format_widget.set_path(path)

    def clear_vis_custom_screenshot(self):
        self.vis_custom_screenshot_path = None
        self.vis_custom_screenshot_label.setText("截图保存路径: 未选择 (可选)")
        self.vis_custom_screenshot_label.setToolTip("")

    def on_custom_visualization_finished(self):
        self.vis_custom_start_button.setEnabled(not self.is_worker_running())
        self.vis_custom_start_button.setText("开始可视化")

    def select_vis_custom_screenshot_path(self):
        self._select_save_file("选择截图保存路径", "vis_custom_screenshot_path", self.vis_custom_screenshot_label,
                               "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)", default_suffix=".png")

    def start_custom_color_visualization(self):
        if not self.vis_custom_input_txt_path: QMessageBox.warning(self, "选择文件",
                                                                   "请选择要可视化的点云文本文件。"); return
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "其他任务进行中，请稍候。"); return
        if self.visualization_thread and self.visualization_thread.is_alive(): QMessageBox.warning(self, "可视化进行中",
                                                                                                   "另一个可视化窗口似乎仍在运行。"); return

        color_map_str = self.vis_custom_color_map_edit.toPlainText()
        screenshot_file = self.vis_custom_screenshot_path if self.vis_custom_screenshot_path else None  # Pass None if not set

        self.vis_custom_status_label.setText("准备自定义颜色可视化...")
        self.vis_custom_start_button.setEnabled(False)
        self.vis_custom_start_button.setText("可视化进行中…")

        self.visualization_thread = threading.Thread(target=_run_custom_color_visualization,
                                                     args=(
                                                     self.vis_custom_input_txt_path, color_map_str, screenshot_file,
                                                     self.custom_visualization_signals.status.emit),
                                                     kwargs={"options": self.vis_format_widget.options(),
                                                             "finished_func": self.custom_visualization_signals.finished.emit})
        self.visualization_thread.daemon = True
        self.visualization_thread.start()

    def select_downsample_single_input_file(self):
        self._select_file("选择输入TXT文件", "downsample_single_input_path", self.downsample_single_input_label,
                          "Text Files (*.txt)")

    def select_downsample_single_output_file(self):
        self._select_save_file("选择降采样后保存路径", "downsample_single_output_path",
                               self.downsample_single_output_label, "Text Files (*.txt)", default_suffix=".txt")

    def select_downsample_batch_input_dir(self):
        self._select_directory("选择输入文件夹", "downsample_batch_input_dir", self.downsample_batch_input_label)

    def select_downsample_batch_output_dir(self):
        self._select_directory("选择输出文件夹", "downsample_batch_output_dir", self.downsample_batch_output_label)

    def start_downsample(self, is_batch):
        if self.is_worker_running():
            QMessageBox.warning(self, "操作受限", "请等待当前任务完成后再开始新任务。")
            return

        mode = "batch" if is_batch else "single"

        # Get paths
        input_path = getattr(self, f"downsample_{mode}_input_path" if not is_batch else f"downsample_{mode}_input_dir")
        output_path = getattr(self,
                              f"downsample_{mode}_output_path" if not is_batch else f"downsample_{mode}_output_dir")

        if not input_path or not output_path:
            QMessageBox.warning(self, "路径缺失", "请选择所有必要的输入和输出路径。")
            return

        # # Get method and parameters
        # method_combo = getattr(self, f"downsample_{mode}_method_combo")
        # method_text = method_combo.currentText().split(" ")[0].lower()  # e.g., "random" from "随机采样 (Random)"

        # Get method and parameters
        method_combo = getattr(self, f"downsample_{mode}_method_combo")
        # Correctly extract the English keyword from "中文 (English)" format
        full_text = method_combo.currentText()
        try:
            method_text = full_text.split('(')[1].split(')')[0].strip().lower()
        except IndexError:
            # Fallback in case the format ever changes, though it shouldn't with current setup
            method_text = full_text.lower()

        params = {}
        if method_text in ['random', 'uniform', 'farthest']:
            params['target_points'] = getattr(self, f"downsample_{mode}_target_points_spin").value()
        elif method_text == 'voxel':
            params['voxel_size'] = getattr(self, f"downsample_{mode}_voxel_size_spin").value()

        # Get UI elements
        status_label = getattr(self, f"downsample_{mode}_status_label")
        progress_bar = getattr(self, f"downsample_{mode}_progress_bar")

        # Create and start worker
        self.downsample_worker = DownsampleWorker(input_path, output_path, method_text, params, is_batch)
        self.clear_worker_connections(self.downsample_worker)
        self.downsample_worker.status_signal.connect(status_label.setText)
        self.downsample_worker.progress_signal.connect(progress_bar.setValue)
        bind_task(self, "downsample_worker", self.on_downsample_finished)

        progress_bar.setValue(0)
        status_label.setText(f"正在初始化下采样 ({method_text})...")
        self.set_ui_processing_state(True)
        self.downsample_worker.start()

    def on_downsample_finished(self):
        is_batch = self.downsample_worker.is_batch if self.downsample_worker else False
        mode = "batch" if is_batch else "single"
        status_label = getattr(self, f"downsample_{mode}_status_label")
        final_status = status_label.text()

        self.set_ui_processing_state(False)
        self.downsample_worker = None

        if "完成" in final_status and "错误" not in final_status and "中止" not in final_status:
            QMessageBox.information(self, "处理完成", f"点云下采样任务已完成。\n({final_status})")
        elif "中止" in final_status:
            QMessageBox.warning(self, "处理中止", "点云下采样任务已中止。")
        elif "错误" in final_status:
            QMessageBox.critical(self, "处理出错", f"点云下采样任务出错:\n{final_status}")

    # --- General Utility Methods ---
    def load_and_visualize_point_cloud(self):
        if self.is_worker_running(): QMessageBox.warning(self, "正在处理", "请等待后台任务完成后再加载。"); return
        if self.visualization_thread and self.visualization_thread.is_alive(): QMessageBox.warning(self, "可视化进行中",
                                                                                                   "另一个可视化窗口似乎仍在运行。"); return
        start_dir = os.path.dirname(self.file_label.toolTip()) if self.file_label.toolTip() else ""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择点云文件", start_dir,
                                                   "Point Cloud Files (*.pcd *.ply *.xyz *.pts);;All Files (*)")
        if not file_path: return
        QApplication.setOverrideCursor(Qt.WaitCursor);
        self.statusBar().showMessage(f"正在加载: {os.path.basename(file_path)}...");
        QApplication.processEvents()
        try:
            pcd = read_point_cloud(file_path)
            if not pcd.has_points():
                if file_path.lower().endswith('.xyz'):
                    data = np.loadtxt(file_path);
                    pcd = o3d.geometry.PointCloud()
                    if data.ndim == 1 and data.shape[0] >= 3: data = data.reshape(1, -1)  # Single point
                    if data.shape[1] < 3: raise ValueError("< 3 columns")
                    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                    if data.shape[1] >= 6:
                        colors_raw = data[:, 3:6]
                        if np.max(colors_raw) > 1.0 and np.all(colors_raw >= 0) and np.all(colors_raw <= 255):
                            colors_norm = colors_raw / 255.0
                        else:
                            colors_norm = np.clip(colors_raw, 0.0, 1.0)
                        pcd.colors = o3d.utility.Vector3dVector(colors_norm)
                    if not pcd.has_points(): raise ValueError("加载XYZ后为空。")
                else:
                    raise ValueError("加载点云为空或格式无法识别。")
            self.point_cloud = pcd
            self.file_label.setText(f"当前: ...{os.sep}{os.path.basename(file_path)}");
            self.file_label.setToolTip(file_path)
            info_text = f"点数: {len(self.point_cloud.points):,}"
            if self.point_cloud.has_colors(): info_text += " | 有颜色"
            if self.point_cloud.has_normals(): info_text += " | 有法线"
            self.points_info.setText(info_text)
            self.statusBar().showMessage(f"加载成功: {os.path.basename(file_path)}. 启动可视化...", 5000);
            QApplication.processEvents()
            self.visualize_point_cloud(self.point_cloud)
        except Exception as e:
            self.point_cloud = None;
            self.file_label.setText("加载出错");
            self.file_label.setToolTip("")
            self.points_info.setText("点云信息：无");
            error_msg = f"加载点云出错: {str(e)}"
            self.statusBar().showMessage(error_msg, 5000);
            QMessageBox.critical(self, "加载错误", error_msg);
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def visualize_point_cloud(self, pcd_to_view):
        if pcd_to_view is None or not isinstance(pcd_to_view, o3d.geometry.PointCloud) or not pcd_to_view.has_points():
            self.statusBar().showMessage("无法可视化空点云", 3000);
            return
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.statusBar().showMessage("可视化窗口已打开", 3000);
            QMessageBox.information(self, "提示", "可视化窗口已运行。");
            return
        pcd_copy = prepare_point_cloud_for_visualization(pcd_to_view)  # Ensures colors are 0-1
        if not pcd_copy.has_points(): self.statusBar().showMessage("准备可视化时点云为空", 3000); return
        self.statusBar().showMessage("正在启动可视化窗口...", 2000)
        window_name = f"查看: {os.path.basename(self.file_label.toolTip() or '点云')}"
        self.visualization_thread = threading.Thread(target=_run_visualization, args=([pcd_copy],),
                                                     kwargs={"window_name": window_name})
        self.visualization_thread.daemon = True;
        self.visualization_thread.start()

    def closeEvent(self, event):
        # Active tasks are stopped asynchronously by TaskBinding.
        event.accept()


if __name__ == '__main__':  # This part is usually in main.py, but for completeness if run directly
    app = QApplication(sys.argv)
    if 'Fusion' in QApplication.styleKeys():
        try:
            app.setStyle('Fusion'); print("Applied Fusion style.")
        except Exception as e:
            print(f"Could not apply Fusion style: {e}")
    ex = PointCloudProcessor()
    ex.show()
    sys.exit(app.exec_())
