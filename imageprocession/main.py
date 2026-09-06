# main.py
import importlib
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDesktopWidget,
    QMainWindow,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)


def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


TOOL_WINDOWS = {
    "cutter_window": ("image_cutter", "ImageCutterWindow"),
    "tif_rgb_converter_window": ("tif_rgb_converter", "TifRgbConverterWindow"),
    "visual_window": ("visualizer", "VisualizationWindow"),
    "voc_creator_window": ("voc_creator", "VOCDatasetCreatorWindow"),
    "tif_processor_window": ("tif_processor", "TifProcessorWindow"),
}


def load_window_class(attr_name: str):
    module_name, class_name = TOOL_WINDOWS[attr_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像文件处理与数据集工具")
        self.setGeometry(0, 0, 480, 430)
        self._center_window()

        try:
            self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        except Exception as exc:
            print(f"Could not set window icon: {exc}")

        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignCenter)

        self._add_tool_button(
            layout,
            " 图像切片工具",
            QStyle.SP_FileDialogListView,
            "打开图像切片工具，用于将大幅遥感影像及其标签切成小块。",
            self.open_cutter_window,
        )
        self._add_tool_button(
            layout,
            " TIF 转 RGB 工具",
            QStyle.SP_FileDialogDetailedView,
            "打开 TIF 转 RGB 工具，用于将单个或批量 TIF 转换为 RGB 图片。",
            self.open_tif_rgb_converter_window,
        )
        self._add_tool_button(
            layout,
            " 图像可视化工具",
            QStyle.SP_DesktopIcon,
            "打开图像可视化工具，用于查看图像和对应的语义分割标签。",
            self.open_visual_window,
        )
        self._add_tool_button(
            layout,
            " VOC 数据集制作工具",
            QStyle.SP_DriveHDIcon,
            "打开 VOC 格式数据集制作工具，用于整理图像和标签并划分训练/验证集。",
            self.open_voc_creator_window,
        )
        self._add_tool_button(
            layout,
            " TIF 矢量栅格化工具",
            QStyle.SP_CommandLink,
            "打开 TIF 处理工具，用于将 GPKG 矢量数据栅格化到 TIF 图像上。",
            self.open_tif_processor_window,
        )

        self.setCentralWidget(main_widget)

        self.cutter_window = None
        self.tif_rgb_converter_window = None
        self.visual_window = None
        self.voc_creator_window = None
        self.tif_processor_window = None

    def _add_tool_button(self, layout, text, icon_id, tooltip, callback):
        button = QPushButton(text)
        button.setIcon(self.style().standardIcon(icon_id))
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        button.setMinimumWidth(280)
        layout.addWidget(button)

    def _center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _show_or_raise(self, attr_name: str):
        window = getattr(self, attr_name)
        if window is None or not window.isVisible():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                window_class = load_window_class(attr_name)
                window = window_class()
                setattr(self, attr_name, window)
                window.show()
            except Exception as exc:
                traceback.print_exc()
                QMessageBox.critical(self, "宸ュ叿鍚姩澶辫触", f"鏃犳硶鎵撳紑宸ュ叿:\n{exc}")
            finally:
                QApplication.restoreOverrideCursor()
        else:
            window.activateWindow()
            window.raise_()

    def open_cutter_window(self):
        self._show_or_raise("cutter_window")

    def open_tif_rgb_converter_window(self):
        self._show_or_raise("tif_rgb_converter_window")

    def open_visual_window(self):
        self._show_or_raise("visual_window")

    def open_voc_creator_window(self):
        self._show_or_raise("voc_creator_window")

    def open_tif_processor_window(self):
        self._show_or_raise("tif_processor_window")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        with open(resource_path("style.qss"), "r", encoding="utf-8") as handle:
            app.setStyleSheet(handle.read())
    except FileNotFoundError:
        print("style.qss not found. Using default styles.")
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
