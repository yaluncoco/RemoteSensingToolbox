import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QFileDialog, QTextEdit, QLabel,
    QMessageBox, QGroupBox, QFormLayout, QDesktopWidget, QStyle
)
from PIL import UnidentifiedImageError
from PyQt5.QtCore import Qt
from PIL import Image
from PIL.Image import Resampling

import ast # For literal_eval
from typing import Optional

rasterio = None
RioResampling = None
RASTERIO_AVAILABLE = False
RASTERIO_IMPORT_ERROR: Optional[Exception] = None
_RASTERIO_CHECKED = False


Image.MAX_IMAGE_PIXELS = None
MAX_PREVIEW_PIXELS = 2048
TIF_EXTENSIONS = (".tif", ".tiff")
_PYPLOT = None


def get_pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        import matplotlib.pyplot as pyplot

        _PYPLOT = pyplot
    return _PYPLOT


def ensure_rasterio_available() -> bool:
    global rasterio, RioResampling, RASTERIO_AVAILABLE, RASTERIO_IMPORT_ERROR, _RASTERIO_CHECKED
    if _RASTERIO_CHECKED:
        return RASTERIO_AVAILABLE

    try:
        import rasterio as rasterio_module
        from rasterio.enums import Resampling as rio_resampling

        rasterio = rasterio_module
        RioResampling = rio_resampling
        RASTERIO_AVAILABLE = True
        RASTERIO_IMPORT_ERROR = None
    except Exception as exc:
        rasterio = None
        RioResampling = None
        RASTERIO_AVAILABLE = False
        RASTERIO_IMPORT_ERROR = exc
    _RASTERIO_CHECKED = True
    return RASTERIO_AVAILABLE


def is_tif_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in TIF_EXTENSIONS


def preview_shape(width: int, height: int, max_size: int = MAX_PREVIEW_PIXELS) -> tuple[int, int]:
    scale = min(max_size / max(width, height), 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def stretch_to_uint8(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.dtype == np.uint8:
        return data

    data = data.astype(np.float32, copy=False)
    output = np.zeros(data.shape, dtype=np.uint8)
    for channel_index in range(data.shape[2]):
        band = data[:, :, channel_index]
        valid = np.isfinite(band) & (band > 0)
        valid_values = band[valid]
        if valid_values.size == 0:
            continue
        p2, p98 = np.percentile(valid_values, (2, 98))
        if p98 <= p2:
            continue
        stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
        stretched[~np.isfinite(stretched)] = 0
        output[:, :, channel_index] = stretched.astype(np.uint8)
    return output


def read_tif_rgb_preview(path: str, max_size: int = MAX_PREVIEW_PIXELS) -> np.ndarray:
    if not ensure_rasterio_available():
        raise RuntimeError(f"TIF 大图预览需要 rasterio：{RASTERIO_IMPORT_ERROR}")

    with rasterio.open(path) as src:
        out_width, out_height = preview_shape(src.width, src.height, max_size)
        if src.count == 1:
            indexes = [1]
        else:
            indexes = [1, 2, 3]

        data = src.read(
            indexes=indexes,
            out_shape=(len(indexes), out_height, out_width),
            masked=True,
            resampling=RioResampling.bilinear,
        )

    data = np.asarray(np.ma.filled(data, 0))
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    rgb = np.transpose(data[:3], (1, 2, 0))
    return stretch_to_uint8(rgb)


def read_tif_label_preview(path: str, max_size: int = MAX_PREVIEW_PIXELS) -> np.ndarray:
    if not ensure_rasterio_available():
        raise RuntimeError(f"TIF 标签预览需要 rasterio：{RASTERIO_IMPORT_ERROR}")

    with rasterio.open(path) as src:
        out_width, out_height = preview_shape(src.width, src.height, max_size)
        label = src.read(
            1,
            out_shape=(out_height, out_width),
            masked=True,
            resampling=RioResampling.nearest,
        )
    return np.asarray(np.ma.filled(label, 0))


def read_image_preview(path: str) -> np.ndarray:
    if is_tif_path(path):
        return read_tif_rgb_preview(path)

    with Image.open(path) as image_pil:
        image_pil.thumbnail((MAX_PREVIEW_PIXELS, MAX_PREVIEW_PIXELS), Resampling.BILINEAR)
        return np.array(image_pil.convert("RGB"))


def read_label_preview(path: str) -> np.ndarray:
    if is_tif_path(path):
        return read_tif_label_preview(path)

    with Image.open(path) as label_pil:
        label_pil.thumbnail((MAX_PREVIEW_PIXELS, MAX_PREVIEW_PIXELS), Resampling.NEAREST)
        label = np.array(label_pil)
    if label.ndim == 3:
        return label[:, :, 0]
    return label


def build_color_label_image(label: np.ndarray, colors: dict) -> np.ndarray:
    valid_colors: dict[int, list[int]] = {}
    invalid_items: list[tuple[object, object]] = []
    for val, color in colors.items():
        if (
            isinstance(val, int)
            and isinstance(color, list)
            and len(color) == 3
            and all(isinstance(component, int) and 0 <= component <= 255 for component in color)
        ):
            valid_colors[val] = color
        else:
            invalid_items.append((val, color))

    for val, color in invalid_items:
        print(f"警告: Colors字典中存在无效条目 (键: {val}, 值: {color})，已跳过。键应为整数，值应为RGB列表。")

    if not valid_colors:
        return np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    label_min = int(np.min(label))
    label_max = int(np.max(label))
    if label_min >= 0 and label_max <= 65535:
        palette = np.zeros((label_max + 1, 3), dtype=np.uint8)
        for val, color in valid_colors.items():
            if 0 <= val <= label_max:
                palette[val] = color
        return palette[label.astype(np.int64, copy=False)]

    color_label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for val, color in valid_colors.items():
        color_label_img[label == val] = color
    return color_label_img


class VisualizationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像可视化工具")
        self.setGeometry(0, 0, 600, 550)
        self._center_window()
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_DesktopIcon))

        self.image_path: Optional[str] = None
        self.label_path: Optional[str] = None

        self._init_ui()
        self._update_visualize_button_state()

    def _center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_ui(self):
        """Initializes the user interface elements."""
        main_layout = QVBoxLayout(self)

        files_group = QGroupBox("文件选择")
        files_layout = QFormLayout(files_group)
        files_layout.setSpacing(10)

        self.select_image_btn = QPushButton("选择图像文件 (.jpg, .png, .tif)")
        self.select_image_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.select_image_btn.setToolTip("选择要显示的原始图像；大 TIF 会自动按预览尺寸读取，避免整图占用内存。")
        self.select_image_btn.clicked.connect(self.load_image)
        files_layout.addRow(self.select_image_btn)

        self.select_label_btn = QPushButton("选择标签文件 (.png, .jpg, .tif)")
        self.select_label_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.select_label_btn.setToolTip("选择对应的标签图像；大 TIF 标签会用最近邻方式读取预览，保持类别像素值。")
        self.select_label_btn.clicked.connect(self.load_label)
        files_layout.addRow(self.select_label_btn)
        main_layout.addWidget(files_group)

        config_group = QGroupBox("显示配置")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(10)

        config_layout.addWidget(QLabel("类别名称 (Label Names):"))
        self.label_edit = QTextEdit()
        default_label_names = (
            "# 示例格式 (Python 字典):\n"
            "{0: '背景', 1: '建筑', 2: '植被', 3: '水体', 4: '道路'}"
        )
        self.label_edit.setPlaceholderText("输入 label_names，例如：\n{0: 'background', 1: 'tree', ...}")
        self.label_edit.setText(default_label_names) # Set default example
        self.label_edit.setToolTip("定义标签图中像素值对应的类别名称。")
        config_layout.addWidget(self.label_edit)

        config_layout.addWidget(QLabel("类别颜色 (Colors):"))
        self.color_edit = QTextEdit()
        default_colors = (
            "# 示例格式 (Python 字典, RGB值):\n"
            "{0: [0,0,0], 1: [255,0,0], 2: [0,255,0], 3: [0,0,255], 4: [255,255,0]}"
        )
        self.color_edit.setPlaceholderText("输入 colors，例如：\n{0: [0,0,0], 1: [255,0,0], ...}")
        self.color_edit.setText(default_colors) # Set default example
        self.color_edit.setToolTip("定义每个类别在可视化时显示的颜色 (RGB列表)。")
        config_layout.addWidget(self.color_edit)
        main_layout.addWidget(config_group)

        self.visualize_btn = QPushButton("开始可视化")
        self.visualize_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.visualize_btn.setToolTip("根据以上配置显示图像和着色后的标签。")
        self.visualize_btn.clicked.connect(self.visualize)
        main_layout.addWidget(self.visualize_btn, alignment=Qt.AlignCenter)

    def _update_button_text(self, button: QPushButton, base_text: str, path: Optional[str]):
        if path:
            button.setText(f"{base_text}: {os.path.basename(path)}")
        else:
            button.setText(base_text)

    def _update_visualize_button_state(self):
        enabled = bool(self.image_path and self.label_path)
        self.visualize_btn.setEnabled(enabled)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if path:
            self.image_path = path
            self._update_button_text(self.select_image_btn, "图像文件", path)
            self._update_visualize_button_state()

    def load_label(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择标签文件", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.label_path = path
            self._update_button_text(self.select_label_btn, "标签文件", path)
            self._update_visualize_button_state()

    def visualize(self):
        if not self.image_path or not self.label_path:
            QMessageBox.warning(self, "错误", "请先选择图像和标签文件")
            return

        label_names_str = self.label_edit.toPlainText()
        colors_str = self.color_edit.toPlainText()

        # Remove comments before parsing
        cleaned_label_names_str = "\n".join([line for line in label_names_str.splitlines() if not line.strip().startswith("#")])
        cleaned_colors_str = "\n".join([line for line in colors_str.splitlines() if not line.strip().startswith("#")])

        try:
            # Use ast.literal_eval as in the original code for parsing dictionaries
            label_names = ast.literal_eval(cleaned_label_names_str) if cleaned_label_names_str.strip() else {}
            colors = ast.literal_eval(cleaned_colors_str) if cleaned_colors_str.strip() else {}
        except (SyntaxError, ValueError) as e: # More specific exceptions for parsing
            QMessageBox.critical(self, "格式错误", f"Label Names 或 Colors 格式错误，请确保是有效的Python字典。\n错误详情: {e}")
            return
        except Exception as e: # Catch any other parsing related errors
            QMessageBox.critical(self, "格式错误", f"解析 Label Names 或 Colors 时发生未知错误: {e}")
            return

        if not isinstance(colors, dict):
             QMessageBox.critical(self, "格式错误", "Colors 必须是一个有效的 Python 字典格式。")
             return
        # label_names is not directly used in the original plotting logic, but parsed for potential use/consistency

        try:
            image = read_image_preview(self.image_path)
            label = read_label_preview(self.label_path)
            plt = get_pyplot()

            # Original logic for handling label dimensions
            if label.ndim == 3:
                if label.shape[2] >= 1: # Ensure there's at least one channel
                    label = label[:, :, 0]  # Take the first channel
                else:
                    QMessageBox.critical(self, "标签格式错误", "3维标签图像通道数不足。")
                    return
            elif label.ndim != 2: # If not 3-dim, it must be 2-dim
                QMessageBox.critical(self, "标签格式错误", f"标签图像应为单通道灰度图或可转换为单通道的RGB图。当前维度: {label.ndim}")
                return

            color_label_img = build_color_label_image(label, colors)

            # Original Matplotlib plotting logic
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("原始图像（预览）")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(color_label_img)
            plt.title("标签图像（预览）") # label_names is not used for title here as per original
            plt.axis('off')

            plt.show()

        except FileNotFoundError:
            QMessageBox.critical(self, "文件错误", "无法找到指定的图像或标签文件。请检查路径是否正确。")
        except UnidentifiedImageError:
            QMessageBox.critical(self, "图像格式错误", "无法识别图像文件格式，请确保文件未损坏且格式受支持。")
        except Exception as e:
            QMessageBox.critical(self, "可视化错误", f"显示图像时发生未知错误: {e}")
