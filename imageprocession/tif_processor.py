# tif_processor.py
import os
import ast
import traceback
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image as PILImage, UnidentifiedImageError


try:
    import fiona
    import rasterio
    from rasterio.features import rasterize

    FIONA_RASTERIO_AVAILABLE = True
except ImportError:
    FIONA_RASTERIO_AVAILABLE = False

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QLineEdit,
    QMessageBox, QGroupBox, QFormLayout, QProgressBar, QTextEdit,
    QApplication, QDesktopWidget, QStyle
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

DEFAULT_ORDERED_LABELS = """# 每行一个标签名称，顺序将决定 ID (从1开始)
# 示例:
# ABBA
# ACPE
# ACRU
# ACSA
# BEAL
# BEPA
# FAGR
# LALA
# PIST
# Picea
# Populus
# THOC
# TSCA
# Mort
"""

DEFAULT_FIXED_COLOR_MAP = """# Python 字典格式: "LabelName": [R, G, B]
# 示例:
# {
#    "ABBA":    [68, 32, 130],
#    "ACPE":    [28, 46, 43],
#    "ACRU":    [121, 66, 189],
#    "ACSA":    [120, 155, 52],
#    "BEAL":    [130, 183, 14],
#    "BEPA":    [41, 248, 133],
#    "FAGR":    [165, 77, 202],
#    "LALA":    [116, 189, 192],
#    "PIST":    [16, 219, 247],
#    "Picea":   [231, 238, 231],
#    "Populus": [242, 137, 179],
#    "THOC":    [132, 148, 95],
#    "TSCA":    [54, 126, 138],
#    "Mort":    [200, 30, 30]
# }
"""


class TifProcessThread(QThread):
    progress_updated = pyqtSignal(int, str)  # percentage, message
    finished_signal = pyqtSignal(str, str)  # final message, type ("info", "error")

    def __init__(self, gpkg_path: str, tif_path: str, output_dir: str,
                 date_str: str, zone_id_str: str,  # Changed zone_id to str for flexibility
                 ordered_labels_list: List[str],
                 fixed_color_map_dict: Dict[str, List[int]],
                 label_field: str = "Label"):
        super().__init__()
        self.gpkg_path = gpkg_path
        self.tif_path = tif_path
        self.output_dir = output_dir
        self.date_str = date_str
        self.zone_id_str = zone_id_str  # e.g., "3" or "A"
        self.ordered_labels_list = ordered_labels_list
        self.fixed_color_map_dict = fixed_color_map_dict
        self.label_field = label_field
        self._is_interrupted = False

    def requestInterruption(self):
        self._is_interrupted = True
        super().requestInterruption()

    def run(self):
        if not FIONA_RASTERIO_AVAILABLE:
            self.finished_signal.emit(
                "错误：缺少必要的库 (fiona, rasterio)。请安装它们后重试。", "error")
            return

        try:
            self.progress_updated.emit(0, "开始处理 TIF 文件...")

            zone_str_formatted = f"Z{self.zone_id_str}"  # As per original script Z{id}
            base_output_name = f"{self.date_str}-sbl-{zone_str_formatted.lower()}"  # e.g. 2021-10-07-sbl-z3

            out_fp_color = os.path.join(self.output_dir, f"{base_output_name}_labels_color_fixed.tif")
            out_fp_id = os.path.join(self.output_dir, f"{base_output_name}_labels_id_fixed.tif")
            out_fp_palette = os.path.join(self.output_dir, f"{base_output_name}_labels_palette_fixed.tif")

            os.makedirs(self.output_dir, exist_ok=True)
            self.progress_updated.emit(5, f"输出目录 '{self.output_dir}' 已确认/创建。")

            # 1. Create label and color mappings
            label_to_id = {label: idx + 1 for idx, label in enumerate(self.ordered_labels_list)}
            id_to_label = {v: k for k, v in label_to_id.items()}

            id_to_rgb: Dict[int, tuple[int, int, int]] = {}
            for idx_val, label_name in id_to_label.items():
                # Ensure color tuple, not list
                color_list = self.fixed_color_map_dict.get(label_name, [255, 255, 255])
                id_to_rgb[idx_val] = (color_list[0], color_list[1], color_list[2])
            id_to_rgb[0] = (0, 0, 0)  # Background

            mapping_info = "\n📋 类别ID→RGB颜色映射:\n"
            for idx_val in sorted(id_to_label.keys()):
                mapping_info += f"{idx_val:2d}: {id_to_label[idx_val]:20s} → RGB{id_to_rgb[idx_val]}\n"
            mapping_info += " 0: background           → RGB(0, 0, 0)"
            self.progress_updated.emit(10, mapping_info)
            if self._is_interrupted: return

            # 2. Read image metadata from input TIF
            self.progress_updated.emit(15, f"正在读取TIF元数据: {self.tif_path}")
            with rasterio.open(self.tif_path) as src:
                meta = src.meta.copy()
                height, width = src.height, src.width
                transform = src.transform
            self.progress_updated.emit(20, f"TIF元数据读取完成: Height={height}, Width={width}")
            if self._is_interrupted: return

            # 3. Rasterize vector data
            self.progress_updated.emit(25, f"正在栅格化矢量数据: {self.gpkg_path}")
            shapes = []
            unknown_labels_found = []
            with fiona.open(self.gpkg_path, 'r') as src_vec:
                properties_schema = src_vec.schema.get("properties", {})
                if self.label_field not in properties_schema:
                    available_fields = ", ".join(properties_schema.keys()) or "无"
                    self.finished_signal.emit(
                        f"错误：矢量文件中没有类别字段 '{self.label_field}'。\n可用字段：{available_fields}",
                        "error",
                    )
                    return
                for i, feat in enumerate(src_vec):
                    if self._is_interrupted: return
                    label_value = feat["properties"].get(self.label_field)
                    label = str(label_value) if label_value is not None else ""
                    if label not in label_to_id:
                        if label not in unknown_labels_found:
                            unknown_labels_found.append(label)
                            self.progress_updated.emit(25 + (i % 5),
                                                       f"⚠️ 跳过未知标签: {label} (在矢量文件中找到，但未在有序标签列表或颜色映射中定义)")
                        continue
                    geom = feat["geometry"]
                    shapes.append((geom, label_to_id[label]))

            if not shapes:
                self.finished_signal.emit(f"错误：在矢量文件中没有找到与所提供标签匹配的有效几何图形。", "error")
                return
            self.progress_updated.emit(40, f"读取了 {len(shapes)} 个有效几何图形。开始栅格化...")
            if self._is_interrupted: return

            id_raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,  # Background ID
                dtype="uint8"
            )
            self.progress_updated.emit(50, "矢量数据已栅格化为ID图层。")
            if self._is_interrupted: return

            # 4. Output single-channel ID TIF
            self.progress_updated.emit(55, f"正在保存单通道ID标签图: {out_fp_id}")
            meta_id = meta.copy()
            meta_id.update(count=1, dtype="uint8", compress="lzw", driver="GTiff")
            with rasterio.open(out_fp_id, "w", **meta_id) as dst:
                dst.write(id_raster, 1)
            self.progress_updated.emit(65, "单通道ID标签图保存完成。")
            if self._is_interrupted: return

            # 5. Build RGB TIF
            self.progress_updated.emit(70, "正在生成RGB彩色标签图...")
            max_id = int(max(int(id_raster.max()), max(id_to_rgb.keys(), default=0)))
            palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
            for idx_val, rgb_color_tuple in id_to_rgb.items():
                if 0 <= idx_val <= max_id:
                    palette[idx_val] = rgb_color_tuple
            rgb_raster = np.moveaxis(palette[id_raster], -1, 0)
            self.progress_updated.emit(75, "RGB彩色标签图生成完成。")
            if self._is_interrupted: return

            # 6. Write RGB TIF
            self.progress_updated.emit(80, f"正在保存RGB彩色标签图: {out_fp_color}")
            meta_color = meta.copy()
            meta_color.update(count=3, dtype="uint8", compress="lzw", driver="GTiff")
            with rasterio.open(out_fp_color, "w", **meta_color) as dst:
                dst.write(rgb_raster[0], 1)
                dst.write(rgb_raster[1], 2)
                dst.write(rgb_raster[2], 3)
            self.progress_updated.emit(85, "RGB彩色标签图保存完成。")
            if self._is_interrupted: return

            # 7. Generate P-mode TIF
            self.progress_updated.emit(90, "正在生成调色板（P模式）标签图...")
            palette_data = []
            # Create a full 256-entry palette. IDs are 1-based, 0 is background.
            for i in range(256):
                rgb = id_to_rgb.get(i, (0, 0, 0))  # Default to black if ID not in map
                palette_data.extend(rgb)

            palette_img = PILImage.fromarray(id_raster, mode='P')
            palette_img.putpalette(palette_data)
            self.progress_updated.emit(95, f"正在保存调色板（P模式）标签图: {out_fp_palette}")
            palette_img.save(out_fp_palette, format='TIFF',
                             compression='tiff_lzw')  # Specify format and compression for PIL
            self.progress_updated.emit(98, "调色板（P模式）标签图保存完成。")
            if self._is_interrupted: return

            final_message = (
                "✅ 标签图生成完成！\n"
                f"🗂️ 单通道标签图（训练用）: {out_fp_id}\n"
                f"🌈 彩色可视化标签图      : {out_fp_color}\n"
                f"🎨 调色板模式标签图（P） : {out_fp_palette}"
            )
            if unknown_labels_found:
                final_message += f"\n\n⚠️ 注意：在矢量文件中找到但未处理的标签: {', '.join(unknown_labels_found)}"

            self.progress_updated.emit(100, "所有处理完成。")
            self.finished_signal.emit(final_message, "info")

        except FileNotFoundError as e:
            self.finished_signal.emit(f"文件未找到错误: {e}", "error")
        except (fiona.errors.DriverError, rasterio.errors.RasterioIOError) as e:
            self.finished_signal.emit(f"Fiona/Rasterio IO 错误: {e}", "error")
        except UnidentifiedImageError as e:
            self.finished_signal.emit(f"PIL 无法识别图像文件: {e}", "error")
        except Exception as e:
            self.progress_updated.emit(self.property("current_progress_value") or 50, f"发生错误: {e}")
            traceback_str = traceback.format_exc()
            self.finished_signal.emit(f"处理过程中发生未知错误: {e}\n详细信息:\n{traceback_str}", "error")
        finally:
            if self._is_interrupted:
                self.finished_signal.emit("操作已由用户取消。", "info")


class TifProcessorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIF 文件矢量栅格化工具")
        self.setGeometry(0, 0, 750, 700)  # Adjusted size for more content
        self._center_window()
        try:
            self.setWindowIcon(self.style().standardIcon(QStyle.SP_CommandLink))
        except AttributeError:
            pass

        self.gpkg_path: Optional[str] = None
        self.tif_path: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.processor_thread: Optional[TifProcessThread] = None

        self._init_ui()
        self._update_start_button_state()

        if not FIONA_RASTERIO_AVAILABLE:
            QMessageBox.critical(
                self, "依赖缺失",
                "必要的库 (fiona, rasterio) 未找到。\n"
                "此功能将无法使用，请安装它们 (e.g., `pip install fiona rasterio Pillow`)."
            )
            self.start_btn.setEnabled(False)
            for child in self.findChildren(QWidget):  # Disable all widgets in this window
                child.setEnabled(False)

    def _center_window(self):
        try:
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())
        except Exception:
            pass

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # --- Inputs Group ---
        inputs_group = QGroupBox("输入文件与参数")
        inputs_form_layout = QFormLayout(inputs_group)
        inputs_form_layout.setSpacing(10)

        self.gpkg_path_btn = QPushButton("选择矢量文件 (.gpkg)")
        self.gpkg_path_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.gpkg_path_btn.setToolTip("选择包含几何图形和类别字段的 GPKG 文件。")
        self.gpkg_path_btn.clicked.connect(self.select_gpkg_file)
        inputs_form_layout.addRow(self.gpkg_path_btn)

        self.tif_path_btn = QPushButton("选择参考栅格图像 (.tif)")
        self.tif_path_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.tif_path_btn.setToolTip("选择作为栅格化参考（获取尺寸、投影等）的TIF文件。")
        self.tif_path_btn.clicked.connect(self.select_tif_file)
        inputs_form_layout.addRow(self.tif_path_btn)

        self.output_dir_btn = QPushButton("选择输出目录")
        self.output_dir_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.output_dir_btn.setToolTip("选择处理后的标签TIF文件保存的目录。")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        inputs_form_layout.addRow(self.output_dir_btn)

        self.label_field_edit = QLineEdit("Label")
        self.label_field_edit.setToolTip("矢量属性表中表示类别名称的字段，默认使用 Label。")
        inputs_form_layout.addRow(QLabel("类别字段名:"), self.label_field_edit)

        self.date_edit = QLineEdit("2021-10-07")
        self.date_edit.setToolTip("用于输出文件命名的日期字符串 (例如 YYYY-MM-DD)。")
        inputs_form_layout.addRow(QLabel("输出文件名日期 (Date String):"), self.date_edit)

        self.zone_id_edit = QLineEdit("3")
        self.zone_id_edit.setToolTip("用于输出文件命名的区域标识符 (例如 3, A, X1)。")
        inputs_form_layout.addRow(QLabel("输出文件名区域ID (Zone ID):"), self.zone_id_edit)
        main_layout.addWidget(inputs_group)

        # --- Configuration Group ---
        config_group = QGroupBox("标签与颜色配置")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(10)

        config_layout.addWidget(QLabel("有序标签列表 (Ordered Labels - 每行一个):"))
        self.ordered_labels_edit = QTextEdit()
        self.ordered_labels_edit.setPlaceholderText("例如:\nRoad\nBuilding\nWater\n...")
        self.ordered_labels_edit.setText(DEFAULT_ORDERED_LABELS)
        self.ordered_labels_edit.setToolTip("定义矢量类别字段值到整数ID的映射顺序 (ID从1开始)。\n背景将自动设为ID 0。")
        self.ordered_labels_edit.setFixedHeight(120)
        config_layout.addWidget(self.ordered_labels_edit)

        config_layout.addWidget(QLabel("固定颜色映射 (Fixed Color Map - Python字典格式):"))
        self.fixed_color_map_edit = QTextEdit()
        self.fixed_color_map_edit.setPlaceholderText("例如:\n{\n  \"Road\": [255,0,0],\n  \"Building\": [0,255,0]\n}")
        self.fixed_color_map_edit.setText(DEFAULT_FIXED_COLOR_MAP)
        self.fixed_color_map_edit.setToolTip(
            "定义每个标签名称在可视化时显示的颜色 (RGB列表)。\n未在此处定义的标签将使用默认颜色（白色）。")
        self.fixed_color_map_edit.setFixedHeight(120)
        config_layout.addWidget(self.fixed_color_map_edit)
        main_layout.addWidget(config_group)

        # --- Action Button & Progress ---
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.setFixedHeight(35)
        self.start_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("处理日志将显示在此处...")
        self.log_edit.setFixedHeight(150)
        main_layout.addWidget(self.log_edit)

        self.setLayout(main_layout)

    def _update_button_text(self, button: QPushButton, base_text: str, path: Optional[str]):
        if path:
            button.setText(f"{base_text}: {os.path.basename(path)}")
        else:
            button.setText(base_text)

    def _update_start_button_state(self):
        enabled = bool(self.gpkg_path and self.tif_path and self.output_dir and FIONA_RASTERIO_AVAILABLE)
        self.start_btn.setEnabled(enabled)

    def select_gpkg_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择矢量文件", "", "GeoPackage files (*.gpkg);;All Files (*)")
        if path:
            self.gpkg_path = path
            self._update_button_text(self.gpkg_path_btn, "矢量文件 (.gpkg)", path)
            self._update_start_button_state()

    def select_tif_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考栅格图像", "", "TIFF files (*.tif *.tiff);;All Files (*)")
        if path:
            self.tif_path = path
            self._update_button_text(self.tif_path_btn, "参考栅格图像 (.tif)", path)
            self._update_start_button_state()

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir = path
            self._update_button_text(self.output_dir_btn, "输出目录",
                                     os.path.basename(path))  # Show only dir name for brevity
            self._update_start_button_state()

    def _parse_ordered_labels(self) -> Optional[List[str]]:
        labels_text = self.ordered_labels_edit.toPlainText()
        ordered_labels_list = [
            line.strip() for line in labels_text.splitlines()
            if line.strip() and not line.strip().startswith('#')
        ]
        if not ordered_labels_list:
            QMessageBox.warning(self, "配置错误", "有序标签列表不能为空。请至少定义一个标签。")
            return None
        return ordered_labels_list

    def _parse_fixed_color_map(self) -> Optional[Dict[str, List[int]]]:
        color_map_text = self.fixed_color_map_edit.toPlainText()
        # Remove comments before parsing
        cleaned_color_map_text = "\n".join(
            [line for line in color_map_text.splitlines() if not line.strip().startswith("#")])
        if not cleaned_color_map_text.strip():  # Handle empty or all-comment input
            QMessageBox.warning(self, "配置错误", "固定颜色映射不能为空。请提供有效的Python字典。")
            return None
        try:
            fixed_color_map_dict = ast.literal_eval(cleaned_color_map_text)
            if not isinstance(fixed_color_map_dict, dict):
                raise ValueError("颜色映射必须是一个字典。")

            # Validate structure: keys are strings, values are lists of 3 ints (0-255)
            for key, value in fixed_color_map_dict.items():
                if not isinstance(key, str):
                    raise ValueError(f"颜色映射的键必须是字符串。发现: {key} (类型: {type(key)})")
                if not (isinstance(value, list) and len(value) == 3 and
                        all(isinstance(v, int) and 0 <= v <= 255 for v in value)):
                    raise ValueError(f"标签 '{key}' 的颜色值必须是包含3个整数 (0-255) 的列表。发现: {value}")
            return fixed_color_map_dict
        except (SyntaxError, ValueError) as e:
            QMessageBox.warning(self, "配置错误",
                                f"固定颜色映射格式错误: {e}\n请确保是有效的Python字典，键为字符串，值为RGB列表 (例如 \"Label\": [R,G,B])。")
            return None
        except Exception as e:
            QMessageBox.critical(self, "配置错误", f"解析固定颜色映射时发生未知错误: {e}")
            return None

    def start_processing(self):
        if not all([self.gpkg_path, self.tif_path, self.output_dir]):
            QMessageBox.warning(self, "输入缺失", "请选择矢量文件、参考TIF文件和输出目录。")
            return

        date_str = self.date_edit.text().strip()
        zone_id_str = self.zone_id_edit.text().strip()
        if not date_str or not zone_id_str:
            QMessageBox.warning(self, "参数缺失", "日期字符串和区域ID不能为空。")
            return

        label_field = self.label_field_edit.text().strip() or "Label"

        ordered_labels = self._parse_ordered_labels()
        if ordered_labels is None: return

        fixed_color_map = self._parse_fixed_color_map()
        if fixed_color_map is None: return

        # Ensure all ordered_labels have an entry in fixed_color_map for consistency,
        # or inform the user/use a default. The thread currently defaults to white if not found.
        # For now, we'll let the thread handle defaults as per original logic.

        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self.log_edit.append("开始处理...")
        QApplication.processEvents()

        self.processor_thread = TifProcessThread(
            self.gpkg_path, self.tif_path, self.output_dir,
            date_str, zone_id_str, ordered_labels, fixed_color_map, label_field
        )
        self.processor_thread.setProperty("current_progress_value", 0)  # Store progress
        self.processor_thread.progress_updated.connect(self.update_progress)
        self.processor_thread.finished_signal.connect(self.on_processing_finished)
        self.processor_thread.start()

    def update_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.log_edit.append(message)
        self.log_edit.ensureCursorVisible()
        if self.processor_thread:
            self.processor_thread.setProperty("current_progress_value", value)
        QApplication.processEvents()

    def on_processing_finished(self, message: str, msg_type: str):
        self.log_edit.append(f"--------------------\n{msg_type.upper()}: {message}")
        self.log_edit.ensureCursorVisible()
        self._update_start_button_state()

        if msg_type == "info":
            QMessageBox.information(self, "操作完成", message)
            self.progress_bar.setValue(100)
        elif msg_type == "error":
            QMessageBox.critical(self, "操作失败", message)
        # For "warning" or other types, progress might not be 100.

    def closeEvent(self, event):
        if self.processor_thread and self.processor_thread.isRunning():
            reply = QMessageBox.question(self, '退出确认',
                                         "处理正在进行中，确定要退出吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.log_edit.append("用户请求取消操作...")
                self.processor_thread.requestInterruption()
                if not self.processor_thread.wait(3000):
                    self.log_edit.append("线程未能及时停止。")
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    # This is for testing the TifProcessorWindow independently
    app = QApplication([])
    if not FIONA_RASTERIO_AVAILABLE:
        QMessageBox.critical(
            None, "关键依赖缺失",
            "运行此模块需要 fiona 和 rasterio 库。\n请安装它们 (e.g., `pip install fiona rasterio Pillow`) 然后重试。"
        )
    else:
        window = TifProcessorWindow()
        window.show()
    app.exec_()
