import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QLineEdit,
    QHBoxLayout, QMessageBox, QGroupBox, QFormLayout, QProgressBar, QTextEdit,
    QApplication, QDesktopWidget, QStyle
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from typing import Optional, List


class VOCCreatorThread(QThread):
    """Worker thread for VOC dataset creation."""
    progress_updated = pyqtSignal(int, str)  # percentage, message
    finished_signal = pyqtSignal(str, str)  # final message, type ("info", "error")

    def __init__(self, source_image_dir: str, source_label_dir: str,
                 voc_root_output_dir: str, dataset_name: str, train_ratio: float):
        super().__init__()
        self.source_image_dir = source_image_dir
        self.source_label_dir = source_label_dir
        self.voc_root_output_dir = voc_root_output_dir
        self.dataset_name = dataset_name  # e.g., "VOC2012"
        self.train_ratio = train_ratio
        self._is_interrupted = False  # Custom interruption flag

    def requestInterruption(self):  # Override QThread's method
        self._is_interrupted = True
        super().requestInterruption()

    def run(self):
        try:
            self.progress_updated.emit(0, f"Initializing VOC dataset creation for '{self.dataset_name}'...")
            try:
                from sklearn.model_selection import train_test_split
            except ImportError:
                self.finished_signal.emit(
                    "scikit-learn library not found. Please install (`pip install scikit-learn`).",
                    "error",
                )
                return

            # 1. Define VOC paths
            voc_devkit_path = os.path.join(self.voc_root_output_dir, "VOCdevkit")
            dataset_path = os.path.join(voc_devkit_path, self.dataset_name)

            jpeg_images_path = os.path.join(dataset_path, "JPEGImages")
            segmentation_class_path = os.path.join(dataset_path, "SegmentationClass")
            image_sets_path = os.path.join(dataset_path, "ImageSets")
            image_sets_seg_path = os.path.join(image_sets_path, "Segmentation")

            if self._is_interrupted: return

            self.progress_updated.emit(5, "Creating VOC directory structure...")
            os.makedirs(jpeg_images_path, exist_ok=True)
            os.makedirs(segmentation_class_path, exist_ok=True)
            os.makedirs(image_sets_seg_path, exist_ok=True)

            if self._is_interrupted: return

            self.progress_updated.emit(10, "Scanning source directories...")

            image_filenames_copied = []  # Store base names of successfully copied image/label pairs

            # Get all potential image files first
            try:
                source_images_all = os.listdir(self.source_image_dir)
            except FileNotFoundError:
                self.finished_signal.emit(f"Source image directory not found: {self.source_image_dir}", "error")
                return

            source_images_files = [
                f for f in source_images_all
                if os.path.isfile(os.path.join(self.source_image_dir, f)) and
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ]

            total_files_to_process = len(source_images_files)
            if total_files_to_process == 0:
                self.progress_updated.emit(15, "No image files found in the source image directory.")
                self.finished_signal.emit("No image files found to process.", "info")
                return

            self.progress_updated.emit(15, f"Found {total_files_to_process} potential image files. Processing...")
            processed_count = 0

            for img_file in source_images_files:
                if self._is_interrupted:
                    self.finished_signal.emit("Operation cancelled by user.", "info")
                    return

                base_name, img_ext = os.path.splitext(img_file)

                # Attempt to find corresponding label file
                possible_label_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp']
                label_file_found = None

                for lbl_ext in possible_label_extensions:
                    potential_label_file = base_name + lbl_ext
                    path_to_label = os.path.join(self.source_label_dir, potential_label_file)
                    if os.path.exists(path_to_label) and os.path.isfile(path_to_label):
                        label_file_found = potential_label_file
                        break

                processed_count += 1
                current_progress_val = 15 + int(60 * (processed_count / total_files_to_process))

                if not label_file_found:
                    self.progress_updated.emit(
                        current_progress_val,
                        f"Warning: Label for {img_file} not found in {self.source_label_dir}. Skipping."
                    )
                    continue

                src_img_path = os.path.join(self.source_image_dir, img_file)
                src_lbl_path = os.path.join(self.source_label_dir, label_file_found)

                # Standardize filenames for VOC
                dst_img_filename = base_name + ".jpg"
                dst_lbl_filename = base_name + ".png"

                dst_img_path = os.path.join(jpeg_images_path, dst_img_filename)
                dst_lbl_path = os.path.join(segmentation_class_path, dst_lbl_filename)

                try:
                    from PIL import Image as PILImage  # Import here to catch ImportError early if Pillow not installed

                    # Process and save image as JPEG
                    with PILImage.open(src_img_path) as pil_img:
                        if pil_img.mode != "RGB":
                            pil_img_rgb = pil_img.convert("RGB")
                            pil_img_rgb.save(dst_img_path, "JPEG", quality=95)
                        else:
                            pil_img.save(dst_img_path, "JPEG", quality=95)

                    # Process and save label as PNG
                    with PILImage.open(src_lbl_path) as pil_lbl:
                        # Ensure label is single channel (e.g., 'L' for 8-bit grayscale, 'P' for palette)
                        if pil_lbl.mode in ("RGB", "RGBA"):
                            self.progress_updated.emit(
                                current_progress_val,  # Use existing progress value
                                f"Info: Label {label_file_found} ({pil_lbl.mode}) converting to grayscale for PNG."
                            )
                            pil_lbl_l = pil_lbl.convert("L")
                            pil_lbl_l.save(dst_lbl_path, "PNG")
                        elif pil_lbl.mode not in ("L", "P", "1"):
                            self.progress_updated.emit(
                                current_progress_val,  # Use existing progress value
                                f"Info: Label {label_file_found} ({pil_lbl.mode}) attempting conversion to 'L' for PNG."
                            )
                            pil_lbl_l = pil_lbl.convert("L")
                            pil_lbl_l.save(dst_lbl_path, "PNG")
                        else:  # Already a suitable mode
                            pil_lbl.save(dst_lbl_path, "PNG")

                    image_filenames_copied.append(base_name)  # Add base name for splitting

                except ImportError:
                    self.finished_signal.emit("Pillow (PIL) library not found. Please install (`pip install Pillow`).",
                                              "error")
                    return  # Stop thread if PIL is missing
                except Exception as e_pil:
                    self.progress_updated.emit(
                        current_progress_val,
                        f"Warning: Error processing {img_file} or {label_file_found} ({e_pil}). Skipping file."
                    )
                    # Optionally, could attempt shutil.copy2 as a fallback here if desired
                    continue  # Skip this file pair

                self.progress_updated.emit(
                    current_progress_val,
                    f"Processed: {img_file} -> {dst_img_filename}, {label_file_found} -> {dst_lbl_filename}"
                )

            if self._is_interrupted: return

            if not image_filenames_copied:
                self.progress_updated.emit(75, "No image/label pairs were successfully processed and copied.")
                self.finished_signal.emit("No image/label pairs were copied. Aborting split.", "info")
                return

            self.progress_updated.emit(75,
                                       f"Successfully processed {len(image_filenames_copied)} image/label pairs. Splitting dataset...")

            train_names: List[str] = []
            val_names: List[str] = []
            val_size = 1.0 - self.train_ratio

            if not image_filenames_copied:  # Should be caught above
                pass  # train_names and val_names remain empty
            elif len(image_filenames_copied) == 1 and val_size > 0.0 and val_size < 1.0:
                self.progress_updated.emit(78,
                                           "Warning: Only one image/label pair. Assigning to train set as it cannot be split.")
                train_names = list(image_filenames_copied)  # Ensure it's a copy
            elif val_size == 0.0:  # All to train
                train_names = list(image_filenames_copied)
            elif val_size == 1.0:  # All to val
                val_names = list(image_filenames_copied)
            else:  # Normal split for 2 or more items
                if len(image_filenames_copied) < 2:  # Should be caught by the first 'elif' if val_size is between 0 and 1
                    train_names = list(image_filenames_copied)  # Assign all to train if somehow missed
                else:
                    try:
                        train_names, val_names = train_test_split(
                            image_filenames_copied,
                            test_size=val_size,
                            random_state=42,  # Ensures reproducibility
                            stratify=None  # No stratification for simple filename split
                        )
                    except ValueError as e_split:  # e.g. test_size results in 0 for a small sample
                        self.progress_updated.emit(78, f"Warning during split: {e_split}. Assigning all to train.")
                        train_names = list(image_filenames_copied)
                        val_names = []

            with open(os.path.join(image_sets_seg_path, "train.txt"), 'w') as f:
                f.write('\n'.join(train_names))
            self.progress_updated.emit(85, f"train.txt created with {len(train_names)} entries.")

            with open(os.path.join(image_sets_seg_path, "val.txt"), 'w') as f:
                f.write('\n'.join(val_names))
            self.progress_updated.emit(90, f"val.txt created with {len(val_names)} entries.")

            # Create trainval.txt
            trainval_names = sorted(list(set(train_names + val_names)))
            with open(os.path.join(image_sets_seg_path, "trainval.txt"), 'w') as f:
                f.write('\n'.join(trainval_names))
            self.progress_updated.emit(95, f"trainval.txt created with {len(trainval_names)} entries.")

            self.progress_updated.emit(100, "VOC dataset creation complete!")
            self.finished_signal.emit(f"VOC dataset '{self.dataset_name}' created successfully at {dataset_path}",
                                      "info")

        except FileNotFoundError as e:  # This should be caught earlier for source dirs
            self.finished_signal.emit(f"File/Directory not found error: {e}", "error")
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            self.finished_signal.emit(f"An unexpected error occurred: {e}\nTraceback:\n{traceback_str}", "error")
            self.progress_updated.emit(self.progress_bar.value() if hasattr(self, 'progress_bar') else 50,
                                       f"Error: {e}")


class VOCDatasetCreatorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VOC 数据集格式制作工具")
        self.setGeometry(0, 0, 700, 550)
        self._center_window()
        # Attempt to set a standard icon
        try:
            self.setWindowIcon(self.style().standardIcon(QStyle.SP_DriveHDIcon))
        except AttributeError:  # In case standardIcon is not available or style is None
            pass

        self.source_image_dir: Optional[str] = None
        self.source_label_dir: Optional[str] = None
        self.voc_root_output_dir: Optional[str] = None
        self.creator_thread: Optional[VOCCreatorThread] = None

        self._init_ui()
        self._update_start_button_state()

    def _center_window(self):
        try:
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())
        except Exception:  # Fallback if desktop widget is not available
            pass

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        paths_group = QGroupBox("输入与输出路径选择")
        paths_layout = QFormLayout(paths_group)
        paths_layout.setSpacing(10)

        self.source_image_dir_btn = QPushButton("选择原始图像目录")
        try:
            self.source_image_dir_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        except:
            pass
        self.source_image_dir_btn.setToolTip("选择包含原始图像文件的目录 (例如，Image Cutter的输出)。")
        self.source_image_dir_btn.clicked.connect(self.select_source_image_dir)
        paths_layout.addRow(self.source_image_dir_btn)

        self.source_label_dir_btn = QPushButton("选择原始标签目录")
        try:
            self.source_label_dir_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        except:
            pass
        self.source_label_dir_btn.setToolTip("选择包含原始标签掩码文件的目录 (例如，Image Cutter的输出)。")
        self.source_label_dir_btn.clicked.connect(self.select_source_label_dir)
        paths_layout.addRow(self.source_label_dir_btn)

        self.voc_root_output_dir_btn = QPushButton("选择VOC数据集根目录")
        try:
            self.voc_root_output_dir_btn.setIcon(self.style().standardIcon(QStyle.SP_DriveHDIcon))
        except:
            pass
        self.voc_root_output_dir_btn.setToolTip("选择 `VOCdevkit` 将被创建于此的目录。")
        self.voc_root_output_dir_btn.clicked.connect(self.select_voc_root_dir)
        paths_layout.addRow(self.voc_root_output_dir_btn)
        main_layout.addWidget(paths_group)

        params_group = QGroupBox("数据集参数设置")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)

        self.dataset_name_edit = QLineEdit("VOC2012")
        self.dataset_name_edit.setToolTip("生成的VOC数据集的名称 (例如 VOC2007, VOC2012, MyDataset)。")
        params_layout.addRow(QLabel("数据集名称:"), self.dataset_name_edit)

        self.train_ratio_edit = QLineEdit("0.8")
        self.train_ratio_edit.setToolTip("训练集所占比例 (0.0 到 1.0)。剩余部分将作为验证集。")
        params_layout.addRow(QLabel("训练集比例 (0-1):"), self.train_ratio_edit)
        main_layout.addWidget(params_group)

        self.start_btn = QPushButton("开始创建VOC数据集")
        try:
            self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        except:
            pass
        self.start_btn.clicked.connect(self.start_creation_process)
        main_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("处理日志将显示在此处...")
        self.log_edit.setFixedHeight(150)  # Give log a fixed height
        main_layout.addWidget(self.log_edit)

        # Pass progress_bar to the thread instance for error case updates
        if self.creator_thread:
            self.creator_thread.progress_bar = self.progress_bar

    def _update_button_text(self, button: QPushButton, base_text: str, path: Optional[str]):
        if path:
            # For directories, show the full path for clarity
            button.setText(f"{base_text}: {path}")
        else:
            button.setText(base_text)

    def _update_start_button_state(self):
        enabled = bool(self.source_image_dir and self.source_label_dir and self.voc_root_output_dir)
        self.start_btn.setEnabled(enabled)

    def select_source_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择原始图像目录")
        if path:
            self.source_image_dir = path
            self._update_button_text(self.source_image_dir_btn, "原始图像目录", path)
            self._update_start_button_state()

    def select_source_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择原始标签目录")
        if path:
            self.source_label_dir = path
            self._update_button_text(self.source_label_dir_btn, "原始标签目录", path)
            self._update_start_button_state()

    def select_voc_root_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择VOC数据集根输出目录")
        if path:
            self.voc_root_output_dir = path
            self._update_button_text(self.voc_root_output_dir_btn, "VOC数据集根目录", path)
            self._update_start_button_state()

    def start_creation_process(self):
        if not all([self.source_image_dir, self.source_label_dir, self.voc_root_output_dir]):
            QMessageBox.warning(self, "输入缺失", "请选择所有必要的输入和输出目录。")
            return

        dataset_name = self.dataset_name_edit.text().strip()
        if not dataset_name:
            QMessageBox.warning(self, "参数错误", "数据集名称不能为空。")
            return

        try:
            train_ratio = float(self.train_ratio_edit.text())
            if not (0.0 <= train_ratio <= 1.0):
                raise ValueError("训练集比例必须在 0.0 和 1.0 之间。")
        except ValueError as e:
            QMessageBox.warning(self, "参数错误", f"无效的训练集比例: {e}")
            return

        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self.log_edit.append("开始处理...")
        QApplication.processEvents()  # Update UI before thread starts

        self.creator_thread = VOCCreatorThread(
            self.source_image_dir, self.source_label_dir, self.voc_root_output_dir,
            dataset_name, train_ratio
        )
        # Pass progress_bar to the thread instance for error case updates
        self.creator_thread.progress_bar = self.progress_bar
        self.creator_thread.progress_updated.connect(self.update_progress)
        self.creator_thread.finished_signal.connect(self.on_creation_finished)
        self.creator_thread.start()

    def update_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.log_edit.append(message)
        # self.log_edit.ensureCursorVisible() # Scroll to bottom
        QApplication.processEvents()

    def on_creation_finished(self, message: str, msg_type: str):
        self.log_edit.append(f"--------------------\n{msg_type.upper()}: {message}")
        # self.log_edit.ensureCursorVisible()

        # Re-enable start button only if it was originally enabled by having all paths
        self._update_start_button_state()
        # Don't re-enable if it was disabled due to missing paths.
        # If all paths are still valid, _update_start_button_state will enable it.

        if msg_type == "info":
            QMessageBox.information(self, "操作完成", message)
            self.progress_bar.setValue(100)  # Only set to 100 on successful completion
        elif msg_type == "error":
            QMessageBox.critical(self, "操作失败", message)
        # For errors, progress bar remains at its last reported state or the error update.

    def closeEvent(self, event):
        if self.creator_thread and self.creator_thread.isRunning():
            reply = QMessageBox.question(self, '退出确认',
                                         "VOC数据集创建正在进行中，确定要退出吗？\n未完成的操作将丢失。",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.log_edit.append("用户请求取消操作...")
                self.creator_thread.requestInterruption()
                if not self.creator_thread.wait(3000):  # Wait up to 3 seconds
                    self.log_edit.append("线程未能及时停止，可能强制关闭。")
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
