from PointCloud.file_io import read_point_cloud, write_point_cloud
from Toolbox.file_io import atomic_output
from PointCloud.processing import downsample_points, SamplingCancelled
# workers.py
import os
import time
import numpy as np
import open3d as o3d
import math
import traceback
from PyQt5.QtCore import pyqtSignal, QThread
from pathlib import Path  # For ScanNet worker
import random  # For ScanNet worker


# tqdm is for console, will replace with signals for GUI

class PepperProcessWorker(QThread):
    """处理辣椒植株点云的工作线程 (Pipeline, SOR, ROR)"""
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    # Emits (processed_pcd_object, original_input_filepath) for single file mode
    result_signal = pyqtSignal(object, str)

    def __init__(self, processing_type, params, input_dir=None, output_dir=None, single_file=None,
                 enable_intermediate_results=False):
        super().__init__()
        self.processing_type = processing_type
        self.params = params
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.single_file = single_file
        self.enable_intermediate_results = enable_intermediate_results if processing_type == "pipeline" else False
        self._is_running = True
        self.current_file_for_error = ""

    def stop(self):
        self._is_running = False
        self.status_signal.emit("停止信号已发送...")

    def run(self):
        try:
            if not self.output_dir:
                raise ValueError("Output directory must be specified.")

            if self.single_file:
                if not self._is_running: return
                self.current_file_for_error = os.path.basename(self.single_file)
                self.status_signal.emit(
                    f"正在处理文件: {self.current_file_for_error} (模式: {self.processing_type.upper()})")

                processed_pcd = None
                if self.processing_type == "pipeline":
                    processed_pcd, _ = self.process_pipeline(
                        self.single_file,
                        self.output_dir,
                        self.enable_intermediate_results
                    )
                elif self.processing_type == "sor":
                    processed_pcd = self.apply_filter(
                        self.single_file,
                        self.statistical_outlier_removal,
                        self.params,
                        "_sor"
                    )
                elif self.processing_type == "ror":
                    processed_pcd = self.apply_filter(
                        self.single_file,
                        self.radius_outlier_removal,
                        self.params,
                        "_ror"
                    )
                else:
                    raise ValueError(f"未知的处理类型: {self.processing_type}")

                if processed_pcd is not None and self._is_running:
                    self.result_signal.emit(processed_pcd, self.single_file)

                if self._is_running:
                    self.progress_signal.emit(100)

            elif self.input_dir:
                self.batch_process_point_clouds(self.input_dir, self.output_dir)
            else:
                raise ValueError("Either single_file or input_dir must be provided.")

            if self._is_running:
                self.status_signal.emit("处理完成")
            else:
                self.status_signal.emit("处理已中止")
            self.current_file_for_error = ""

        except Exception as e:
            error_context = f"文件 {self.current_file_for_error}" if self.current_file_for_error else f"目录 {self.input_dir or self.output_dir}"
            error_message = f"处理过程出错 ({self.processing_type.upper()} - {error_context}): {str(e)}"
            self.status_signal.emit(error_message)
            print(f"--- PepperProcessWorker Error ({self.processing_type.upper()}) ---")
            traceback.print_exc()
            print(f"---------------------------------")
        finally:
            self.finished_signal.emit()

    def load_point_cloud(self, filepath):
        if not self._is_running: return None
        try:
            pcd = read_point_cloud(filepath)
            if not pcd.has_points():
                if filepath.lower().endswith('.xyz'):
                    try:
                        data = np.loadtxt(filepath)
                        if data.shape[1] < 3: raise ValueError("< 3 columns")
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                        if data.shape[1] >= 6:
                            colors = data[:, 3:6]
                            if np.max(colors) > 1.0: colors = colors / 255.0
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                        if not pcd.has_points(): raise ValueError("Empty after numpy")
                    except Exception as e_xyz:
                        self.status_signal.emit(
                            f"警告: 文件 {os.path.basename(filepath)} 加载失败 (尝试XYZ失败: {e_xyz}).")
                        return None
                else:
                    self.status_signal.emit(f"警告：文件 {os.path.basename(filepath)} 为空或加载失败。")
                    return None
            return pcd
        except Exception as e:
            self.status_signal.emit(f"加载点云 {os.path.basename(filepath)} 出错: {str(e)}")
            return None

    def save_point_cloud(self, pcd, output_path):
        if not self._is_running: return False
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_point_cloud(output_path, pcd)
            return True
        except Exception as write_err:
            self.status_signal.emit(f"保存文件 {os.path.basename(output_path)} 时出错: {write_err}")
            return False

    def apply_filter(self, input_file, filter_func, params, output_suffix):
        if not self._is_running: return None
        self.status_signal.emit(f"加载: {os.path.basename(input_file)}")
        pcd = self.load_point_cloud(input_file)
        if pcd is None: return None

        self.status_signal.emit(f"应用 {filter_func.__name__}...")
        filtered_pcd = filter_func(pcd, **params)

        if not self._is_running: return None

        if filtered_pcd is None or not filtered_pcd.has_points():
            self.status_signal.emit(f"警告：应用 {filter_func.__name__} 后点云为空或无效。")
            return None

        self.status_signal.emit(f"过滤后点数: {len(filtered_pcd.points)}")

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{base_name}{output_suffix}.ply"
        output_path = os.path.join(self.output_dir, output_filename)

        if self.save_point_cloud(filtered_pcd, output_path):
            self.status_signal.emit(f"已保存结果到: {os.path.basename(output_path)}")
        else:
            self.status_signal.emit(f"警告: 保存过滤结果失败, 但处理后的点云仍在内存中。")
        return filtered_pcd

    def process_pipeline(self, input_file, base_output_dir, enable_intermediate_results=False):
        if not self._is_running: return None, None
        result_dir = None
        if enable_intermediate_results:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(base_output_dir, f"{base_name}_pipeline_intermediate_{timestamp}")
            try:
                os.makedirs(result_dir, exist_ok=True)
            except OSError as e:
                self.status_signal.emit(f"无法创建中间结果目录 {result_dir}: {e}")
                enable_intermediate_results = False
                result_dir = None

        self.status_signal.emit("加载点云...")
        pcd = self.load_point_cloud(input_file)
        if pcd is None: return None, result_dir

        self.status_signal.emit(f"原始点云有 {len(pcd.points)} 个点")
        if enable_intermediate_results and result_dir:
            self.save_point_cloud(pcd, os.path.join(result_dir, "0_original.ply"))
        current_pcd = pcd

        if not self._is_running: return None, result_dir
        self.status_signal.emit("步骤1: 使用DBSCAN提取主要结构")
        dbscan_params = {"eps": 0.03, "min_points": 15}
        dbscan_intermediate_filename = "1_dbscan_filtered.ply"
        save_path_dbscan = None
        if enable_intermediate_results and result_dir:
            save_path_dbscan = os.path.join(result_dir, dbscan_intermediate_filename)
            if 'save_path' in self.dbscan_noise_removal.__code__.co_varnames:
                dbscan_params["save_path"] = save_path_dbscan
        try:
            result = self.dbscan_noise_removal(current_pcd, **dbscan_params)
            if isinstance(result, tuple):
                processed_pcd = result[0]
            else:
                processed_pcd = result
            if processed_pcd is None or not processed_pcd.has_points():
                self.status_signal.emit(f"警告：DBSCAN 后点云为空或无效。停止处理此文件。")
                return None, result_dir
            current_pcd = processed_pcd
            if save_path_dbscan and 'save_path' not in self.dbscan_noise_removal.__code__.co_varnames:
                self.save_point_cloud(current_pcd, save_path_dbscan)
        except Exception as e:
            self.status_signal.emit(f"DBSCAN 步骤失败: {str(e)}");
            traceback.print_exc();
            return None, result_dir

        if not self._is_running: return None, result_dir
        self.status_signal.emit(f"处理完成! 最终点云有 {len(current_pcd.points)} 个点")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{base_name}_dbscan_pipeline.ply"
        final_output_path = os.path.join(base_output_dir, output_filename)
        if self.save_point_cloud(current_pcd, final_output_path):
            self.status_signal.emit(f"已保存最终结果到: {os.path.basename(final_output_path)}")
        else:
            self.status_signal.emit(f"警告: 保存最终流程化结果失败。")
        if enable_intermediate_results and result_dir:
            self.status_signal.emit(f"中间结果已保存到目录: {result_dir}")
        return current_pcd, result_dir

    def statistical_outlier_removal(self, pcd, nb_neighbors=20, std_ratio=2.0, save_path=None):
        if not self._is_running or pcd is None or not pcd.has_points(): return None
        try:
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
                inlier_cloud, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors),
                                                                   std_ratio=float(std_ratio))
            if not self._is_running: return None
            num_removed = len(pcd.points) - len(inlier_cloud.points)
            self.status_signal.emit(f"SOR 移除点数: {num_removed}")
            if save_path and inlier_cloud.has_points(): self.save_point_cloud(inlier_cloud, save_path)
            return inlier_cloud
        except Exception as e:
            self.status_signal.emit(f"SOR 滤波失败: {str(e)}"); traceback.print_exc(); return None

    def radius_outlier_removal(self, pcd, nb_points=16, radius=0.05, save_path=None):
        if not self._is_running or pcd is None or not pcd.has_points(): return None
        try:
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
                inlier_cloud, ind = pcd.remove_radius_outlier(nb_points=int(nb_points), radius=float(radius))
            if not self._is_running: return None
            num_removed = len(pcd.points) - len(inlier_cloud.points)
            self.status_signal.emit(f"ROR 移除点数: {num_removed}")
            if save_path and inlier_cloud.has_points(): self.save_point_cloud(inlier_cloud, save_path)
            return inlier_cloud
        except Exception as e:
            self.status_signal.emit(f"ROR 滤波失败: {str(e)}"); traceback.print_exc(); return None

    def dbscan_noise_removal(self, pcd, eps=0.05, min_points=10, save_path=None):
        if not self._is_running: return None, None
        if not pcd or not pcd.has_points(): return None, None
        try:
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
                labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
        except Exception as e:
            self.status_signal.emit(f"DBSCAN聚类计算失败: {str(e)}");
            traceback.print_exc();
            return None, None
        if not self._is_running: return None, None
        if labels is None or len(labels) == 0 or len(labels) != len(pcd.points):
            self.status_signal.emit(f"警告：DBSCAN返回无效或空的标签 (eps={eps}, min_points={min_points})");
            return None, labels
        unique_labels, counts = np.unique(labels, return_counts=True);
        valid_mask = (unique_labels != -1)
        if not np.any(valid_mask):
            self.status_signal.emit(
                f"警告：DBSCAN未找到任何有效聚类 (eps={eps}, min_points={min_points})，所有点均为噪声。");
            return None, labels
        valid_labels = unique_labels[valid_mask];
        valid_counts = counts[valid_mask]
        max_cluster_idx_in_valid = np.argmax(valid_counts);
        max_cluster_label = valid_labels[max_cluster_idx_in_valid]
        indices_to_keep = np.where(labels == max_cluster_label)[0]
        if len(indices_to_keep) == 0: self.status_signal.emit("警告：DBSCAN最大聚类为空。"); return None, labels
        max_cluster_pcd = pcd.select_by_index(indices_to_keep)
        if not self._is_running: return None, labels
        if max_cluster_pcd.has_points():
            if save_path: self.save_point_cloud(max_cluster_pcd, save_path)
            return max_cluster_pcd, labels
        else:
            self.status_signal.emit("警告：选择最大聚类后点云为空。"); return None, labels

    def batch_process_point_clouds(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        valid_extensions = ['.ply', '.pcd', '.xyz', '.pts'];
        point_cloud_files = []
        try:
            for filename in os.listdir(input_dir):
                if not self._is_running: self.status_signal.emit("批量处理中止于文件列表阶段。"); return
                file_path = os.path.join(input_dir, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in valid_extensions: point_cloud_files.append(file_path)
        except FileNotFoundError:
            self.status_signal.emit(f"错误：输入目录 {input_dir} 不存在"); return
        except Exception as e:
            self.status_signal.emit(f"读取输入目录时出错: {str(e)}"); return
        if not point_cloud_files: self.status_signal.emit(f"在 {input_dir} 中未找到有效的点云文件"); return
        if not self._is_running: return
        self.status_signal.emit(
            f"找到 {len(point_cloud_files)} 个点云文件待处理 (模式: {self.processing_type.upper()})")
        total_files = len(point_cloud_files);
        processed_count = 0;
        failed_count = 0
        for i, pc_file in enumerate(point_cloud_files):
            if not self._is_running: self.status_signal.emit(f"批量处理中止于文件 {i + 1}/{total_files}。"); return
            filename = os.path.basename(pc_file);
            self.current_file_for_error = filename;
            base_name, _ = os.path.splitext(filename)
            self.status_signal.emit(f"[{i + 1}/{total_files}] 处理文件: {filename}")
            processed_pcd = None
            try:
                if self.processing_type == "pipeline":
                    processed_pcd, _ = self.process_pipeline(pc_file, output_dir, self.enable_intermediate_results)
                    if processed_pcd is not None and processed_pcd.has_points():
                        processed_count += 1
                    else:
                        failed_count += 1
                else:
                    filter_func = None;
                    output_suffix_for_filename = ""
                    if self.processing_type == "sor":
                        filter_func = self.statistical_outlier_removal; output_suffix_for_filename = "_sor"
                    elif self.processing_type == "ror":
                        filter_func = self.radius_outlier_removal; output_suffix_for_filename = "_ror"
                    else:
                        raise ValueError(f"无效处理类型: {self.processing_type}")
                    pcd_loaded = self.load_point_cloud(pc_file)
                    if pcd_loaded is None: failed_count += 1; continue
                    processed_pcd = filter_func(pcd_loaded, **self.params)
                    if not self._is_running: return
                    if processed_pcd and processed_pcd.has_points():
                        output_filename = f"{base_name}{output_suffix_for_filename}.ply"
                        output_path = os.path.join(output_dir, output_filename)
                        if self.save_point_cloud(processed_pcd, output_path):
                            processed_count += 1
                        else:
                            failed_count += 1
                    elif processed_pcd is None:
                        failed_count += 1
                    else:
                        self.status_signal.emit(f"处理文件 {filename} 结果为空，跳过保存。"); failed_count += 1
            except Exception as e:
                if self._is_running:
                    self.status_signal.emit(f"处理文件 {filename} 时出错: {e}");
                    failed_count += 1
                    print(f"--- Batch Processing Error ({self.processing_type.upper()} - {filename}) ---");
                    traceback.print_exc();
                    print(f"-----------------------------------------")
            progress = int(((i + 1) / total_files) * 100);
            self.progress_signal.emit(progress)
        self.current_file_for_error = ""
        if self._is_running:
            self.status_signal.emit(
                f"批量处理完成 (模式: {self.processing_type.upper()}). 成功: {processed_count}, 失败/跳过: {failed_count}.")


# --- ScanNet Conversion Worker ---
class ConvertScanNetWorker(QThread):
    """Converts TXT files to ScanNet data format and splits them."""
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_dir, output_dir, split_ratio_str, seed_str):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.split_ratio_str = split_ratio_str
        self.seed_str = seed_str
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.status_signal.emit("停止信号已发送 (ScanNet转换)...")

    def run(self):
        try:
            self.status_signal.emit("开始 ScanNet 格式转换...")
            if not self._is_running:
                self.status_signal.emit("ScanNet 转换任务启动后立即被中止。")
                return

            # 1. Validation
            if not self.output_dir or not isinstance(self.output_dir, str) or not self.output_dir.strip():
                raise ValueError(f"传递给工作线程的输出目录无效或为空: '{self.output_dir}'")

            os.makedirs(self.output_dir, exist_ok=True)
            self.status_signal.emit(f"主输出目录: {self.output_dir}")

            try:
                split_ratios = [float(r.strip()) for r in self.split_ratio_str.split(',')]
                if len(split_ratios) != 3 or not math.isclose(sum(split_ratios), 1.0, abs_tol=1e-5):
                    raise ValueError("分割比例必须是3个数字，总和为1.0")
            except ValueError as e:
                self.status_signal.emit(f"错误: 无效的分割比例 '{self.split_ratio_str}': {e}")
                return
            try:
                seed = int(self.seed_str)
            except ValueError:
                self.status_signal.emit(f"错误: 无效的随机种子 '{self.seed_str}'，必须是整数。")
                return

            # 2. File Discovery
            txt_files_all = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".txt")]
            if not txt_files_all:
                self.status_signal.emit(f"在 {self.input_dir} 中未找到 .txt 文件。")
                self.progress_signal.emit(100)
                return

            # 3. Shuffling & Splitting
            random.seed(seed)
            random.shuffle(txt_files_all)

            n_total = len(txt_files_all)
            n_train = int(n_total * split_ratios[0])
            n_val = int(n_total * split_ratios[1])
            train_files = txt_files_all[:n_train]
            val_files = txt_files_all[n_train: n_train + n_val]
            test_files = txt_files_all[n_train + n_val:]

            splits_map = {"train": train_files, "val": val_files, "test": test_files}
            self.status_signal.emit(
                f"总文件数: {n_total}. 训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")

            processed_files_overall_count = 0

            # 4. Main Loop
            for split_name, files_in_split in splits_map.items():
                if not self._is_running: break
                if not files_in_split: continue  # Skip empty splits

                self.status_signal.emit(f"正在处理 {split_name} 集 ({len(files_in_split)} 个文件)...")

                # 5a. Get Paths
                list_file_path = os.path.join(self.output_dir, f"scannetv2_{split_name}.txt")
                split_data_dir = os.path.join(self.output_dir, split_name)
                os.makedirs(split_data_dir, exist_ok=True)

                # 5b. Open List File
                with open(list_file_path, "w") as f_list:
                    # 5c. Inner Loop
                    for txt_filename in files_in_split:
                        if not self._is_running: break

                        current_txt_path = os.path.join(self.input_dir, txt_filename)

                        try:
                            data = np.loadtxt(current_txt_path)
                        except Exception as e:
                            self.status_signal.emit(f"警告: 无法加载 {txt_filename}: {e}. 跳过.")
                            continue

                        if data.ndim == 1 and data.shape[0] == 0:
                            self.status_signal.emit(f"警告: 文件 {txt_filename} 为空. 跳过.")
                            continue
                        if data.ndim == 1:
                            data = data.reshape(1, -1)

                        if data.shape[1] != 10:
                            self.status_signal.emit(
                                f"警告: {txt_filename} 的形状 ({data.shape}) 与预期的 (N, 10) 不符. 跳过.")
                            continue
                        if data.shape[0] == 0:
                            self.status_signal.emit(f"警告: 文件 {txt_filename} 不包含点数据. 跳过.")
                            continue

                        # Extract data
                        coord = data[:, 0:3].astype(np.float32)
                        color = data[:, 3:6].astype(np.float32)
                        segment = data[:, 6].astype(np.int32)
                        normal = data[:, 7:10].astype(np.float32)
                        instance = np.ones_like(segment, dtype=np.int32) * -1

                        sample_name_stem = Path(txt_filename).stem
                        if not sample_name_stem:
                            self.status_signal.emit(f"警告: 文件 '{txt_filename}' 的 stem 为空. 跳过.")
                            continue

                        # Create scene directory and save .npy files
                        scene_specific_files_dir = os.path.join(split_data_dir, sample_name_stem)
                        os.makedirs(scene_specific_files_dir, exist_ok=True)

                        np.save(os.path.join(scene_specific_files_dir, "coord.npy"), coord)
                        np.save(os.path.join(scene_specific_files_dir, "color.npy"), color)
                        np.save(os.path.join(scene_specific_files_dir, "normal.npy"), normal)
                        np.save(os.path.join(scene_specific_files_dir, "segment20.npy"), segment)
                        np.save(os.path.join(scene_specific_files_dir, "instance.npy"), instance)

                        f_list.write(f"{sample_name_stem}\n")

                        processed_files_overall_count += 1
                        if n_total > 0:
                            progress = int((processed_files_overall_count / n_total) * 100)
                            self.progress_signal.emit(progress)

                if not self._is_running: break

            # 6. Final Status
            if not self._is_running:
                self.status_signal.emit("ScanNet转换已中止.")
            else:
                self.status_signal.emit(
                    f"ScanNet格式转换完成! {processed_files_overall_count} 个文件已处理。数据及列表已保存到: {self.output_dir}")
            self.progress_signal.emit(100)

        except Exception as e:
            # 7. Error Handling
            error_msg = f"ScanNet转换过程出错: {str(e)}"
            self.status_signal.emit(error_msg)
            print(f"--- ConvertScanNetWorker Python Exception ---")
            traceback.print_exc()
            print(f"-------------------------------------------")
        finally:
            self.finished_signal.emit()


# --- Check NPY Shape Worker ---
class CheckShapeWorker(QThread):
    """Loads an NPY file and reports its shape, dtype, and ndim."""
    result_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, npy_file_path):
        super().__init__()
        self.npy_file_path = npy_file_path
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.status_signal.emit("停止信号已发送 (NPY检查)...")

    def run(self):
        try:
            self.status_signal.emit(f"正在加载: {os.path.basename(self.npy_file_path)}...")
            if not self._is_running:
                self.status_signal.emit("NPY检查已中止。")
                return

            data = np.load(self.npy_file_path, mmap_mode="r", allow_pickle=False)

            if not self._is_running:
                self.status_signal.emit("NPY检查已中止。")
                return

            shape_info = f"文件: {os.path.basename(self.npy_file_path)}\n"
            shape_info += f"形状 (Shape): {data.shape}\n"
            shape_info += f"数据类型 (Dtype): {data.dtype}\n"
            shape_info += f"维度数 (Ndim): {data.ndim}\n"

            if data.size == 0:
                shape_info += "数据: 数组为空\n"
            elif data.size <= 20:
                shape_info += f"数据预览:\n{data}\n"
            elif data.ndim == 1 and data.shape[0] > 0:
                shape_info += f"前5个元素: {data[:min(5, data.shape[0])]}\n"  # Handle arrays smaller than 5
            elif data.ndim > 1 and data.shape[0] > 0:
                shape_info += f"前10个元素: {data.flat[:min(10, data.size)]}\n"  # Handle rows with < 5 columns

            self.result_signal.emit(shape_info)
            self.status_signal.emit("NPY文件信息检查完成。")

        except FileNotFoundError:
            self.status_signal.emit(f"错误: 文件未找到 {self.npy_file_path}")
            self.result_signal.emit(f"错误: 文件未找到 {self.npy_file_path}")
        except Exception as e:
            error_msg = f"检查NPY文件时出错: {str(e)}"
            self.status_signal.emit(error_msg)
            self.result_signal.emit(error_msg)
            print(f"--- CheckShapeWorker Error ---")
            traceback.print_exc()
            print(f"-----------------------------")
        finally:
            self.finished_signal.emit()


# --- Merge Prediction Worker ---
class MergePredictionWorker(QThread):
    """Merges predicted labels into an original point cloud TXT file."""
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    result_signal = pyqtSignal(str)

    def __init__(self, original_txt_path, prediction_txt_path, output_txt_path):
        super().__init__()
        self.original_txt_path = original_txt_path
        self.prediction_txt_path = prediction_txt_path
        self.output_txt_path = output_txt_path
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.status_signal.emit("停止信号已发送 (预测合并)...")

    def run(self):
        try:
            self.status_signal.emit("开始合并预测标签...")
            if not self._is_running: return

            self.status_signal.emit(f"加载原始点云: {os.path.basename(self.original_txt_path)}")
            points_data = np.loadtxt(self.original_txt_path)
            if not self._is_running: return

            self.status_signal.emit(f"加载预测标签: {os.path.basename(self.prediction_txt_path)}")
            pred_labels_raw = np.loadtxt(self.prediction_txt_path, dtype=int)

            pred_labels = np.atleast_1d(pred_labels_raw).astype(int, copy=False)
            if pred_labels.size > 0:
                min_label = np.min(pred_labels)
                contains_zero = np.any(pred_labels == 0)
                if min_label == 1 and not contains_zero:
                    self.status_signal.emit("检测到预测标签从1开始，将减1处理。")
                    pred_labels = pred_labels - 1
                elif min_label < 0:
                    self.status_signal.emit(f"警告: 预测标签包含负值 ({min_label})，按原样使用。")
                else:  # min_label is 0 or (min_label > 1 and 0 is present)
                    self.status_signal.emit("预测标签似乎已是0开始或包含0，按原样使用。")
            else:  # pred_labels_raw is empty
                self.status_signal.emit("警告: 预测标签文件为空，无法确定索引起始。")

            if not self._is_running: return

            if points_data.shape[0] == 0:
                self.status_signal.emit(f"错误: 原始点云文件 {os.path.basename(self.original_txt_path)} 为空。")
                return
            if pred_labels.shape[0] == 0 and points_data.shape[
                0] != 0:  # Check if labels became empty but points are not
                self.status_signal.emit(
                    f"错误: 预测标签文件 {os.path.basename(self.prediction_txt_path)} 为空或处理后为空。")
                return

            if points_data.ndim == 1:
                points_data = points_data.reshape(1, -1)

            if points_data.shape[0] != pred_labels.shape[0]:
                msg = f"错误: 点云数量 ({points_data.shape[0]}) 与预测标签数量 ({pred_labels.shape[0]}) 不一致！"
                self.status_signal.emit(msg)
                return

            if points_data.shape[1] < 7:
                msg = f"错误: 原始点云文件列数不足 ({points_data.shape[1]})，无法替换标签列 (索引6)。"
                self.status_signal.emit(msg)
                return

            points_data[:, 6] = pred_labels

            if not self._is_running: return

            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_txt_path)
            if output_dir:  # If output_txt_path includes a directory
                os.makedirs(output_dir, exist_ok=True)

            fmt_list = []
            num_cols = points_data.shape[1]

            if num_cols >= 1: fmt_list.extend(['%.8f'] * min(3, num_cols))  # X Y Z
            if num_cols >= 4: fmt_list.extend(['%d'] * min(3, num_cols - 3))  # R G B
            if num_cols >= 7: fmt_list.extend(['%d'])  # Label
            if num_cols > 7:  fmt_list.extend(['%.6f'] * (num_cols - 7))  # Normals etc.

            # Adjust if not enough columns for all categories
            current_len = len(fmt_list)

            if current_len < num_cols:  # Add generic format for remaining columns
                fmt_list.extend(['%s'] * (num_cols - current_len))
            elif current_len > num_cols:  # Trim if too many format specifiers
                # This case should ideally not happen with min() logic but as a safeguard
                temp_fmt_list = []
                temp_len = 0
                for fmt_spec_group in fmt_list:
                    for fmt_spec in fmt_spec_group.split():
                        if temp_len < num_cols:
                            temp_fmt_list.append(fmt_spec)
                            temp_len += 1
                        else:
                            break
                    if temp_len >= num_cols:
                        break
                fmt_list = temp_fmt_list

            if not fmt_list and num_cols > 0:
                fmt_list = ['%s'] * num_cols

            file_format = ' '.join(fmt_list)

            self.status_signal.emit(f"正在保存合并后的文件到: {os.path.basename(self.output_txt_path)}")
            np.savetxt(self.output_txt_path, points_data, fmt=file_format)

            if not self._is_running:
                if os.path.exists(self.output_txt_path):
                    try:
                        os.remove(self.output_txt_path)
                    except OSError:
                        pass
                self.status_signal.emit("合并已中止，输出文件可能未保存或不完整。")
                return

            self.status_signal.emit("预测标签合并完成！")
            self.result_signal.emit(self.output_txt_path)

        except FileNotFoundError as fnf:
            self.status_signal.emit(f"错误: 文件未找到 - {str(fnf)}")
        except Exception as e:
            error_msg = f"合并预测时出错: {str(e)}"
            self.status_signal.emit(error_msg)
            print(f"--- MergePredictionWorker Error ---")
            traceback.print_exc()
            print(f"----------------------------------")
        finally:
            self.finished_signal.emit()


class GenericConvertWorker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_path, output_dir, target_format, is_batch):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.target_format = target_format.lower()
        self.is_batch = is_batch
        self._is_running = True
        self.valid_input_extensions = ['.ply', '.pcd', '.xyz', '.pts']
        self.valid_output_formats = ['pcd', 'ply', 'xyz']

    def stop(self):
        self._is_running = False
        self.status_signal.emit("停止信号已发送...")

    def run(self):
        try:
            if self.target_format not in self.valid_output_formats:
                raise ValueError(f"不支持的目标输出格式: {self.target_format}")
            os.makedirs(self.output_dir, exist_ok=True)
            if self.is_batch:
                self.process_batch()
            else:
                self.process_single_file()
            if self._is_running:
                self.status_signal.emit("转换完成")
            else:
                self.status_signal.emit("转换已中止")
        except ValueError as ve:
            self.status_signal.emit(f"转换错误: {str(ve)}")
        except FileNotFoundError as fnf:
            self.status_signal.emit(f"转换错误: 文件或目录未找到 - {str(fnf)}")
        except Exception as e:
            self.status_signal.emit(f"转换过程出错: {str(e)}")
            print(f"--- GenericConvertWorker Error ---");
            traceback.print_exc();
            print(f"---")
        finally:
            self.finished_signal.emit()

    def process_single_file(self):
        if not self._is_running: return
        if not os.path.isfile(self.input_path): raise FileNotFoundError(f"输入文件不存在: {self.input_path}")
        filename = os.path.basename(self.input_path);
        name_without_ext, input_ext = os.path.splitext(filename)
        if input_ext.lower() == f".{self.target_format}":
            self.status_signal.emit(f"跳过: 输入文件 {filename} 已是目标格式 {self.target_format.upper()}");
            self.progress_signal.emit(100);
            return
        output_filename = f"{name_without_ext}.{self.target_format}";
        output_path = os.path.join(self.output_dir, output_filename)
        self.status_signal.emit(f"正在转换文件: {filename} 到 {self.target_format.upper()}...");
        self.progress_signal.emit(10)
        try:
            pcd = read_point_cloud(self.input_path)
            if not pcd.has_points():
                if self.input_path.lower().endswith('.xyz'):
                    try:
                        data = np.loadtxt(self.input_path)
                        if data.shape[1] < 3: raise ValueError("< 3 columns")
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                        if data.shape[1] >= 6:
                            colors = data[:, 3:6]
                            if np.max(colors) > 1.0 and np.all(colors >= 0) and np.all(colors <= 255):
                                colors = colors / 255.0
                            elif np.any(colors < 0) or np.any(colors > 1.0):  # Clamp if not 0-1 and not 0-255
                                colors = np.clip(colors, 0.0, 1.0)
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                        if not pcd.has_points(): raise ValueError("Empty after numpy")
                    except Exception as e_xyz:
                        self.status_signal.emit(
                            f"警告: 输入文件 {filename} 为空或无法加载 (XYZ尝试失败: {e_xyz})，跳过。");
                        self.progress_signal.emit(100);
                        return
                else:
                    self.status_signal.emit(f"警告: 输入文件 {filename} 为空或无法加载，跳过。");
                    self.progress_signal.emit(100);
                    return

            if not self._is_running:
                return
            self.progress_signal.emit(50)
            self.write_point_cloud(pcd, output_path)
            if not self._is_running:
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path);
                    except OSError:
                        pass; return
            self.status_signal.emit(f"文件已成功转换为: {output_path}");
            self.progress_signal.emit(100)
        except Exception as e:
            self.status_signal.emit(f"转换文件 {filename} 时出错: {str(e)}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path);
                except OSError:
                    pass

    def process_batch(self):
        if not self._is_running: return
        if not os.path.isdir(self.input_path): raise FileNotFoundError(f"输入目录不存在: {self.input_path}")
        input_files = [];
        files_processed = 0;
        files_failed = 0
        try:
            for filename in os.listdir(self.input_path):
                file_path = os.path.join(self.input_path, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in self.valid_input_extensions and ext.lower() != f".{self.target_format}":
                        input_files.append(file_path)
        except Exception as e:
            self.status_signal.emit(f"列出输入目录文件时出错: {e}"); return
        if not input_files: self.status_signal.emit(
            f"在 {self.input_path} 中未找到需要转换为 {self.target_format.upper()} 的有效文件"); self.progress_signal.emit(
            100); return
        total_files = len(input_files);
        self.status_signal.emit(f"找到 {total_files} 个文件待转换为 {self.target_format.upper()}...")
        for i, input_file in enumerate(input_files):
            if not self._is_running: return
            filename = os.path.basename(input_file);
            name_without_ext, _ = os.path.splitext(filename)
            output_filename = f"{name_without_ext}.{self.target_format}";
            output_path = os.path.join(self.output_dir, output_filename)
            self.status_signal.emit(f"[{i + 1}/{total_files}] 转换: {filename}")
            try:
                pcd = read_point_cloud(input_file)
                if not pcd.has_points():
                    if input_file.lower().endswith('.xyz'):
                        try:
                            data = np.loadtxt(input_file)
                            if data.shape[1] < 3: raise ValueError("< 3 columns")
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                            if data.shape[1] >= 6:
                                colors = data[:, 3:6]
                                if np.max(colors) > 1.0 and np.all(colors >= 0) and np.all(colors <= 255):
                                    colors = colors / 255.0
                                elif np.any(colors < 0) or np.any(colors > 1.0):
                                    colors = np.clip(colors, 0.0, 1.0)
                                pcd.colors = o3d.utility.Vector3dVector(colors)
                            if not pcd.has_points(): raise ValueError("Empty after numpy")
                        except Exception as e_xyz:
                            self.status_signal.emit(
                                f"警告: 文件 {filename} 为空或无法加载 (XYZ尝试失败: {e_xyz})，跳过。");
                            files_failed += 1;
                            continue
                    else:
                        self.status_signal.emit(f"警告: 文件 {filename} 为空或无法加载，跳过。");
                        files_failed += 1;
                        continue

                if not self._is_running: return
                self.write_point_cloud(pcd, output_path);
                files_processed += 1
                if not self._is_running:
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path);
                        except OSError:
                            pass; return
            except Exception as e:
                self.status_signal.emit(f"处理文件 {filename} 时出错: {e}");
                files_failed += 1
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path);
                    except OSError:
                        pass
            finally:
                if self._is_running and total_files > 0: progress = int(
                    ((i + 1) / total_files) * 100); self.progress_signal.emit(progress)
        if self._is_running: self.status_signal.emit(
            f"批量转换完成。成功: {files_processed}, 失败/跳过: {files_failed}.")

    def write_point_cloud(self, pcd, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.target_format == "pcd":
            write_point_cloud(output_path, pcd, write_ascii=False)
        elif self.target_format == "ply":
            write_point_cloud(output_path, pcd, write_ascii=True)  # PLY default ascii for wider compatibility
        elif self.target_format == "xyz":
            points = np.asarray(pcd.points);
            data_to_save = points;
            fmt_list = ['%.6f'] * 3
            if pcd.has_colors():
                colors_np = np.asarray(pcd.colors)
                if np.max(colors_np) <= 1.0 and np.min(colors_np) >= 0:
                    colors_int = (colors_np * 255).astype(np.uint8)
                else:
                    colors_int = np.clip(colors_np, 0, 255).astype(np.uint8)
                data_to_save = np.hstack((points, colors_int));
                fmt_list.extend(['%d'] * 3)
            fmt = ' '.join(fmt_list);
            np.savetxt(output_path, data_to_save, fmt=fmt, delimiter=' ')
        else:
            raise ValueError(f"Internal error: Unsupported format {self.target_format} in write_point_cloud")


def _load_txt_pointcloud(filepath):
    """加载TXT格式的点云文件 (Helper function for worker)"""
    try:
        data = np.loadtxt(filepath, ndmin=2)
        if data.size == 0:
            raise ValueError("点云文件为空")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"点云文件至少需要3列坐标，当前为 {data.shape[1]} 列")

        points = data[:, :3]
        colors_or_features = data[:, 3:] if data.shape[1] > 3 else None
        return points, colors_or_features
    except Exception as e:
        # This error will be caught and reported by the worker
        raise IOError(f"加载点云失败 {os.path.basename(filepath)}: {e}")


def _save_txt_pointcloud(filepath, points, colors_or_features=None):
    """保存点云为TXT格式 (Helper function for worker)"""
    try:
        with atomic_output(filepath) as temporary:
            with temporary.open("w", encoding="utf-8") as output:
                for start in range(0, len(points), 65536):
                    rows = points[start:start + 65536]
                    if colors_or_features is not None:
                        rows = np.hstack([rows, colors_or_features[start:start + 65536]])
                    np.savetxt(output, rows, fmt="%.10g")
        return True
    except Exception as e:
        raise IOError(f"保存点云失败 {os.path.basename(filepath)}: {e}")


class DownsampleWorker(QThread):
    """在后台执行点云下采样的工作线程"""
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_path, output_path, method, params, is_batch):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.method = method
        self.params = params
        self.is_batch = is_batch
        self._is_running = True

    def stop(self):
        self._is_running = False
        self.status_signal.emit("停止信号已发送 (下采样)...")

    def run(self):
        try:
            if self.is_batch:
                self.batch_downsample()
            else:
                self.downsample_single_file()

            if not self._is_running:
                self.status_signal.emit("下采样处理已中止。")
            # Final success message handled within methods
        except Exception as e:
            error_msg = f"下采样过程出错: {str(e)}"
            self.status_signal.emit(error_msg)
            print(f"--- DownsampleWorker Error ---")
            traceback.print_exc()
            print(f"-----------------------------")
        finally:
            self.finished_signal.emit()

    def downsample_single_file(self):
        """处理单个文件"""
        if not self._is_running: return
        self.status_signal.emit(f"正在处理: {os.path.basename(self.input_path)}")
        self.progress_signal.emit(10)

        try:
            if Path(self.input_path).resolve() == Path(self.output_path).resolve():
                raise ValueError("输出路径不能覆盖输入文件。")
            points, features = _load_txt_pointcloud(self.input_path)
            if points is None: return

            original_count = len(points)
            self.status_signal.emit(f"原始点数: {original_count:,}")
            self.progress_signal.emit(30)

            if not self._is_running: return
            downsampled_points, downsampled_features = self._perform_downsampling(points, features)
            if downsampled_points is None: return

            self.progress_signal.emit(70)
            if not self._is_running: return

            _save_txt_pointcloud(self.output_path, downsampled_points, downsampled_features)

            self.status_signal.emit(f"✓ 处理完成，已保存到: {os.path.basename(self.output_path)}")
            self.progress_signal.emit(100)

        except (IOError, ValueError) as e:
            self.status_signal.emit(f"错误: {e}")
        except Exception as e:
            self.status_signal.emit(f"处理文件 {os.path.basename(self.input_path)} 时发生未知错误: {e}")

    def batch_downsample(self):
        """批量处理文件夹"""
        if not self._is_running: return
        input_folder = self.input_path
        output_folder = self.output_path

        try:
            output_root = Path(output_folder).resolve()
            if Path(input_folder).resolve() == output_root:
                raise ValueError("批量输出目录不能与输入目录相同。")
            txt_files = []
            for directory, folders, filenames in os.walk(input_folder):
                folders[:] = [name for name in folders if (Path(directory) / name).resolve() != output_root]
                txt_files.extend(os.path.join(directory, name) for name in sorted(filenames)
                                 if name.lower().endswith(".txt"))
            txt_files.sort()
        except Exception as e:
            self.status_signal.emit(f"错误: 无法读取输入文件夹 {input_folder}: {e}")
            return

        if not txt_files:
            self.status_signal.emit("在输入文件夹中未找到 .txt 文件。")
            return

        total_files = len(txt_files)
        success_count = 0
        self.status_signal.emit(f"找到 {total_files} 个文件，开始批量处理...")

        for i, file_path in enumerate(txt_files):
            if not self._is_running: return

            rel_path = os.path.relpath(file_path, input_folder)
            current_output_path = os.path.join(output_folder, rel_path)

            self.status_signal.emit(f"[{i + 1}/{total_files}] 处理: {os.path.basename(file_path)}")

            try:
                points, features = _load_txt_pointcloud(file_path)
                if points is None: continue

                if not self._is_running: return
                downsampled_points, downsampled_features = self._perform_downsampling(points, features)
                if downsampled_points is None: continue

                if not self._is_running: return
                _save_txt_pointcloud(current_output_path, downsampled_points, downsampled_features)
                success_count += 1

            except (IOError, ValueError) as e:
                self.status_signal.emit(f"跳过 {os.path.basename(file_path)}: {e}")
            except Exception as e:
                self.status_signal.emit(f"处理 {os.path.basename(file_path)} 时发生未知错误: {e}")

            self.progress_signal.emit(int(((i + 1) / total_files) * 100))

        if self._is_running:
            self.status_signal.emit(f"批量处理完成。成功: {success_count}, 失败/跳过: {total_files - success_count}")

    def _perform_downsampling(self, points, features):
        try:
            sampled, sampled_features = downsample_points(
                points, features, self.method, self.params,
                should_cancel=lambda: not self._is_running,
                progress=self.progress_signal.emit,
            )
            self.status_signal.emit(f"下采样后点数: {len(sampled):,}")
            return sampled, sampled_features
        except SamplingCancelled:
            return None, None
