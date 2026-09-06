from PointCloud.file_io import read_point_cloud, write_point_cloud
# visualization.py
import os
import numpy as np
import open3d as o3d
from PyQt5.QtCore import QTimer  # QCoreApplication removed as direct GUI access from thread is risky
import ast  # For parsing color map string
import traceback


def _run_visualization(geometries, window_name="点云可视化", width=1024, height=768):
    """
    Runs Open3D visualization. Intended to be run in a separate thread.
    """
    try:
        print("启动 Open3D 可视化...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"{window_name} (按 H 查看帮助)",
            width=width, height=height,
            # Consider adding point_show_normal=True if normals are often present and useful
        )
        print("Open3D 可视化窗口已关闭。")
    except Exception as e:
        error_msg = f"可视化线程出错: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        # Error reporting to GUI should be handled by the caller via status_update_func or similar


def _run_sequential_visualization(original_pcd, processed_pcd, original_path, status_update_func=None):
    """
    Runs sequential Open3D visualization. Intended for a separate thread.
    Uses a callback function (status_update_func) via QTimer to update status.
    """
    original_name = os.path.basename(original_path)
    try:
        if status_update_func:
            QTimer.singleShot(0, lambda: status_update_func(f"显示原始点云: {original_name}..."))

        print(f"Visualizing Original: {original_name} ({len(original_pcd.points)} points)")
        _run_visualization(
            [original_pcd],
            window_name=f"原始: {original_name} ({len(original_pcd.points)} pts) - 关闭此窗口查看处理后结果",
        )

        if status_update_func:
            QTimer.singleShot(0, lambda: status_update_func(f"显示处理后点云..."))

        print(f"Visualizing Processed: ({len(processed_pcd.points)} points)")
        _run_visualization(
            [processed_pcd],
            window_name=f"处理后 ({len(processed_pcd.points)} pts)",
        )

        print("Sequential visualization finished.")
        if status_update_func:
            QTimer.singleShot(0, lambda: status_update_func("对比可视化完成。"))

    except Exception as e:
        error_msg = f"对比可视化出错: {str(e)}"
        print(f"Error during sequential visualization thread: {error_msg}")
        traceback.print_exc()
        if status_update_func:
            QTimer.singleShot(0, lambda: status_update_func(error_msg))


def parse_color_map_string_for_vis(text_map_str, status_update_func=None):
    """ Parses a string representation of a color map into a dictionary for visualization. """
    color_map = {}
    default_color = [0.5, 0.5, 0.5]
    for line_num, line in enumerate(text_map_str.splitlines()):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            if '#' in line: line = line.split('#', 1)[0].strip()
            key_str, value_str = line.split(':', 1)
            key = int(key_str.strip())
            value_list = ast.literal_eval(value_str.strip())

            if not (isinstance(value_list, list) and len(value_list) == 3 and all(
                    isinstance(v, (int, float)) for v in value_list)):
                raise ValueError("颜色值必须是包含3个数字的列表 (R,G,B)")

            parsed_color = []
            for v_idx, v_comp in enumerate(value_list):
                v_comp_float = float(v_comp)
                if 0 <= v_comp_float <= 1:
                    parsed_color.append(v_comp_float)
                elif 1 < v_comp_float <= 255:
                    parsed_color.append(v_comp_float / 255.0)
                else:
                    raise ValueError(f"颜色分量 {v_comp_float} (索引 {v_idx}) 超出范围 [0, 255] 或 [0, 1]")
            color_map[key] = parsed_color

        except Exception as e:
            err_msg = f"警告: 无法解析颜色映射行 {line_num + 1}: '{line}'. 错误: {e}"
            print(err_msg)
            if status_update_func: QTimer.singleShot(0, lambda msg=err_msg: status_update_func(msg))

    if not color_map and text_map_str.strip():
        warn_msg = "警告: 颜色映射为空或所有行都无法解析。将使用默认颜色。"
        print(warn_msg)
        if status_update_func: QTimer.singleShot(0, lambda msg=warn_msg: status_update_func(msg))

    return color_map, default_color


def _run_custom_color_visualization(pcd_txt_path, color_map_text_str, screenshot_path=None, status_update_func=None):
    """
    Loads a TXT point cloud, applies custom colors based on labels (column 6),
    visualizes it, and optionally saves a screenshot.
    """
    try:
        if status_update_func: QTimer.singleShot(0, lambda: status_update_func(
            f"加载点云: {os.path.basename(pcd_txt_path)}..."))

        try:
            data = np.loadtxt(pcd_txt_path)
        except Exception as e:
            raise ValueError(f"无法加载点云文件 '{pcd_txt_path}': {e}")

        if data.ndim == 1 and data.shape[0] == 0: raise ValueError("点云文件为空。")
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] < 7: raise ValueError(f"点云文件列数 ({data.shape[1]}) 不足7列，无法获取标签 (索引6)。")

        points = data[:, 0:3]
        labels = data[:, 6].astype(int)

        if status_update_func: QTimer.singleShot(0, lambda: status_update_func("解析颜色映射..."))
        color_map_dict, default_color_val = parse_color_map_string_for_vis(color_map_text_str, status_update_func)

        unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
        color_palette = np.array(
            [color_map_dict.get(int(label), default_color_val) for label in unique_labels],
            dtype=float
        )
        applied_colors = color_palette[inverse_indices]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(applied_colors)

        if not pcd.has_points():
            raise ValueError("处理后点云为空。")

        window_title = f"自定义颜色可视化: {os.path.basename(pcd_txt_path)}"
        if screenshot_path: window_title += " (关闭窗口后自动保存截图)"

        if status_update_func: QTimer.singleShot(0, lambda: status_update_func("启动可视化窗口..."))

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, width=1280, height=720)
        vis.add_geometry(pcd)

        # Set render options for better visualization (optional)
        opt = vis.get_render_option()
        opt.point_size = 2.0  # Default is 5, can be too large for dense clouds
        # opt.background_color = np.asarray([0.15, 0.15, 0.15])  # Darker gray
        opt.background_color = np.asarray([1.0, 1.0, 1.0])  # 白色背景
        vis.run()  # Blocking call, user adjusts view

        if screenshot_path:
            if status_update_func: QTimer.singleShot(0, lambda: status_update_func(
                f"正在保存截图到 {os.path.basename(screenshot_path)}..."))

            # Ensure directory exists for screenshot
            screenshot_dir = os.path.dirname(screenshot_path)
            if screenshot_dir:  # If path includes a directory
                os.makedirs(screenshot_dir, exist_ok=True)

            success = vis.capture_screen_image(screenshot_path, do_render=True)
            if success:
                msg = f"截图已保存: {os.path.basename(screenshot_path)}"
                print(msg)
                if status_update_func: QTimer.singleShot(0, lambda m=msg: status_update_func(m))
            else:
                msg = f"错误: 保存截图失败 {os.path.basename(screenshot_path)}"
                print(msg)
                if status_update_func: QTimer.singleShot(0, lambda m=msg: status_update_func(m))

        vis.destroy_window()
        if status_update_func: QTimer.singleShot(0, lambda: status_update_func("自定义颜色可视化完成。"))

    except Exception as e:
        error_msg = f"自定义颜色可视化出错: {str(e)}"
        print(f"--- _run_custom_color_visualization Error ---")
        traceback.print_exc()
        if status_update_func: QTimer.singleShot(0, lambda msg=error_msg: status_update_func(msg))


def prepare_point_cloud_for_visualization(pcd, default_color=None):
    """
    Prepares a deep copy of a point cloud for visualization, adding color if needed.
    Returns an empty PointCloud if input is invalid.
    """
    if not pcd or not isinstance(pcd, o3d.geometry.PointCloud) or not pcd.has_points():
        print("Warning: Preparing empty or invalid point cloud for visualization.")
        return o3d.geometry.PointCloud()

    try:
        pcd_copy = o3d.geometry.PointCloud()
        pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())

        if pcd.has_colors():
            colors = np.asarray(pcd.colors).copy()  # Operate on a copy
            # Check color range - Open3D expects 0-1 floats
            max_c, min_c = np.max(colors), np.min(colors)
            if max_c > 1.0 and min_c >= 0 and max_c <= 255:  # Heuristic for 0-255
                print("Info: Input colors seem to be 0-255, normalizing to 0-1 for visualization.")
                colors = colors / 255.0
            elif min_c < 0 or max_c > 1.0:  # If not 0-1 and not clearly 0-255
                print(f"Warning: Colors are outside expected 0-1 range (min: {min_c}, max: {max_c}). Clamping to 0-1.")
                colors = np.clip(colors, 0.0, 1.0)
            pcd_copy.colors = o3d.utility.Vector3dVector(colors)
        elif default_color is not None:
            if isinstance(default_color, (list, tuple)) and len(default_color) == 3:
                pcd_copy.paint_uniform_color(default_color)
            else:
                print(f"Warning: Invalid default_color provided: {default_color}. Using gray.")
                pcd_copy.paint_uniform_color([0.5, 0.5, 0.5])
        else:
            pcd_copy.paint_uniform_color([0.5, 0.5, 0.5])

        if pcd.has_normals():
            pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())

        return pcd_copy
    except Exception as e:
        print(f"Error preparing point cloud copy for visualization: {str(e)}")
        traceback.print_exc()
        return o3d.geometry.PointCloud()
