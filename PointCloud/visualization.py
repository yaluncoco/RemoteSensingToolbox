from PointCloud.file_io import read_point_cloud, write_point_cloud
# visualization.py
import os
import numpy as np
import open3d as o3d
from PyQt5.QtCore import QTimer  # QCoreApplication removed as direct GUI access from thread is risky
import ast  # For parsing color map string
import traceback
from Toolbox.file_io import atomic_output
from PointCloud.file_io import _ascii_workspace
from PointCloud.visualization_data import (
    VisualizationOptions, read_visualization_text, extract_visualization_data, apply_label_colors,
)


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
    """Parse label colors without silently accepting malformed or duplicate entries."""
    color_map = {}
    default_color = [0.5, 0.5, 0.5]
    for line_num, line in enumerate(text_map_str.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        try:
            key_str, value_str = line.split(':', 1)
            key = int(key_str.strip())
            values = ast.literal_eval(value_str.strip())
            if not (isinstance(values, list) and len(values) == 3 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)):
                raise ValueError("颜色值必须是包含 3 个数字的列表 [R, G, B]")
            color = np.asarray(values, dtype=float)
            if not np.isfinite(color).all() or np.any((color < 0) | (color > 255)):
                raise ValueError("颜色值必须位于 0–1 或 0–255 范围")
            if key in color_map:
                raise ValueError(f"标签 {key} 重复，请每个标签只写一行")
            color_map[key] = (color / (255 if color.max() > 1 else 1)).tolist()
        except Exception as exc:
            raise ValueError(f"颜色映射第 {line_num} 行有误：{exc}") from exc
    return color_map, default_color


def _run_custom_color_visualization(pcd_txt_path, color_map_text_str, screenshot_path=None,
                                    status_update_func=None, options=None, finished_func=None):
    """Read the selected layout. GUI callbacks must be thread-safe (Qt signals)."""
    options = options or VisualizationOptions()
    vis = None
    window_created = False

    def status(message):
        if status_update_func:
            status_update_func(message)

    try:
        status(f"加载并检查点云: {os.path.basename(pcd_txt_path)}...")
        data = read_visualization_text(pcd_txt_path, options)
        points, labels, colors = extract_visualization_data(data, options)
        detail = f"{len(points):,} 个点"
        if labels is not None:
            color_map, default_color = parse_color_map_string_for_vis(color_map_text_str)
            colors = apply_label_colors(labels, color_map, default_color)
            unique_labels = np.unique(labels)
            missing = sum(int(label) not in color_map for label in unique_labels)
            detail += f"，{len(unique_labels)} 类标签"
            if missing:
                detail += f"（{missing} 类未配色，显示灰色）"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        window_title = f"自定义颜色可视化: {os.path.basename(pcd_txt_path)}"
        if screenshot_path:
            window_title += " (关闭窗口后自动保存截图)"
        status(f"启动可视化窗口：{detail}...")
        vis = o3d.visualization.Visualizer()
        window_created = vis.create_window(window_name=window_title, width=1280, height=720)
        if not window_created:
            raise RuntimeError("无法创建三维窗口，请检查显卡驱动和 OpenGL 支持。")
        if not vis.add_geometry(pcd):
            raise RuntimeError("点云无法加入三维窗口。")
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        status(f"可视化中：{detail}；关闭三维窗口后返回。")
        vis.run()

        if screenshot_path:
            status(f"正在保存截图到 {os.path.basename(screenshot_path)}...")
            # Open3D returns None even on success. Check the file, not the return value.
            # An ASCII temporary PNG also avoids Open3D's Windows Unicode-path issue.
            from PIL import Image
            with _ascii_workspace() as directory:
                capture = directory / "capture.png"
                vis.capture_screen_image(str(capture), do_render=True)
                if not capture.is_file() or not capture.stat().st_size:
                    raise OSError("保存截图失败。")
                with Image.open(capture) as screenshot, atomic_output(screenshot_path) as output:
                    screenshot.convert("RGB").save(output)
            status(f"可视化完成：{detail}；截图已保存至 {screenshot_path}")
        else:
            status(f"可视化完成：{detail}。")
    except Exception as exc:
        traceback.print_exc()
        status(f"自定义颜色可视化出错: {exc}")
    finally:
        try:
            if vis is not None and window_created:
                vis.destroy_window()
        finally:
            if finished_func:
                finished_func()


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
