"""Validated point sampling with features kept on their original rows."""
from __future__ import annotations

import math
import numbers

import numpy as np


class SamplingCancelled(Exception):
    pass


def downsample_points(points, features, method, params, should_cancel=None, progress=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("点云必须是非空的 N×3 坐标数组。")
    if not np.isfinite(points).all():
        raise ValueError("点云坐标包含 NaN 或无穷值，请先清理数据。")
    if features is not None:
        features = np.asarray(features)
        if features.ndim != 2 or len(features) != len(points):
            raise ValueError("点云特征行数必须与坐标一致。")

    def check_cancel():
        if should_cancel and should_cancel():
            raise SamplingCancelled()

    check_cancel()
    count = len(points)
    rng = np.random.default_rng(params.get("seed"))
    if method in ("random", "uniform", "farthest"):
        target = params.get("target_points", 10000)
        if isinstance(target, bool) or not isinstance(target, numbers.Integral) or target < 1:
            raise ValueError("目标点数必须是正整数。")
        if target >= count:
            return points, features
        if method == "random":
            indexes = np.sort(rng.choice(count, target, replace=False))
        elif method == "uniform":
            indexes = np.linspace(0, count - 1, target, dtype=np.int64)
        else:
            # Subtract coordinates directly: norm identities lose precision at large GIS offsets.
            indexes = np.empty(target, dtype=np.int64)
            distances = np.full(count, np.inf)
            selected = int(rng.integers(count))
            for number in range(target):
                check_cancel()
                indexes[number] = selected
                for start in range(0, count, 65536):
                    check_cancel()
                    delta = points[start:start + 65536] - points[selected]
                    squared = np.einsum("ij,ij->i", delta, delta)
                    np.minimum(distances[start:start + 65536], squared, out=distances[start:start + 65536])
                distances[selected] = -np.inf
                selected = int(np.argmax(distances))
                if progress and number % max(1, target // 100) == 0:
                    progress(30 + int((number + 1) / target * 40))
    elif method == "voxel":
        import open3d as o3d

        voxel_size = float(params.get("voxel_size", 0.01))
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("体素大小必须是有限的正数。")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        centroids, _, groups = cloud.voxel_down_sample_and_trace(
            voxel_size, cloud.get_min_bound() - voxel_size * 0.5,
            cloud.get_max_bound() + voxel_size * 0.5, False,
        )
        centers = np.asarray(centroids.points)
        indexes = np.empty(len(groups), dtype=np.int64)
        # Select a real point nearest each voxel center, preserving RGB scale and all label columns.
        for number, group in enumerate(groups):
            if number % 256 == 0:
                check_cancel()
                if progress:
                    progress(30 + int(number / max(1, len(groups)) * 40))
            members = np.asarray(group, dtype=np.int64)
            delta = points[members] - centers[number]
            indexes[number] = members[np.argmin(np.einsum("ij,ij->i", delta, delta))]
        indexes.sort()
    else:
        raise ValueError(f"未知的下采样方法：{method}")
    check_cancel()
    return points[indexes], features[indexes] if features is not None else None
