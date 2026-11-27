"""Numerical forestry estimates in metre-based local coordinates.

No input/output or Qt dependencies. Missing measurements are explicit, never
replaced by seed-distance diameters. All random sampling is reproducible.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class MetricOptions:
    ground_cell: float = .5
    ground_support: float = 3.
    stem_low: float = .3
    stem_high: float = 2.3
    stem_grid: float = .16
    stem_merge: float = .35
    breast_height: float = 1.3
    dbh_band: float = .24
    min_radius: float = .03
    max_radius: float = .4
    fit_tolerance: float = .02
    min_fit_points: int = 12
    min_inlier: float = .5
    min_arc: float = 120.
    max_assign_distance: float = 6.
    min_tree_points: int = 100
    min_height: float = 2.
    profile_step: float = .25
    volume_step: float = .2
    form_factor: float = .5

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"参数 {name} 必须为有限正数。")
        if self.min_radius >= self.max_radius or self.stem_low >= self.stem_high:
            raise ValueError("半径或树干高度范围的上限必须大于下限。")
        if self.min_inlier > 1 or self.min_arc > 360 or self.form_factor > 1:
            raise ValueError("比例必须不超过 1，圆弧角度不超过 360°。")
        if self.min_fit_points < 6 or int(self.min_fit_points) != self.min_fit_points:
            raise ValueError("胸径拟合至少需要 6 个点。")
        if int(self.min_tree_points) != self.min_tree_points:
            raise ValueError("最少树点数必须为整数。")
        if self.dbh_band >= 2 * self.breast_height or self.fit_tolerance >= self.max_radius:
            raise ValueError("胸径切片或拟合容差过大。")
        return self


def fit_circle(points, options, check=lambda: None):
    """Batched three-point RANSAC + geometric refinement; horizontal diameter."""
    p = np.asarray(points, dtype=float)
    empty = dict(DBH_cm=None, DBH_fit_inlier=None, DBH_fit_rmse_m=None,
                 DBH_arc_deg=None, DBH_points=len(p), DBH_status="insufficient_points")
    if len(p) < options.min_fit_points:
        return empty
    # Duplicate returns must not masquerade as independent angular support.
    p = np.unique(p, axis=0)
    if len(p) < options.min_fit_points:
        return empty
    origin = np.median(p, axis=0)
    p = p - origin
    rng = np.random.default_rng(7831)
    if len(p) > 4000:
        p = p[rng.choice(len(p), 4000, replace=False)]
    best, best_score = None, (-1, -np.inf)
    for _ in range(4):
        check()
        triplets = rng.integers(0, len(p), (64, 3))
        a, b, c = (p[triplets[:, j]] for j in range(3))
        u, v = b-a, c-a
        det = 2*(u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0])
        valid = np.abs(det) > 1e-10
        u, v, a, det = u[valid], v[valid], a[valid], det[valid]
        u2, v2 = (u*u).sum(1), (v*v).sum(1)
        centers = a + np.column_stack((u2*v[:, 1]-v2*u[:, 1], u[:, 0]*v2-v[:, 0]*u2))/det[:, None]
        radii = np.linalg.norm(centers-a, axis=1)
        valid = (radii >= options.min_radius) & (radii <= options.max_radius)
        for center, radius in zip(centers[valid], radii[valid]):
            errors = np.abs(np.linalg.norm(p-center, axis=1)-radius)
            inlier = errors <= options.fit_tolerance
            score = (int(inlier.sum()), -float(errors[inlier].mean()) if inlier.any() else -np.inf)
            if score > best_score:
                best, best_score = np.r_[center, radius], score
    if best is None:
        return {**empty, "DBH_status": "no_circle"}
    for _ in range(2):
        check()
        errors = np.abs(np.linalg.norm(p-best[:2], axis=1)-best[2])
        inlier = errors <= options.fit_tolerance
        if inlier.sum() < options.min_fit_points:
            return {**empty, "DBH_status": "insufficient_inliers"}
        q = p[inlier]
        result = least_squares(lambda x: np.linalg.norm(q-x[:2], axis=1)-x[2], best,
                               bounds=([-np.inf, -np.inf, options.min_radius],
                                       [np.inf, np.inf, options.max_radius]),
                               loss="soft_l1", f_scale=options.fit_tolerance/2, max_nfev=80)
        best = result.x
    errors = np.abs(np.linalg.norm(p-best[:2], axis=1)-best[2])
    inlier = errors <= options.fit_tolerance
    angles = np.sort(np.mod(np.arctan2(*(p[inlier]-best[:2])[:, ::-1].T), 2*np.pi))
    arc = float(360 - np.degrees(np.diff(np.r_[angles, angles[0]+2*np.pi]).max())) if len(angles) else 0.
    occupied = len(np.unique(np.floor(angles/(2*np.pi)*36)))
    fraction = float(inlier.mean())
    rmse = float(np.sqrt(np.mean(errors[inlier]**2))) if inlier.any() else None
    reason = "ok"
    if inlier.sum() < options.min_fit_points or fraction < options.min_inlier:
        reason = "weak_inliers"
    elif arc < options.min_arc or occupied < max(6, int(options.min_arc/15)):
        reason = "partial_arc"
    elif best[2] <= options.min_radius*1.01 or best[2] >= options.max_radius*.99:
        reason = "radius_at_limit"
    return dict(DBH_cm=float(best[2]*200) if reason == "ok" else None,
                DBH_fit_inlier=fraction, DBH_fit_rmse_m=rmse, DBH_arc_deg=arc,
                DBH_points=empty["DBH_points"], DBH_status=reason,
                center=(best[:2]+origin).tolist() if reason == "ok" else None)


def detect_stems(xyz, heights, options, check=lambda: None):
    """Sparse XY occupancy graph: memory scales with occupied cells, not extent."""
    mask = np.isfinite(heights) & (heights >= options.stem_low) & (heights <= options.stem_high)
    p, h = xyz[mask, :2], heights[mask]
    if len(p) < options.min_fit_points:
        raise ValueError("树干高度带内点数不足；请检查地面或使用已有单木编号。")
    cells, inv, counts = np.unique(np.floor(p/options.stem_grid).astype(np.int64), axis=0,
                                   return_inverse=True, return_counts=True)
    low, high = np.full(len(cells), np.inf), np.full(len(cells), -np.inf)
    np.minimum.at(low, inv, h)
    np.maximum.at(high, inv, h)
    good = (counts >= max(3, min(20, np.median(counts)*.5))) & (high-low >= .5*(options.stem_high-options.stem_low))
    cells, weights = cells[good], counts[good]
    if not len(cells):
        raise ValueError("未找到连续树干，建议先使用单木分割并复核编号。")
    check()
    pairs = cKDTree(cells).query_pairs(1.42, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells),)*2)
    n, group = connected_components(graph, directed=False)
    centers = np.column_stack([np.bincount(group, (cells[:, j]+.5)*weights)/np.bincount(group, weights)
                               for j in range(2)])*options.stem_grid
    support = np.bincount(group, weights)
    used = np.zeros(n, bool)
    index, merged = cKDTree(centers), []
    for i in np.argsort(-support, kind="stable"):
        check()
        if used[i]:
            continue
        neighbors = np.asarray(index.query_ball_point(centers[i], options.stem_merge))
        neighbors = neighbors[~used[neighbors]]
        used[neighbors] = True
        merged.append(np.average(centers[neighbors], axis=0, weights=support[neighbors]))
    return np.asarray(merged)


def area(points):
    if len(points) < 3:
        return None
    try:
        return float(ConvexHull(points).volume)  # SciPy's volume is area in 2D.
    except QhullError:
        return None


def measure_tree(tree_id, points, heights, support_distances, origin, options, check=lambda: None, stem_center=None):
    """Compute one row; points are grouped once by the workflow."""
    flags = []
    valid = np.isfinite(heights)
    p, h = points[valid], heights[valid]
    row = dict(TreeID=int(tree_id), X=None, Y=None, Height_m=None, HeightMax_m=None,
               ClearBole_m=None, CrownBase_m=None, CrownLength_m=None, CrownRatio=None,
               CrownEW_m=None, CrownNS_m=None, CrownDiameter_m=None, CrownArea_m2=None,
               CrownVolume_m3=None, CrownSliceCoverage=None, StemVolume_m3=None,
               NPoints=len(points), GroundSupportedFraction=float(valid.mean()),
               GroundDistanceP95_m=float(np.quantile(support_distances, .95)))
    if not len(p):
        row.update(fit_circle(np.empty((0, 2)), options))
        row["QualityFlags"] = "ground_unsupported"
        row["ReviewRequired"] = True
        return row
    low = (h >= options.stem_low) & (h <= options.stem_high)
    center = (np.asarray(stem_center) if stem_center is not None else
              np.median(p[low, :2] if low.any() else p[:, :2], axis=0))
    # Breast-height slice is measured against terrain at each return location.
    breast = np.abs(h-options.breast_height) <= options.dbh_band/2
    near = np.linalg.norm(p[:, :2]-center, axis=1) <= options.max_radius*2 + .2
    fit = fit_circle(p[breast & near, :2], options, check)
    fitted_center = fit.pop("center", None)
    if fitted_center is not None:
        center = np.array(fitted_center)
    row.update(fit)
    row.update(X=float(center[0]+origin[0]), Y=float(center[1]+origin[1]))
    if fit["DBH_status"] != "ok":
        flags.append("dbh_"+fit["DBH_status"])
    if not low.any():
        flags.append("no_low_stem_support")
    if valid.mean() < .95:
        flags.append("ground_support_incomplete")
    if len(p) < options.min_tree_points:
        flags.append("sparse_tree")
    height = float(np.quantile(h, .99))
    row.update(Height_m=height, HeightMax_m=float(h.max()))
    if height < options.min_height:
        flags.append("low_height")
    if h.max()-height > max(1., .1*height):
        flags.append("top_outliers_or_sparse_top")
    # A horizontal slice over a leaning bole can bias diameter; flag clear drift.
    lower, upper = (h >= .5) & (h < 1.), (h >= 1.6) & (h < 2.1)
    if lower.sum() >= 12 and upper.sum() >= 12:
        if np.linalg.norm(np.median(p[lower, :2], axis=0)-np.median(p[upper, :2], axis=0)) > .2:
            flags.append("possible_lean_or_branches")
    k = np.floor(np.maximum(h, 0)/options.profile_step).astype(np.int64)
    order = np.argsort(k, kind="stable")
    keys, starts, counts = np.unique(k[order], return_index=True, return_counts=True)
    widths = np.zeros(len(keys))
    for j, (start, count) in enumerate(zip(starts, counts)):
        if count >= 6:
            q = p[order[start:start+count], :2]
            widths[j] = float(np.diff(np.quantile(q, [.02, .98], axis=0), axis=0).mean())
    good = (widths >= widths.max()*.5) & (widths > options.max_radius*2)
    consecutive = np.flatnonzero(good[:-1] & good[1:] & (np.diff(keys) == 1))
    if not len(consecutive):
        flags.append("crown_base_unresolved")
    else:
        base = float(keys[consecutive[0]]*options.profile_step)
        crown = p[(h >= base) & (h <= height)]
        crown_h = h[(h >= base) & (h <= height)]
        width = np.diff(np.quantile(crown[:, :2], [.02, .98], axis=0), axis=0)[0]
        row.update(ClearBole_m=base, CrownBase_m=base, CrownLength_m=max(0., height-base),
                   CrownRatio=max(0., height-base)/height if height > 0 else None,
                   CrownEW_m=float(width[0]), CrownNS_m=float(width[1]), CrownDiameter_m=float(width.mean()),
                   CrownArea_m2=area(crown[:, :2]))
        # Integrate slice convex hulls. Empty/degenerate slices stay missing.
        n = max(1, int(math.ceil((height-base)/options.volume_step)))
        bins = np.minimum(np.floor((crown_h-base)/options.volume_step).astype(np.int64), n-1)
        order = np.argsort(bins, kind="stable")
        keys, starts, counts = np.unique(bins[order], return_index=True, return_counts=True)
        volume, covered = 0., 0
        for key, start, count in zip(keys, starts, counts):
            check()
            a = area(crown[order[start:start+count], :2]) if count >= 6 else None
            if a is not None:
                volume += a * min(options.volume_step, height-(base+key*options.volume_step))
                covered += 1
        row["CrownSliceCoverage"] = covered/n
        if covered/n >= .8:
            row["CrownVolume_m3"] = volume
        else:
            flags.append("crown_slices_incomplete")
    if row["DBH_cm"] is not None and height >= options.min_height:
        row["StemVolume_m3"] = math.pi*(row["DBH_cm"]/200)**2*height*options.form_factor
    row["QualityFlags"] = "|".join(flags)
    row["ReviewRequired"] = bool(flags)
    return row
