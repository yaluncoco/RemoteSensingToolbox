"""Conservative, stem-constrained correction of TreeIso proposals.

The engine labels remain available for audit. Only stems supported by circles at
three heights can split a proposal. Unanchored low vegetation is never absorbed
just because it is close in XY. All distances use local metre coordinates.
"""
from dataclasses import asdict, dataclass

import laspy
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from PointCloud.forest_metrics import MetricOptions, detect_stems, fit_circle


REFINEMENT_VERSION = "stem-graph-2"


@dataclass(frozen=True)
class RefinementOptions:
    slice_heights: tuple = (.7, 1.25, 1.8, 2.35)
    slice_thickness: float = .22
    stem_voxel: float = .025
    slice_gap: float = .10
    min_slices: int = 3
    max_lean: float = .45
    max_axis_residual: float = .08
    graph_gap: float = .65
    fragment_agreement: float = .85
    fragment_min_contacts: int = 8
    mixed_low_fraction: float = .5
    min_low_stem_share: float = .1


def read_stem_band(source, origin, ground, check=lambda: None):
    """Read only the low-height band from full-resolution records, in chunks.

    Use the same nearest-eight ground estimator as segmentation. Thin at 2.5 cm,
    not the canopy's 20 cm spacing: thin trunks otherwise lose their circle.
    """
    tree = cKDTree(ground[:, :2])
    parts, heights = [], []
    with laspy.open(source) as reader:
        for chunk in reader.chunk_iterator(400_000):
            check()
            p = np.column_stack((chunk.x, chunk.y, chunk.z)) - origin
            # This broad elevation bound cannot discard a supported stem point.
            p = p[(p[:, 2] >= ground[:, 2].min()+.4) & (p[:, 2] <= ground[:, 2].max()+2.6)]
            if not len(p):
                continue
            d, near = tree.query(p[:, :2], k=min(8, len(ground)), workers=4)
            if d.ndim == 1:
                d, near = d[:, None], near[:, None]
            h = p[:, 2]-np.median(ground[near, 2], axis=1)
            mask = (d[:, 0] < 3) & (h >= .4) & (h <= 2.6)
            p, h = p[mask], h[mask]
            if len(p):
                _, idx = np.unique(np.floor(p/.025).astype(np.int64), axis=0, return_index=True)
                parts.append(p[idx].astype("f4"))
                heights.append(h[idx].astype("f4"))
    if not parts:
        return np.empty((0, 3)), np.empty(0)
    p, h = np.concatenate(parts), np.concatenate(heights)
    _, idx = np.unique(np.floor(p/.025).astype(np.int64), axis=0, return_index=True)
    return p[idx], h[idx]


def detect_anchors(xyz, height, options=None, check=lambda: None):
    """Track circle centres through several slices, allowing a leaning axis.

    Slice components separate neighbouring boles before fitting. A circle in
    one slice alone (a branch, shrub or accidental arc) is not a tree seed.
    """
    options = options or RefinementOptions()
    fit_options = MetricOptions(min_fit_points=12, min_arc=150., min_inlier=.65,
                                fit_tolerance=.025)
    # Vertical occupancy also proposes search windows when understory connects
    # an otherwise valid circular stem to a much larger slice component.
    try:
        proposals = detect_stems(xyz, height, fit_options, check)
    except ValueError:
        proposals = np.empty((0, 2))
    slices = []
    for level in options.slice_heights:
        check()
        p = xyz[np.abs(height-level) <= options.slice_thickness/2, :2]
        if len(p) < 12:
            slices.append([])
            continue
        # Bound graph size independently of scanner return density.
        _, idx = np.unique(np.floor(p/.015).astype(np.int64), axis=0, return_index=True)
        p = p[idx]
        pairs = cKDTree(p).query_pairs(options.slice_gap, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p),)*2)
        _, groups = connected_components(graph, directed=False)
        order = np.argsort(groups, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(groups[order]))+1, len(order)]
        circles = []
        candidates = []
        for start, end in zip(starts[:-1], starts[1:]):
            q = p[order[start:end]]
            if len(q) < 12 or np.ptp(q, axis=0).max() > 1.1:
                continue
            candidates.append((q, None))
        index = cKDTree(p)
        for center in proposals:
            q = p[index.query_ball_point(center, .7)]
            if len(q) >= 12:
                candidates.append((q, center))
        for q, proposal in candidates:
            check()
            if proposal is not None and any(np.linalg.norm(np.array(c["center"])-proposal) < .2 for c in circles):
                continue
            fit = fit_circle(q, fit_options, check)
            if fit["DBH_status"] == "ok":
                circle = {"height": level, "center": fit["center"], "radius": fit["DBH_cm"]/200,
                          "rmse": fit["DBH_fit_rmse_m"], "points": len(q)}
                duplicate = next((i for i, c in enumerate(circles) if
                                  np.linalg.norm(np.array(c["center"])-circle["center"]) < .08 and
                                  abs(c["radius"]-circle["radius"]) < .04), None)
                if duplicate is None:
                    circles.append(circle)
                elif circle["rmse"] < circles[duplicate]["rmse"]:
                    circles[duplicate] = circle
        circles.sort(key=lambda c: (*c["center"], c["radius"]))
        slices.append(circles)
    # Short tracks are extended only if the best match is unambiguous. A missing
    # slice is allowed; the other three still have to agree on one straight axis.
    tracks = []
    for circles in slices:
        check()
        choices = []
        for i, track in enumerate(tracks):
            last = track[-1]
            for j, circle in enumerate(circles):
                dz = circle["height"]-last["height"]
                if not 0 < dz <= 1.11:
                    continue
                distance = np.linalg.norm(np.array(circle["center"])-last["center"])
                radius_delta = abs(circle["radius"]-last["radius"])
                if distance <= options.max_lean*dz+.04 and radius_delta <= max(.04, .3*last["radius"]):
                    choices.append((distance/dz + radius_delta, i, j))
        used_tracks, used_circles = set(), set()
        for score, i, j in sorted(choices):
            if i in used_tracks or j in used_circles:
                continue
            # Close competing matches indicate a fork/crossing, not two proven stems.
            rivals = [s for s, ti, cj in choices if (ti == i or cj == j) and (ti, cj) != (i, j)]
            if rivals and min(rivals)-score < .06:
                continue
            tracks[i].append(circles[j])
            used_tracks.add(i)
            used_circles.add(j)
        tracks.extend([[c] for j, c in enumerate(circles) if j not in used_circles])
    anchors = []
    for track in tracks:
        if len(track) < options.min_slices:
            continue
        h = np.array([c["height"] for c in track])
        centers = np.array([c["center"] for c in track])
        design = np.column_stack((np.ones(len(h)), h-1.3))
        axis = np.linalg.lstsq(design, centers, rcond=None)[0]
        residual = float(np.linalg.norm(design @ axis-centers, axis=1).max())
        if residual > options.max_axis_residual or np.linalg.norm(axis[1]) > options.max_lean:
            continue
        anchors.append(dict(center=axis[0].tolist(), lean=axis[1].tolist(),
                            radius=float(np.median([c["radius"] for c in track])),
                            slices=len(track), axis_residual_m=residual,
                            fit_rmse_m=float(np.mean([c["rmse"] for c in track])),
                            height_min=float(h.min()), height_max=float(h.max())))
    anchors.sort(key=lambda a: (a["center"][1], a["center"][0]))
    return anchors


def _groups(ids):
    order = np.argsort(ids, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(ids[order]))+1, len(order)]
    return [order[a:b] for a, b in zip(starts[:-1], starts[1:])]


def _graph(points, gap):
    distances, neighbors = cKDTree(points).query(points, k=min(13, len(points)), workers=4)
    if distances.ndim == 1:
        return coo_matrix((len(points),)*2).tocsr()
    source = np.broadcast_to(np.arange(len(points))[:, None], neighbors.shape)
    valid = (distances > 0) & (distances <= gap)
    d = distances[valid]
    # A long edge through a crown contact costs more than several short edges
    # along a branch; this is a 3D path metric, not horizontal nearest-stem cells.
    cost = d * (1+(d/.2)**2)
    graph = coo_matrix((cost, (source[valid], neighbors[valid])), shape=(len(points),)*2).tocsr()
    return graph.maximum(graph.T)


def refine_labels(xyz, height, raw, anchors, resolution=.2, options=None, check=lambda: None):
    """Return corrected positive labels, metadata per label and auditable changes.

    Different validated stems never merge. Ambiguous or disconnected material
    keeps a separate proposal, with a reason for manual review.
    """
    options = options or RefinementOptions()
    xyz, height, raw = np.asarray(xyz), np.asarray(height), np.asarray(raw)
    if len(xyz) != len(height) or len(raw) != len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(height).all():
        raise ValueError("Invalid refinement point arrays")
    if not len(raw):
        raise ValueError("No vegetation to refine")
    labels = np.zeros(len(raw), dtype=np.int64)
    seed = np.zeros(len(raw), dtype=np.int64)
    best = np.full(len(raw), np.inf)
    low_idx = np.flatnonzero((height >= .4) & (height <= 2.6))
    low_tree = cKDTree(xyz[low_idx, :2]) if len(low_idx) else None
    active = {}
    for ai, anchor in enumerate(anchors, 1):
        check()
        if low_tree is None:
            break
        near = np.array(low_tree.query_ball_point(anchor["center"], anchor["radius"]+1.3*np.linalg.norm(anchor["lean"])+max(.14, resolution)), dtype=int)
        idx = low_idx[near]
        axis = np.asarray(anchor["center"])+(height[idx, None]-1.3)*np.asarray(anchor["lean"])
        distance = np.linalg.norm(xyz[idx, :2]-axis, axis=1)
        keep = distance <= anchor["radius"]+max(.10, resolution*.6)
        idx, distance = idx[keep], distance[keep]
        if len(idx) < 5 or np.ptp(height[idx]) < .8:
            continue
        score = distance/(anchor["radius"]+.05)
        win = score < best[idx]
        seed[idx[win]], best[idx[win]] = ai, score[win]
        active[ai] = anchor
    # An overlapping fitted tube must not leave a nominal stem with no seeds.
    active = {ai: a for ai, a in active.items() if (seed == ai).sum() >= 5}
    seed[~np.isin(seed, list(active))] = 0
    info = {ai: {"anchor_id": ai, "reason": "validated_stem"} for ai in active}
    next_label = len(anchors)+1
    groups = _groups(raw)
    split_raw, pending, split_details = [], [], []
    gap = max(options.graph_gap, resolution*2.5)
    for idx in groups:
        check()
        owners = []
        for ai in np.unique(seed[idx]):
            if not ai:
                continue
            support = idx[seed[idx] == ai]
            if len(support) >= 3 and np.ptp(height[support]) >= .4:
                owners.append(int(ai))
        if len(owners) == 1:
            # A few stray trunk points do not establish ownership of a large
            # low-vegetation block. Keep such mixed proposals separate rather
            # than absorbing their understory into another part of the tree.
            low = (height[idx] >= .4) & (height[idx] <= 2.6)
            supported = seed[idx] == owners[0]
            unsupported_low = int((low & ~supported).sum())
            mixed = (unsupported_low >= max(50, options.mixed_low_fraction*len(idx))
                     and supported.sum() < options.min_low_stem_share*low.sum())
            if mixed:
                pending.append((idx, "mixed_low_vegetation"))
            else:
                labels[idx] = owners[0]
        elif len(owners) > 1:
            graph = _graph(xyz[idx], gap)
            seeds = np.flatnonzero(np.isin(seed[idx], owners))
            check()
            distance, _, sources = dijkstra(graph, directed=False, indices=seeds, min_only=True, return_predecessors=True)
            check()
            reached = np.isfinite(distance)
            labels[idx[reached]] = seed[idx[ sources[reached] ]]
            # Disconnected canopy pieces retain their identity until the same
            # conservative contact test used for other fragments accepts them.
            if (~reached).any():
                _, components = connected_components(graph[~reached][:, ~reached], directed=False)
                for orphan in _groups(components):
                    pending.append((idx[~reached][orphan], "disconnected_from_stem"))
            split_raw.append(int(raw[idx[0]]))
            split_details.append({"raw_treeiso_id": int(raw[idx[0]]), "anchor_ids": owners,
                                  "unreached_points": int((~reached).sum())})
        else:
            pending.append((idx, "no_validated_stem"))
    # Attach an entire crown fragment only when independent boundary contacts
    # agree. Do not divide an ambiguous fragment or pull in an unseeded low stem.
    # Contacts always reference anchored groups, so weak chains cannot accumulate.
    assigned = np.flatnonzero(labels)
    tree = cKDTree(xyz[assigned]) if len(assigned) else None
    fragment_merges = []
    for idx, reason in pending:
        check()
        low = height[idx] < 2.3
        protected = (reason == "mixed_low_vegetation" or
                     (low.sum() >= 5 and np.ptp(height[idx][low]) >= .65))
        winner, agreement, contacts = 0, 0., 0
        if tree is not None and not protected:
            # The lower attachment is stronger evidence than touching crown
            # tips higher up. Compare contacts at the fragment's base, and only
            # follow edges towards lower points of an already anchored tree.
            boundary = idx[height[idx] <= np.quantile(height[idx], .05)+.5]
            d, nn = tree.query(xyz[boundary], k=min(24, len(assigned)), distance_upper_bound=gap, workers=4)
            if d.ndim == 1:
                d, nn = d[:, None], nn[:, None]
            valid = np.isfinite(d)
            lower = np.zeros(nn.shape, dtype=bool)
            lower[valid] = height[assigned[nn[valid]]] <= np.broadcast_to(height[boundary, None]-.04, nn.shape)[valid]
            valid &= lower
            d[~valid] = np.inf
            nearest = np.argmin(d, axis=1)
            closest = d[np.arange(len(d)), nearest, None]
            contacts = int(np.isfinite(closest).sum())
            if contacts >= options.fragment_min_contacts:
                votes = np.zeros(nn.shape, dtype=np.int64)
                votes[valid] = labels[assigned[nn[valid]]]
                # Give every boundary point unit mass; dense return clusters do
                # not get twelve times the vote. Competing near paths abstain.
                nearest_vote = votes[np.arange(len(d)), nearest]
                competition = valid & (votes != nearest_vote[:, None]) & (d <= closest+.15)
                confident = np.isfinite(closest[:, 0]) & ~competition.any(axis=1)
                ids, counts = np.unique(nearest_vote[confident], return_counts=True)
                pick = int(np.argmax(counts)) if len(counts) else None
                agreement = float(counts[pick]/contacts) if pick is not None else 0.
                # Require support across a meaningful fraction of the fragment,
                # not one accidental contact at an overlapping canopy tip.
                if pick is not None and agreement >= options.fragment_agreement and contacts >= max(8, .05*len(idx)):
                    winner = int(ids[pick])
        if winner:
            labels[idx] = winner
            fragment_merges.append(dict(raw_treeiso_id=int(raw[idx[0]]), anchor_id=winner,
                                        points=len(idx), contacts=contacts, agreement=agreement))
        else:
            labels[idx] = next_label
            info[next_label] = {"anchor_id": None, "reason":
                               reason if reason == "mixed_low_vegetation" else
                               "unverified_low_stem" if protected else reason}
            next_label += 1
    mapping = []
    for idx in _groups(labels):
        label = int(labels[idx[0]])
        sources, counts = np.unique(raw[idx], return_counts=True)
        info[label]["raw_treeiso_ids"] = sources.astype(int).tolist()
        info[label]["dominant_raw_id"] = int(sources[np.argmax(counts)])
        mapping.append(dict(refined_label=label, **info[label], sample_points=len(idx)))
    merged = [r for r in mapping if r["anchor_id"] is not None and len(r["raw_treeiso_ids"]) > 1]
    parameters = asdict(options)
    parameters["slice_heights"] = list(options.slice_heights)
    report = dict(version=REFINEMENT_VERSION, enabled=True, options=parameters,
                  detected_anchor_count=len(anchors), active_anchor_count=len(active),
                  raw_segment_count=len(groups), refined_segment_count=len(mapping),
                  split_segment_count=len(split_raw), merged_group_count=len(merged),
                  attached_fragment_count=len(fragment_merges),
                  unresolved_group_count=sum(r["anchor_id"] is None for r in mapping),
                  anchors=[dict(anchor_id=i, active=i in active, **a) for i, a in enumerate(anchors, 1)],
                  splits=split_details, fragment_attachments=fragment_merges, groups=mapping)
    return labels, info, report
