"""Embedded orthographic 3D point viewer with depth-correct instance picking.

NumPy projects a bounded preview into a QImage. No external window or OpenGL
context is needed; the identical renderer is testable in packaged/offscreen Qt.
"""
import math
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget


def instance_colors(ids):
    ids = np.asarray(ids, dtype=np.uint64)
    # Use integer arithmetic so large/noncontiguous IDs remain distinct.
    value = ids * np.uint64(2654435761)
    rgb = np.column_stack((80 + (value % 155), 90 + ((value >> np.uint64(8)) % 155),
                           80 + ((value >> np.uint64(16)) % 155))).astype("u1")
    rgb[ids == 0] = (104, 122, 131)
    return rgb


class ForestCanvas(QWidget):
    tree_selected = pyqtSignal(object)
    isolation_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("林业三维点云：点击选择树木，拖动旋转，右键拖动平移，滚轮缩放")
        self.scene = None
        self.selected_id = 0
        self.isolated = False
        self.show_ground = True
        self.show_labels = True
        self.normalized = False
        self.color_mode = "instance"
        self.point_size = 2
        self.yaw, self.elevation = -35., 32.
        self.zoom = 1.
        self.pan = np.zeros(2)
        self.center = np.zeros(3)
        self.radius = 1.
        self._image = None
        self._pixels = np.empty((0, 2))
        self._pixel_ids = np.empty(0, dtype="i8")
        self._labels = []
        self._anchors = {}
        self._press = None
        self._dragged = False
        self._interacting = False
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.timeout.connect(self._settled)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._redraw)

    def set_scene(self, scene):
        self.scene = scene
        self._anchors = {}
        if scene is not None:
            self._points = np.asarray(scene["xyz"], dtype="f8")
            self._normalized_points = self._points.copy()
            self._normalized_points[:, 2] = np.nan_to_num(scene["height"], nan=0.)
            for row in scene["rows"]:
                tree_id = int(row["TreeID"])
                mask = scene["ids"] == tree_id
                p = self._points[mask]
                h = np.asarray(scene["height"])[mask]
                if len(p):
                    xy = np.median(p[:, :2], axis=0)
                    self._anchors[tree_id] = (np.r_[xy, p[:, 2].max()], np.r_[xy, np.nanmax(h) if np.isfinite(h).any() else 0.])
        self.selected_id = 0
        self.isolated = False
        self.reset_view()

    def points(self):
        return self._normalized_points if self.normalized else self._points

    def reset_view(self, direction=None):
        if direction == "top":
            self.yaw, self.elevation = 0., 90.
        elif direction == "front":
            self.yaw, self.elevation = 0., 0.
        elif direction == "iso":
            self.yaw, self.elevation = -35., 32.
        self.zoom = 1.
        self.pan[:] = 0
        if self.scene is not None:
            points = self.points()
            if self.isolated and self.selected_id:
                points = points[self.scene["ids"] == self.selected_id]
            if len(points):
                self.center = (points.min(0) + points.max(0)) / 2
                self.radius = max(.5, float(np.linalg.norm(points.max(0) - points.min(0)) / 2))
                xy, _, _ = self._project(points)
                spans = np.maximum(np.ptp(xy, axis=0), 1.)
                self.zoom = min(max(80, self.width()-90)/spans[0], max(80, self.height()-100)/spans[1])
                self.pan = -(xy.min(0) + xy.max(0) - np.array([self.width(), self.height()])) / 2 * self.zoom
        self.invalidate()

    def select_tree(self, tree_id):
        self.selected_id = int(tree_id)
        if self.isolated:
            self.reset_view()
        self.invalidate()

    def isolate(self, value):
        self.isolated = bool(value) and bool(self.selected_id)
        self.reset_view()

    def invalidate(self):
        self._image = None
        # Coalesce trackpad events into at most one frame every 16 ms.
        if not self._timer.isActive():
            self._timer.start(16)

    def _redraw(self):
        if self._image is None:
            self._render()
        self.update()

    def _settled(self):
        self._interacting = False
        self.invalidate()

    def resizeEvent(self, event):
        self.invalidate()
        super().resizeEvent(event)

    def _project(self, points):
        a, e = math.radians(self.yaw), math.radians(self.elevation)
        right = np.array([math.cos(a), -math.sin(a), 0.])
        depth = np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])
        up = np.cross(right, depth)
        p = points - self.center
        scale = min(self.width(), self.height()) * .43 * self.zoom / self.radius
        xy = np.column_stack((p @ right, -(p @ up))) * scale
        xy += np.array([self.width() / 2, self.height() / 2]) + self.pan
        return xy, p @ depth, scale

    def _render(self):
        width, height = max(1, self.width()), max(1, self.height())
        canvas = np.empty((height, width, 3), dtype=np.uint8)
        canvas[:] = (19, 35, 44)
        self._pixels = np.empty((0, 2))
        self._pixel_ids = np.empty(0, dtype="i8")
        self._labels = []
        if self.scene is not None and len(self.scene["xyz"]):
            ids = self.scene["ids"]
            mask = np.ones(len(ids), dtype=bool)
            if not self.show_ground:
                mask &= ids > 0
            if self.isolated and self.selected_id:
                mask &= ids == self.selected_id
            if self.normalized:
                mask &= np.isfinite(self.scene["height"])
            indices = np.flatnonzero(mask)
            interacting = self._dragged or self._interacting
            if interacting and len(indices) > 40000:
                indices = indices[np.linspace(0, len(indices)-1, 40000, dtype=np.int64)]
            points, ids = self.points()[indices], ids[indices]
            colors = instance_colors(ids)
            if self.color_mode == "rgb" and bool(self.scene["has_rgb"]):
                colors = self.scene["rgb"][indices].copy()
            elif self.color_mode == "height":
                h = np.nan_to_num(self.scene["height"][indices], nan=0.)
                t = np.clip(h / max(1., float(np.quantile(h, .99)) if len(h) else 1.), 0, 1)
                colors = np.column_stack((45 + 200*t, 170 + 45*np.sin(t*np.pi), 205 - 140*t)).astype("u1")
            if self.selected_id and not self.isolated:
                colors[ids != self.selected_id] = (colors[ids != self.selected_id] * .56 + 10).astype("u1")
                colors[ids == self.selected_id] = (255, 210, 105)
            elif self.selected_id and self.color_mode == "instance":
                colors[ids == self.selected_id] = (237, 207, 130)
            xy, depth, scale = self._project(points)
            if len(depth) and self.color_mode != "rgb":
                shade = .64 + .36 * (depth - depth.min()) / max(.001, float(np.ptp(depth)))
                colors = (colors * shade[:, None]).astype("u1")
            pixels = np.rint(xy).astype(np.int64)
            visible = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                       (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
            pixels, depth, ids, colors = pixels[visible], depth[visible], ids[visible], colors[visible]
            # Expand splats, then resolve visibility once for both drawing/picking.
            size = 1 if interacting else self.point_size
            offsets = [(0, 0)] if size == 1 else [(x, y) for x in range(size) for y in range(size)]
            pp = np.concatenate([pixels + offset for offset in offsets])
            dd = np.tile(depth, len(offsets))
            ii = np.tile(ids, len(offsets))
            cc = np.tile(colors, (len(offsets), 1))
            good = (pp[:, 0] < width) & (pp[:, 1] < height)
            pp, dd, ii, cc = pp[good], dd[good], ii[good], cc[good]
            flat = pp[:, 1] * width + pp[:, 0]
            order = np.lexsort((-dd, flat))
            _, first = np.unique(flat[order], return_index=True)
            take = order[first]
            self._pixels, self._pixel_ids = pp[take], ii[take]
            canvas[self._pixels[:, 1], self._pixels[:, 0]] = cc[take]
            self._scale = scale
            if self.show_labels:
                # Labels are projected from actual preview geometry, not guessed XY heights.
                label_ids = [self.selected_id] if self.isolated else [int(r["TreeID"]) for r in self.scene["rows"]][:120]
                if self.selected_id and self.selected_id not in label_ids:
                    label_ids.append(self.selected_id)
                for tree_id in label_ids:
                    if tree_id not in self._anchors:
                        continue
                    anchor = self._anchors[tree_id][int(self.normalized)]
                    pos, _, _ = self._project(anchor[None])
                    x, y = pos[0]
                    if 25 < x < width - 25 and 25 < y < height - 25:
                        self._labels.append((int(tree_id), float(x), float(y - 12)))
        self._image = QImage(canvas.data, width, height, int(canvas.strides[0]), QImage.Format_RGB888).copy()

    def paintEvent(self, event):
        if self._image is None:
            self._render()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawImage(0, 0, self._image)
        if self.scene is None:
            painter.setPen(QColor("#dce8e8"))
            painter.drawText(self.rect().adjusted(35, 20, -35, -20), Qt.AlignCenter | Qt.TextWordWrap,
                             "林业实例工作台\n\n打开已有结果，或新建参数分析\n\n点击树木查看参数 · 拖动旋转 · 滚轮缩放")
        for tree_id, x, y in self._labels:
            text = str(tree_id)
            tw = painter.fontMetrics().horizontalAdvance(text) + 14
            painter.setBrush(QColor("#d6aa55" if tree_id == self.selected_id else "#233f4b"))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(x - tw/2, y - 10, tw, 21), 6, 6)
            painter.setPen(QColor("#172d35" if tree_id == self.selected_id else "#e2edf0"))
            painter.drawText(QRectF(x - tw/2, y - 10, tw, 21), Qt.AlignCenter, text)
        if self.scene is not None:
            painter.setPen(QColor("#9bb4c1"))
            painter.drawText(16, self.height() - 15, "左键旋转 / 点选  ·  右键平移  ·  滚轮缩放  ·  双击独立查看")
            scale = getattr(self, "_scale", 1.)
            metres = 10 ** math.floor(math.log10(max(.001, 90 / scale)))
            px = metres * scale
            painter.setPen(QPen(QColor("#cadfe2"), 2))
            painter.drawLine(QPointF(18, 25), QPointF(18 + px, 25))
            painter.drawText(18, 45, f"{metres:g} m")
        painter.end()

    def pick_at(self, x, y):
        if self._image is None:
            self._render()
        for tree_id, lx, ly in reversed(self._labels):
            if abs(x - lx) < max(18, len(str(tree_id))*5) and abs(y - ly) <= 12:
                return tree_id
        if not len(self._pixels):
            return 0
        distance = ((self._pixels - (x, y))**2).sum(1)
        index = int(np.argmin(distance))
        return int(self._pixel_ids[index]) if distance[index] <= 64 else 0

    def mousePressEvent(self, event):
        self._press = self._last = event.pos()
        self._dragged = False
        self.setFocus()

    def mouseMoveEvent(self, event):
        if self._press is None or not event.buttons():
            return
        delta = event.pos() - self._last
        if (event.pos() - self._press).manhattanLength() > 4:
            self._dragged = True
        self._last = event.pos()
        if event.buttons() & (Qt.RightButton | Qt.MiddleButton) or event.modifiers() & Qt.ShiftModifier:
            self.pan += (delta.x(), delta.y())
        else:
            self.yaw += delta.x() * .4
            self.elevation = float(np.clip(self.elevation + delta.y() * .4, -85, 90))
        self.invalidate()

    def mouseReleaseEvent(self, event):
        if not self._dragged and event.button() == Qt.LeftButton:
            tree_id = self.pick_at(event.x(), event.y())
            if tree_id:
                self.tree_selected.emit(tree_id)
        self._press = None
        self._dragged = False
        self.invalidate()

    def mouseDoubleClickEvent(self, event):
        tree_id = self.pick_at(event.x(), event.y())
        if tree_id:
            self.tree_selected.emit(tree_id)
            self.isolation_requested.emit()

    def wheelEvent(self, event):
        delta = event.pixelDelta().y() or event.angleDelta().y() / 3
        self.zoom = float(np.clip(self.zoom * math.exp(delta * .003), .15, 40))
        self._interacting = True
        self._settle.start(140)
        self.invalidate()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Home, Qt.Key_F):
            self.reset_view()
        elif event.key() == Qt.Key_Escape:
            self.isolate(False)
        else:
            super().keyPressEvent(event)
