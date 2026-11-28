"""Smooth wheel scrolling without changing the values of embedded editors."""
from PyQt5.QtCore import QAbstractAnimation, QEasingCurve, QPropertyAnimation
from PyQt5.QtWidgets import QScrollArea


class SoftScrollArea(QScrollArea):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scroll_animation = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._scroll_animation.setDuration(140)
        self._scroll_animation.setEasingCurve(QEasingCurve.OutCubic)

    def wheelEvent(self, event):
        bar = self.verticalScrollBar()
        if event.pixelDelta().y():
            self._scroll_animation.stop()
            bar.setValue(bar.value() - event.pixelDelta().y())
        elif event.angleDelta().y() and bar.maximum() > 0:
            target = (self._scroll_animation.endValue() if self._scroll_animation.state() == QAbstractAnimation.Running else bar.value())
            target = max(bar.minimum(), min(bar.maximum(), int(target - event.angleDelta().y() * .7)))
            self._scroll_animation.stop()
            self._scroll_animation.setStartValue(bar.value())
            self._scroll_animation.setEndValue(target)
            self._scroll_animation.start()
        else:
            super().wheelEvent(event)
            return
        event.accept()
