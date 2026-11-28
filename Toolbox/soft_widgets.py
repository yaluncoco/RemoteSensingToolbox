"""Shared, softly styled controls with standard Qt input and model semantics."""
from PyQt5.QtCore import QAbstractAnimation, QEasingCurve, QPropertyAnimation, Qt
from PyQt5.QtWidgets import QComboBox, QFrame, QListView, QScrollArea, QStyledItemDelegate

from Toolbox.appearance import dropdown_stylesheet


class SoftComboBox(QComboBox):
    """Consistent dropdowns, including tools launched without the dashboard."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(dropdown_stylesheet())
        self.setCursor(Qt.PointingHandCursor)
        self.setMaxVisibleItems(10)
        view = QListView(self)
        view.setObjectName("comboPopup")
        view.setMouseTracking(True)
        view.setUniformItemSizes(True)
        view.setVerticalScrollMode(QListView.ScrollPerPixel)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setTextElideMode(Qt.ElideRight)
        self.setView(view)
        # The combo menu delegate paints a native menu, ignoring row QSS.
        self.setItemDelegate(QStyledItemDelegate(view))
        popup = view.window()
        popup.setObjectName("comboPopupFrame")
        popup.setFrameShape(QFrame.NoFrame)
        popup.setAttribute(Qt.WA_TranslucentBackground)

    def showPopup(self):
        # Compact toolbar controls can still have long, descriptive options.
        screen = self.screen()
        available = screen.availableGeometry().width() - 24 if screen else 1000
        text_width = max((self.fontMetrics().horizontalAdvance(self.itemText(i))
                          for i in range(self.count())), default=0)
        self.view().setMinimumWidth(min(available, max(self.width(), text_width + 52)))
        super().showPopup()


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
