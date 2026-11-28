"""Exercise dropdown input and theme resources in source and frozen builds."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from Toolbox.appearance import theme_path
from Toolbox.soft_widgets import SoftComboBox


def check_dropdown_controls(app, screenshots=None):
    for name in ("chevron-down.svg", "chevron-up.svg", "chevron-disabled.svg"):
        icon = QIcon(str(theme_path("icons") / name))
        image = icon.pixmap(28, 28).toImage()
        if image.isNull() or not any(image.pixelColor(x, y).alpha()
                                     for x in range(image.width()) for y in range(image.height())):
            raise AssertionError(f"Dropdown icon did not render: {name}")

    window = QWidget()
    layout = QVBoxLayout(window)
    combo = SoftComboBox()
    layout.addWidget(combo)
    for text, value in (("实例颜色", "instance"), ("不可用的 RGB", "rgb"), ("离地高度", "height")):
        combo.addItem(text, value)
    combo.model().item(1).setEnabled(False)
    window.resize(260, 80)
    window.show()
    try:
        app.processEvents()
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key_Down)
        if combo.currentData() != "height":
            raise AssertionError("Keyboard navigation did not skip the disabled option")
        combo.setCurrentIndex(0)
        activated = QSignalSpy(combo.activated[int])
        combo.showPopup()
        app.processEvents()
        QTest.keyClick(combo.view(), Qt.Key_Down)
        QTest.keyClick(combo.view(), Qt.Key_Escape)
        if combo.currentData() != "instance" or len(activated) != 0:
            raise AssertionError("Esc changed the selected option")
        combo.showPopup()
        app.processEvents()
        QTest.keyClick(combo.view(), Qt.Key_Down)
        QTest.keyClick(combo.view(), Qt.Key_Return)
        if combo.currentData() != "height" or len(activated) != 1:
            raise AssertionError("Enter failed to select the option once")
        for index in range(24):
            combo.addItem(f"长选项 {index} · 坐标 / 原始颜色 / 实例标签", index)
        combo.showPopup()
        app.processEvents()
        if combo.view().verticalScrollBar().maximum() <= 0:
            raise AssertionError("Long dropdown does not scroll")
        index = combo.model().index(combo.count() - 1, 0)
        combo.view().scrollTo(index)
        app.processEvents()
        QTest.qWait(200)  # Qt ignores popup release events immediately after opening.
        QTest.mouseClick(combo.view().viewport(), Qt.LeftButton,
                         pos=combo.view().visualRect(index).center())
        if combo.currentData() != 23:
            raise AssertionError("Last option could not be selected by mouse")
        if screenshots:
            combo.grab().save(str(screenshots / "dropdown-closed.png"))
            combo.showPopup()
            app.processEvents()
            combo.view().window().grab().save(str(screenshots / "dropdown-popup.png"))
            combo.hidePopup()
        return "Three SVGs render; disabled/keyboard/Esc/Enter/mouse/data roles and scrollable list pass"
    finally:
        combo.hidePopup()
        window.close()
