"""Shared theme resources for the toolbox and standalone tools."""
from functools import lru_cache
from pathlib import Path
import sys

from PyQt5.QtWidgets import QStyleFactory


def theme_path(name):
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / name


@lru_cache(maxsize=1)
def dropdown_stylesheet():
    return theme_path("dropdown.qss").read_text(encoding="utf-8").replace(
        "@ICON_DIR@", theme_path("icons").as_posix())


def apply_application_style(app):
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    app.setStyleSheet(theme_path("style.qss").read_text(encoding="utf-8")
                      + "\n" + dropdown_stylesheet())
