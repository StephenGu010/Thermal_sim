"""Display widget for the pre-rendered 16:9 scope screen."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget


ASPECT_NATIVE = "scope_native"
ASPECT_AMOLED = "amoled_294x126"


class VideoWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(0, 0, 0))
        self.setPalette(pal)
        self._bgr: Optional[np.ndarray] = None
        self._aspect = ASPECT_NATIVE

    def set_frame(self, bgr: np.ndarray) -> None:
        self._bgr = self._apply_aspect(bgr)
        self.update()

    def set_aspect(self, mode: str) -> None:
        self._aspect = mode
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._bgr is None:
            painter.setPen(QColor(170, 190, 190))
            painter.drawText(self.rect(), Qt.AlignCenter, "(no scope frame)\nPress Start")
            return
        h, w = self._bgr.shape[:2]
        qimg = QImage(self._bgr.data, w, h, w * 3, QImage.Format_BGR888).copy()
        painter.drawImage(self._fit_rect(w, h), qimg)

    def _fit_rect(self, src_w: int, src_h: int) -> QRect:
        avail = self.rect()
        src_ratio = src_w / src_h
        avail_ratio = avail.width() / max(avail.height(), 1)
        if src_ratio > avail_ratio:
            w = avail.width()
            h = int(w / src_ratio)
        else:
            h = avail.height()
            w = int(h * src_ratio)
        x = avail.x() + (avail.width() - w) // 2
        y = avail.y() + (avail.height() - h) // 2
        return QRect(x, y, w, h)

    def _apply_aspect(self, bgr: np.ndarray) -> np.ndarray:
        if self._aspect != ASPECT_AMOLED:
            return bgr
        target_w, target_h = 294, 126
        target_ratio = target_w / target_h
        h, w = bgr.shape[:2]
        cur_ratio = w / h
        if cur_ratio > target_ratio:
            new_w = int(h * target_ratio)
            x0 = (w - new_w) // 2
            return bgr[:, x0:x0 + new_w]
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        return bgr[y0:y0 + new_h, :]
