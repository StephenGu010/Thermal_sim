"""Screenshot and video recording for the colourised BGR preview frames.

Uses ``cv2.imencode`` + ``Path.write_bytes`` for screenshots so non-ASCII
paths (common on Chinese Windows installs) work correctly — ``cv2.imwrite``
silently fails on those paths.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_screenshot(bgr: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"thermal_{_timestamp()}.png"
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed for screenshot")
    path.write_bytes(buf.tobytes())
    return path


class Recorder:
    """Wrap cv2.VideoWriter so the UI can start/stop without leaking handles."""

    def __init__(self) -> None:
        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None
        self._size: Optional[tuple[int, int]] = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def start(self, out_dir: Path, size: tuple[int, int], fps: float = 25.0) -> Path:
        if self._writer is not None:
            raise RuntimeError("already recording")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / f"thermal_{_timestamp()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, size)
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError("VideoWriter failed to open (codec/path?)")
        self._size = size
        return self._path

    def write(self, bgr: np.ndarray) -> None:
        if self._writer is None or self._size is None:
            return
        if (bgr.shape[1], bgr.shape[0]) != self._size:
            bgr = cv2.resize(bgr, self._size)
        self._writer.write(bgr)

    def stop(self) -> Optional[Path]:
        if self._writer is None:
            return None
        self._writer.release()
        path = self._path
        self._writer = None
        self._path = None
        self._size = None
        return path
