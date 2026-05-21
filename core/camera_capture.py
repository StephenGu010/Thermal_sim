"""Capture thread that emits 14-bit thermal frames from UVC or mock source.

Two source kinds:
  - "uvc": prefer raw Y16 stream, fallback to 8-bit grayscale and lift to 14-bit
  - "mock": synthesise a 14-bit thermal field with sensor-like noise/drift

Signal payload:
  frame_ready(raw14_u16, original_frame_or_none, frame_id, fps)
where raw14_u16 is uint16 in range [0, 16383].
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import thermal_processing


SOURCE_UVC = "uvc"
SOURCE_MOCK = "mock"

ERR_OCCUPIED_CONFLICT = "occupied_conflict"
ERR_FIRST_FRAME_TIMEOUT = "first_frame_timeout"
ERR_FORMAT_UNSUPPORTED = "format_unsupported"
ERR_STREAM_DISCONNECTED = "stream_disconnected"

SCENE_BLOB = "blob_basic"
SCENE_PERSON = "person_scene"
SCENE_OBJECT = "object_scene"
SCENE_MIXED = "mixed_scene"
ALL_SCENES = (SCENE_BLOB, SCENE_PERSON, SCENE_OBJECT, SCENE_MIXED)


@dataclass
class CaptureConfig:
    source_kind: str = SOURCE_MOCK
    camera_index: int = 0
    width: int = 256
    height: int = 192
    target_fps: int = 25
    scene: str = SCENE_BLOB


@dataclass
class UVCProbeInfo:
    index: int
    backend: str
    openable: bool
    readable: bool
    first_frame_ms: float
    reason: str = ""


class CaptureThread(QThread):
    frame_ready = Signal(np.ndarray, object, int, float)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, config: CaptureConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._fps_window: deque[float] = deque(maxlen=30)

    def run(self) -> None:
        cfg = self._config
        if cfg.source_kind == SOURCE_UVC:
            self._run_uvc(cfg)
        else:
            self._run_mock(cfg)

    def _run_uvc(self, cfg: CaptureConfig) -> None:
        cap, backend_name = _open_uvc_capture(cfg.camera_index)
        if cap is None:
            self.error.emit(pack_capture_error(
                ERR_OCCUPIED_CONFLICT,
                f"无法打开摄像头 index={cfg.camera_index}",
            ))
            return

        _configure_capture_timing(cap, cfg)

        y16_mode = _configure_y16(cap)
        first_frame, first_frame_ms = _await_first_frame(cap, timeout_s=1.8, delay_s=0.06)
        if first_frame is None:
            self.status.emit(
                f"UVC idx={cfg.camera_index} first-frame timeout ({first_frame_ms:.0f}ms), retrying reopen"
            )
            cap.release()
            cap, backend_retry = _open_uvc_capture(cfg.camera_index)
            if cap is None:
                self.error.emit(pack_capture_error(
                    ERR_OCCUPIED_CONFLICT,
                    f"摄像头 index={cfg.camera_index} 可能被占用，请关闭 Photo Booth/相机软件后重试",
                ))
                return
            backend_name = backend_retry
            _configure_capture_timing(cap, cfg)
            y16_mode = _configure_y16(cap)
            first_frame, first_frame_ms = _await_first_frame(cap, timeout_s=1.8, delay_s=0.06)
            if first_frame is None:
                cap.release()
                if (cfg.width, cfg.height) != (256, 192):
                    self.status.emit(
                        f"UVC {cfg.width}x{cfg.height} timed out; falling back to 256x192"
                    )
                    self._run_uvc(replace(cfg, width=256, height=192))
                    return
                self.error.emit(pack_capture_error(
                    ERR_FIRST_FRAME_TIMEOUT,
                    f"摄像头 index={cfg.camera_index} 首帧超时，请稍后重试",
                ))
                return

        actual_w, actual_h = _capture_dimensions(cap, first_frame)
        if y16_mode:
            self.status.emit(
                f"UVC opened idx={cfg.camera_index} backend={backend_name} mode=Y16(raw) "
                f"req={cfg.width}x{cfg.height} actual={actual_w}x{actual_h} first={first_frame_ms:.0f}ms"
            )
        else:
            self.status.emit(
                f"UVC opened idx={cfg.camera_index} backend={backend_name} mode=8bit(fallback) "
                f"req={cfg.width}x{cfg.height} actual={actual_w}x{actual_h} first={first_frame_ms:.0f}ms"
            )

        frame_id = 0
        last = time.perf_counter()
        try:
            raw14 = _frame_to_raw14(first_frame, expect_y16=y16_mode)
            if raw14 is None and y16_mode:
                y16_mode = False
                self.status.emit("Y16 stream not available, switched to 8bit fallback")
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                raw14 = _frame_to_raw14(first_frame, expect_y16=False)
            if raw14 is None:
                self.error.emit(pack_capture_error(ERR_FORMAT_UNSUPPORTED, "无法解析摄像头帧格式"))
                return

            fps = self._tick_fps(last)
            last = time.perf_counter()
            self.frame_ready.emit(raw14, first_frame, frame_id, fps)
            frame_id += 1

            while not self.isInterruptionRequested():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error.emit(pack_capture_error(
                        ERR_STREAM_DISCONNECTED,
                        "摄像头读帧失败，可能已断开或被占用",
                    ))
                    break
                raw14 = _frame_to_raw14(frame, expect_y16=y16_mode)
                if raw14 is None:
                    # Some drivers accept Y16 FOURCC but still return BGR-like frames.
                    if y16_mode:
                        y16_mode = False
                        self.status.emit("Y16 stream not available, switched to 8bit fallback")
                        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                        raw14 = _frame_to_raw14(frame, expect_y16=False)
                    if raw14 is None:
                        self.error.emit(pack_capture_error(ERR_FORMAT_UNSUPPORTED, "无法解析摄像头帧格式"))
                        break

                fps = self._tick_fps(last)
                last = time.perf_counter()
                self.frame_ready.emit(raw14, frame, frame_id, fps)
                frame_id += 1
        finally:
            cap.release()
            self.status.emit("UVC closed")

    def _run_mock(self, cfg: CaptureConfig) -> None:
        w, h = cfg.width, cfg.height
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rng = np.random.default_rng()
        period = 1.0 / max(cfg.target_fps, 1)
        scene = cfg.scene if cfg.scene in ALL_SCENES else SCENE_BLOB
        self.status.emit(f"Mock raw14 running {w}x{h} @ {cfg.target_fps}fps scene={scene}")

        frame_id = 0
        last = time.perf_counter()
        while not self.isInterruptionRequested():
            t = frame_id / float(max(cfg.target_fps, 1))

            # Background thermal field with slow drift + horizontal/vertical bias.
            bg = (
                1850.0
                + (yy / h) * 950.0
                + (xx / w) * 380.0
                + 150.0 * np.sin(t * 0.16 + (yy / h) * 1.7)
                + 120.0 * np.cos(t * 0.12 + (xx / w) * 1.4)
            )

            if scene == SCENE_PERSON:
                signal = _scene_person(xx, yy, w, h, t)
            elif scene == SCENE_OBJECT:
                signal = _scene_object(xx, yy, w, h, t)
            elif scene == SCENE_MIXED:
                signal = _scene_person(xx, yy, w, h, t, cx_off=-w * 0.18) \
                    + _scene_object(xx, yy, w, h, t, cx_off=w * 0.25) * 0.94
            else:
                signal = _scene_blob(xx, yy, w, h, t)

            fixed_pattern = 30.0 * np.sin(xx * 0.23) + 24.0 * np.cos(yy * 0.21)
            temporal_noise = rng.normal(0.0, 78.0, (h, w)).astype(np.float32)
            sparkle = (rng.random((h, w)) < 0.004).astype(np.float32) * rng.uniform(220.0, 520.0)

            raw14_f = bg + signal + fixed_pattern + temporal_noise + sparkle
            raw14_f = cv2.GaussianBlur(raw14_f, (0, 0), sigmaX=0.55)
            raw14 = np.clip(raw14_f, 0, thermal_processing.RAW14_MAX).astype(np.uint16)

            fps = self._tick_fps(last)
            now = time.perf_counter()
            sleep = period - (now - last)
            last = now
            self.frame_ready.emit(raw14, None, frame_id, fps)
            frame_id += 1
            if sleep > 0:
                self.msleep(int(sleep * 1000))

    def _tick_fps(self, last_ts: float) -> float:
        now = time.perf_counter()
        dt = now - last_ts
        if dt > 0:
            self._fps_window.append(1.0 / dt)
        if not self._fps_window:
            return 0.0
        return float(sum(self._fps_window) / len(self._fps_window))


def _open_uvc_capture(index: int) -> tuple[Optional[cv2.VideoCapture], str]:
    for backend, name in _backend_candidates():
        try:
            cap = cv2.VideoCapture(index) if backend == cv2.CAP_ANY else cv2.VideoCapture(index, backend)
        except Exception:
            cap = None
        if cap is not None and cap.isOpened():
            return cap, name
        if cap is not None:
            cap.release()
    return None, "none"


def _backend_candidates() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(name: str, value: Optional[int]) -> None:
        if value is None:
            return
        iv = int(value)
        if iv in seen:
            return
        seen.add(iv)
        out.append((iv, name))

    if sys.platform.startswith("win"):
        add("DSHOW", getattr(cv2, "CAP_DSHOW", None))
        add("MSMF", getattr(cv2, "CAP_MSMF", None))
    elif sys.platform == "darwin":
        add("AVFOUNDATION", getattr(cv2, "CAP_AVFOUNDATION", None))
    else:
        add("V4L2", getattr(cv2, "CAP_V4L2", None))
    add("ANY", getattr(cv2, "CAP_ANY", 0))
    return out


def _configure_y16(cap: cv2.VideoCapture) -> bool:
    try:
        ok_fourcc = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("Y", "1", "6", " "))
    except Exception:
        ok_fourcc = False
    try:
        ok_rgb = cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    except Exception:
        ok_rgb = False
    return bool(ok_fourcc or ok_rgb)


def _configure_capture_timing(cap: cv2.VideoCapture, cfg: CaptureConfig) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, max(1, cfg.target_fps))
    # On Windows, the backend can otherwise queue old frames faster than Qt can
    # display them, which feels like UI jank even when processing is acceptable.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _capture_dimensions(cap: cv2.VideoCapture, frame: np.ndarray) -> tuple[int, int]:
    try:
        actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    except Exception:
        actual_w = actual_h = 0
    if actual_w <= 0 or actual_h <= 0:
        h, w = frame.shape[:2]
        return int(w), int(h)
    return actual_w, actual_h


def pack_capture_error(code: str, message: str) -> str:
    return f"{code}::{message}"


def parse_capture_error(payload: str) -> tuple[str, str]:
    code, sep, rest = payload.partition("::")
    if not sep:
        return "", payload
    return code.strip(), rest.strip()


def _await_first_frame(
    cap: cv2.VideoCapture,
    timeout_s: float = 1.8,
    delay_s: float = 0.06,
) -> tuple[Optional[np.ndarray], float]:
    start = time.perf_counter()
    deadline = start + max(0.1, timeout_s)
    while time.perf_counter() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return frame, elapsed_ms
        time.sleep(max(0.0, delay_s))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return None, elapsed_ms


def _frame_to_raw14(frame: np.ndarray, expect_y16: bool) -> Optional[np.ndarray]:
    if expect_y16:
        if frame.ndim == 2 and frame.dtype == np.uint16:
            return thermal_processing.to_raw14(frame)
        if frame.ndim == 3 and frame.dtype == np.uint8 and frame.shape[2] == 2:
            # Packed little-endian Y16 fallback shape from some UVC drivers.
            h, w = frame.shape[:2]
            packed = frame.view(np.uint16).reshape(h, w)
            return thermal_processing.to_raw14(packed)
        return None

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    if gray.dtype not in (np.uint8, np.uint16):
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return thermal_processing.to_raw14(gray)


# ----------------------------------------------------------------------------
# Mock scene synthesisers. All return float32 (H, W) heat additions.
# ----------------------------------------------------------------------------
def _scene_blob(xx: np.ndarray, yy: np.ndarray, w: int, h: int, t: float) -> np.ndarray:
    hx = w / 2 + 62.0 * np.sin(t * 0.42)
    hy = h / 2 + 28.0 * np.cos(t * 0.31)
    main_blob = 9200.0 * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * 23.0 ** 2))
    sx = 46.0 + 40.0 * np.sin(t * 1.6)
    sy = h * 0.75 + 19.0 * np.cos(t * 1.9)
    spot = 3500.0 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 5.0 ** 2))
    return main_blob + spot


def _scene_person(xx: np.ndarray, yy: np.ndarray, w: int, h: int, t: float, cx_off: float = 0.0) -> np.ndarray:
    cx = w / 2 + cx_off + 8.0 * np.sin(t * 0.5)
    base_y = h * 0.18
    head_cy = base_y + h * 0.07
    head_rx, head_ry = w * 0.052, h * 0.075
    head = 10200.0 * np.exp(-(((xx - cx) / head_rx) ** 2 + ((yy - head_cy) / head_ry) ** 2))

    neck_cy = base_y + h * 0.16
    neck_rx, neck_ry = w * 0.04, h * 0.05
    neck = 9500.0 * np.exp(-(((xx - cx) / neck_rx) ** 2 + ((yy - neck_cy) / neck_ry) ** 2))

    torso_cy = base_y + h * 0.34
    torso_rx, torso_ry = w * 0.13, h * 0.22
    torso = 11000.0 * np.exp(-(((xx - cx) / torso_rx) ** 2 + ((yy - torso_cy) / torso_ry) ** 2))

    arm_off = w * 0.14
    arm_rx, arm_ry = w * 0.035, h * 0.18
    arm_cy = base_y + h * 0.30
    arm_l = 8400.0 * np.exp(-(((xx - (cx - arm_off)) / arm_rx) ** 2 + ((yy - arm_cy) / arm_ry) ** 2))
    arm_r = 8400.0 * np.exp(-(((xx - (cx + arm_off)) / arm_rx) ** 2 + ((yy - arm_cy) / arm_ry) ** 2))

    leg_off = w * 0.05
    leg_rx, leg_ry = w * 0.05, h * 0.22
    leg_cy = base_y + h * 0.62
    leg_l = 9000.0 * np.exp(-(((xx - (cx - leg_off)) / leg_rx) ** 2 + ((yy - leg_cy) / leg_ry) ** 2))
    leg_r = 9000.0 * np.exp(-(((xx - (cx + leg_off)) / leg_rx) ** 2 + ((yy - leg_cy) / leg_ry) ** 2))
    return head + neck + torso + arm_l + arm_r + leg_l + leg_r


def _scene_object(xx: np.ndarray, yy: np.ndarray, w: int, h: int, t: float, cx_off: float = 0.0) -> np.ndarray:
    cx = w * 0.7 + cx_off + 4.0 * np.sin(t * 0.9)
    cy = h * 0.55 + 3.0 * np.cos(t * 1.3)
    box_w, box_h = w * 0.13, h * 0.16
    inside = (np.abs(xx - cx) <= box_w) & (np.abs(yy - cy) <= box_h)
    box = inside.astype(np.float32) * 8800.0
    halo = 2600.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 18.0 ** 2))
    led_cx, led_cy = cx + box_w * 0.6, cy - box_h * 0.4
    led = 10800.0 * np.exp(-((xx - led_cx) ** 2 + (yy - led_cy) ** 2) / (2 * 1.8 ** 2))
    return box + halo + led


def probe_uvc_indices(max_index: int = 12) -> list[int]:
    """Return openable camera indices in natural order."""
    return [item.index for item in probe_uvc_sources(max_index=max_index) if item.openable]


def probe_uvc_sources(max_index: int = 12) -> list[UVCProbeInfo]:
    """Probe indices and capture first-frame latency/readability hints."""
    details: list[UVCProbeInfo] = []
    fail_streak = 0
    for i in range(max_index):
        cap, backend = _open_uvc_capture(i)
        if cap is None:
            details.append(UVCProbeInfo(
                index=i,
                backend=backend,
                openable=False,
                readable=False,
                first_frame_ms=-1.0,
                reason="open_failed",
            ))
            fail_streak += 1
        else:
            frame, first_ms = _await_first_frame(cap, timeout_s=1.8, delay_s=0.06)
            readable = frame is not None
            reason = "" if readable else "first_frame_timeout"
            details.append(UVCProbeInfo(
                index=i,
                backend=backend,
                openable=True,
                readable=readable,
                first_frame_ms=first_ms,
                reason=reason,
            ))
            cap.release()
            fail_streak = 0

        # Avoid long warning storms on platforms where valid indices are dense from 0..N.
        if any(item.openable for item in details) and fail_streak >= 4:
            break
        if not any(item.openable for item in details) and i >= 6 and fail_streak >= 6:
            break
    return details
