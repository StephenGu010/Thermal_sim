"""Thermal-scope PC simulation main window.

The UI intentionally behaves like a scope, not an algorithm workbench: a small
tool strip for source control, a dark device body, a 16:9 scope screen, and two
virtual physical buttons.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core import contour_overlay, hotspot_detector, outline_processing, scope_enhancement, scope_renderer
from core import target_classifier as tc
from core import thermal_processing
from core.camera_capture import (
    ALL_SCENES,
    ERR_FIRST_FRAME_TIMEOUT,
    ERR_FORMAT_UNSUPPORTED,
    ERR_OCCUPIED_CONFLICT,
    ERR_STREAM_DISCONNECTED,
    SCENE_PERSON,
    SOURCE_MOCK,
    SOURCE_UVC,
    CaptureConfig,
    CaptureThread,
    parse_capture_error,
    probe_uvc_sources,
)
from core.frame_recorder import Recorder, save_screenshot
from ui.video_widget import ASPECT_NATIVE, VideoWidget


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "output" / "screenshots"
RECORDING_DIR = PROJECT_ROOT / "output" / "recordings"


SCENE_NAMES = {
    "blob_basic": "Mock Blob",
    "person_scene": "Mock Person",
    "object_scene": "Mock Object",
    "mixed_scene": "Mock Mixed",
}
PROFILE_NAMES = {
    outline_processing.PROFILE_THERMAL: "Thermal Tiny1-C",
    outline_processing.PROFILE_VISIBLE: "Visible Demo",
}
RESOLUTION_PRESETS = {
    "native": {
        "label": "Native 256x192",
        "capture": (256, 192),
        "processing_scale": 1.0,
        "hud": "256x192",
    },
    "hd": {
        "label": "HD 640x480",
        "capture": (640, 480),
        "processing_scale": 1.0,
        "hud": "640x480",
    },
    "p720": {
        "label": "720p 1280x720",
        "capture": (1280, 720),
        "processing_scale": 1.0,
        "hud": "1280x720",
    },
    "tiny_x2": {
        "label": "Tiny x2 Detail",
        "capture": (256, 192),
        "processing_scale": 2.0,
        "hud": "256x192",
    },
}
DETAIL_PRESETS = {
    "clean": {
        "label": "Clean",
        "processing_scale": 1.0,
        "high_delta": 2.5,
        "density": 0.72,
        "bridge": 1,
        "hardness": 0.96,
    },
    "balanced": {
        "label": "Balanced",
        "processing_scale": 1.25,
        "high_delta": 0.0,
        "density": 1.0,
        "bridge": 2,
        "hardness": 0.91,
    },
    "fine": {
        "label": "Fine",
        "processing_scale": 1.5,
        "high_delta": -1.8,
        "density": 1.18,
        "bridge": 2,
        "hardness": 0.88,
    },
    "ultra": {
        "label": "Ultra",
        "processing_scale": 2.0,
        "high_delta": -3.0,
        "density": 1.35,
        "bridge": 3,
        "hardness": 0.84,
    },
}
SMOOTH_PRESETS = {
    "off": {
        "label": "Off",
        "gaussian": 1,
        "bilateral": 1,
        "temporal": 0.95,
        "glow": "off",
    },
    "low": {
        "label": "Low",
        "gaussian": 3,
        "bilateral": 3,
        "temporal": 0.74,
        "glow": "low",
    },
    "mid": {
        "label": "Mid",
        "gaussian": 3,
        "bilateral": 5,
        "temporal": 0.58,
        "glow": "low",
    },
    "high": {
        "label": "High",
        "gaussian": 5,
        "bilateral": 7,
        "temporal": 0.44,
        "glow": "mid",
    },
}
PROBE_MAX_INDEX = 12
SOURCE_COMBO_WIDTH = 210
SCENE_COMBO_WIDTH = 150
PROFILE_COMBO_WIDTH = 165
RESOLUTION_COMBO_WIDTH = 160
DETAIL_COMBO_WIDTH = 105
SMOOTH_COMBO_WIDTH = 90
LAST_SOURCE_FILE = PROJECT_ROOT / "output" / "last_source.json"


class MainWindow(QMainWindow):
    log_line = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tiny1-C Thermal Scope Simulator")
        self.resize(1280, 760)
        self.setMinimumSize(980, 620)

        self._capture: Optional[CaptureThread] = None
        self._recorder = Recorder()
        self._state = scope_renderer.ScopeRenderState()
        self._outline_processor = outline_processing.OutlineProcessor()
        self._detect_cfg = hotspot_detector.DetectConfig(percentile=89.0, min_area=14)
        self._classify_cfg = tc.ClassifyConfig()
        self._last_scope: Optional[np.ndarray] = None
        self._last_raw: Optional[np.ndarray] = None
        self._last_visible_frame: Optional[np.ndarray] = None
        self._last_components: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, list, Optional[tuple[int, int, int]]]
        ] = None
        self._button_down_at: dict[str, float] = {}
        self._active_source_data = None
        self._saved_source_this_run = False
        self._source_probe_cache = []
        self._last_process_ts = 0.0
        self._last_process_ms = 0.0

        self._build_ui()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Select source and press Start.")
        self.log_line.connect(lambda msg: self.statusBar().showMessage(msg, 6000))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        root_layout.addLayout(self._build_toolbar())

        body = QFrame()
        body.setObjectName("scopeBody")
        body.setStyleSheet("""
            QFrame#scopeBody {
                background: #050607;
                border: 2px solid #20272b;
                border-radius: 18px;
            }
            QPushButton#scopeButton {
                background: #141a1d;
                color: #d5ffff;
                border: 2px solid #39464b;
                border-radius: 28px;
                font-size: 16px;
                font-weight: 700;
                padding: 12px;
            }
            QPushButton#scopeButton:pressed {
                background: #d5ffff;
                color: #061012;
            }
        """)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(24, 24, 24, 24)
        body_layout.setSpacing(20)

        self.btn_left = self._make_scope_button("MENU\nNEXT", "left")
        self.btn_right = self._make_scope_button("OK\nADJUST", "right")
        self.video = VideoWidget()
        self.video.set_aspect(ASPECT_NATIVE)

        body_layout.addWidget(self.btn_left, 0)
        body_layout.addWidget(self.video, 1)
        body_layout.addWidget(self.btn_right, 0)
        root_layout.addWidget(body, 1)

        hint = QLabel("Keys: A/Left=MENU  D/Right=ADJUST  Shift+A=Back  Shift+D/S=Screenshot  Space=Freeze")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#8da4a8; font-size:12px;")
        root_layout.addWidget(hint)

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.cmb_source = QComboBox()
        self.cmb_source.setFixedWidth(SOURCE_COMBO_WIDTH)
        self._populate_sources()
        self.cmb_scene = QComboBox()
        self.cmb_scene.setFixedWidth(SCENE_COMBO_WIDTH)
        for scene in ALL_SCENES:
            self.cmb_scene.addItem(SCENE_NAMES.get(scene, scene), scene)
        self.cmb_scene.setCurrentIndex(list(ALL_SCENES).index(SCENE_PERSON))
        self.cmb_profile = QComboBox()
        self.cmb_profile.setFixedWidth(PROFILE_COMBO_WIDTH)
        for profile, label in PROFILE_NAMES.items():
            self.cmb_profile.addItem(label, profile)
        self.cmb_profile.setCurrentIndex(0)
        self.cmb_profile.currentIndexChanged.connect(self._on_profile_changed)
        self.cmb_resolution = QComboBox()
        self.cmb_resolution.setFixedWidth(RESOLUTION_COMBO_WIDTH)
        for key, preset in RESOLUTION_PRESETS.items():
            self.cmb_resolution.addItem(preset["label"], key)
        self.cmb_resolution.setCurrentIndex(3)
        self.cmb_resolution.currentIndexChanged.connect(self._on_resolution_changed)
        self.cmb_detail = QComboBox()
        self.cmb_detail.setFixedWidth(DETAIL_COMBO_WIDTH)
        for key, preset in DETAIL_PRESETS.items():
            self.cmb_detail.addItem(preset["label"], key)
        self.cmb_detail.setCurrentIndex(2)
        self.cmb_detail.currentIndexChanged.connect(self._on_tuning_changed)
        self.cmb_smooth = QComboBox()
        self.cmb_smooth.setFixedWidth(SMOOTH_COMBO_WIDTH)
        for key, preset in SMOOTH_PRESETS.items():
            self.cmb_smooth.addItem(preset["label"], key)
        self.cmb_smooth.setCurrentIndex(1)
        self.cmb_smooth.currentIndexChanged.connect(self._on_tuning_changed)

        self.btn_refresh = QToolButton()
        self.btn_refresh.setText("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_sources)
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._start_capture)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_capture)
        self.btn_stop.setEnabled(False)
        self.btn_shot = QPushButton("Screenshot")
        self.btn_shot.clicked.connect(self._take_screenshot)

        self.lbl_state = QLabel("WHOT  ENH 3  ZOOM 1x")
        self.lbl_state.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_state.setStyleSheet("color:#d5ffff; font-weight:700;")

        for label, widget in (
            ("Source", self.cmb_source),
            ("Scene", self.cmb_scene),
            ("Profile", self.cmb_profile),
            ("Resolution", self.cmb_resolution),
            ("Detail", self.cmb_detail),
            ("Smooth", self.cmb_smooth),
        ):
            lab = QLabel(label)
            lab.setStyleSheet("color:#9aa;")
            bar.addWidget(lab)
            bar.addWidget(widget)
        bar.addWidget(self.btn_refresh)
        bar.addSpacing(12)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_shot)
        bar.addStretch(1)
        bar.addWidget(self.lbl_state)
        return bar

    def _make_scope_button(self, text: str, name: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("scopeButton")
        btn.setFixedWidth(120)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        btn.pressed.connect(lambda n=name: self._button_pressed(n))
        btn.released.connect(lambda n=name: self._button_released(n))
        return btn

    # ------------------------------------------------------------------
    # Capture lifecycle
    # ------------------------------------------------------------------
    def _populate_sources(self) -> None:
        self.cmb_source.clear()
        self.cmb_source.addItem("Mock", SOURCE_MOCK)
        details = probe_uvc_sources(max_index=PROBE_MAX_INDEX)
        self._source_probe_cache = details
        indices = [item.index for item in details if item.openable]
        for idx in indices:
            self.cmb_source.addItem(f"UVC {idx}", (SOURCE_UVC, idx))
        if not indices:
            for idx in range(PROBE_MAX_INDEX + 1):
                self.cmb_source.addItem(f"UVC {idx} (manual)", (SOURCE_UVC, idx))

        # If we have a remembered source and it still exists, restore it.
        remembered = self._load_last_uvc_index()
        restored = False
        if remembered is not None:
            for i in range(self.cmb_source.count()):
                if self.cmb_source.itemData(i) == (SOURCE_UVC, remembered):
                    self.cmb_source.setCurrentIndex(i)
                    restored = True
                    break
        if not restored and self.cmb_source.count() > 1:
            # Fallback to the first UVC source (auto-detected or manual).
            self.cmb_source.setCurrentIndex(1)

    def _refresh_sources(self) -> None:
        current = self.cmb_source.currentData()
        self._populate_sources()
        for i in range(self.cmb_source.count()):
            if self.cmb_source.itemData(i) == current:
                self.cmb_source.setCurrentIndex(i)
                break
        openable = [item for item in self._source_probe_cache if item.openable]
        readable = [item for item in self._source_probe_cache if item.readable]
        if openable:
            timing = ", ".join(
                f"{item.index}:{item.first_frame_ms:.0f}ms{'*' if item.readable else ''}"
                for item in openable
            )
            self._notify(f"Camera list refreshed: openable={len(openable)} readable={len(readable)} [{timing}]")
        else:
            self._notify("Camera list refreshed: no openable UVC, manual indices enabled (0-12)")

    def _start_capture(self) -> None:
        if self._capture is not None:
            return
        source = self.cmb_source.currentData()
        if source is None:
            self._notify("Please select a valid source")
            return
        scene = self.cmb_scene.currentData()
        width, height = self._current_capture_size()
        if source == SOURCE_MOCK:
            cfg = CaptureConfig(source_kind=SOURCE_MOCK, width=width, height=height, scene=scene)
        elif self._is_uvc_item(source):
            _kind, idx = source
            cfg = CaptureConfig(source_kind=SOURCE_UVC, camera_index=int(idx), width=width, height=height, scene=scene)
        else:
            self._notify("Unsupported source entry")
            return

        self._capture = CaptureThread(cfg)
        self._capture.frame_ready.connect(self._on_frame)
        self._capture.status.connect(self._notify)
        self._capture.error.connect(self._on_capture_error)
        self._capture.finished.connect(self._on_capture_finished)
        self._outline_processor.reset()
        self._capture.start()
        self._active_source_data = source
        self._saved_source_this_run = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._notify(
            f"Capture started: {self.cmb_source.currentText()} / {self._profile_text()} / "
            f"{self._resolution_text()}"
        )

    def _stop_capture(self) -> None:
        if self._capture is None:
            return
        self._capture.requestInterruption()
        self._capture.wait(2000)
        if self._capture is not None and not self._capture.isRunning():
            self._on_capture_finished()

    def _on_capture_finished(self) -> None:
        self._capture = None
        self._active_source_data = None
        self._saved_source_this_run = False
        self._outline_processor.reset()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._recorder.is_recording:
            self._recorder.stop()
        self._notify("Capture stopped")

    def _on_capture_error(self, msg: str) -> None:
        code, detail = parse_capture_error(msg)
        if code == ERR_OCCUPIED_CONFLICT:
            user_msg = f"{detail}\n\n可能原因：占用冲突。\n建议：关闭 Photo Booth/微信/浏览器相机页后重试。"
        elif code == ERR_FIRST_FRAME_TIMEOUT:
            user_msg = f"{detail}\n\n可能原因：设备初始化较慢或短时抖动。\n建议：等待 2-3 秒后再 Start。"
        elif code == ERR_FORMAT_UNSUPPORTED:
            user_msg = f"{detail}\n\n可能原因：当前输出格式不受支持。\n建议：切换其他 UVC 索引或关闭 Y16-only 设备占用。"
        elif code == ERR_STREAM_DISCONNECTED:
            user_msg = f"{detail}\n\n可能原因：线缆/供电/占用变化。\n建议：重新插拔并 Refresh 后重试。"
        else:
            user_msg = detail or msg
        code_tag = f"[{code}] " if code else ""
        self._notify(f"Error: {code_tag}{detail or msg}")
        QMessageBox.warning(self, "Capture error", user_msg)

    # ------------------------------------------------------------------
    # Scope controls
    # ------------------------------------------------------------------
    def _button_pressed(self, name: str) -> None:
        self._button_down_at[name] = time.monotonic()

    def _button_released(self, name: str) -> None:
        dt = time.monotonic() - self._button_down_at.pop(name, time.monotonic())
        is_long = dt >= 0.6
        if name == "left":
            self._left_long() if is_long else self._left_short()
        else:
            self._right_long() if is_long else self._right_short()

    def _left_short(self) -> None:
        if not self._state.menu_open:
            self._state.menu_open = True
            self._state.menu_index = 0
        else:
            self._state.menu_index = (self._state.menu_index + 1) % 5
        self._rerender_last()

    def _right_short(self) -> None:
        if not self._state.menu_open:
            self._cycle_zoom()
        else:
            idx = self._state.menu_index
            if idx == 0:
                self._state.enhancement_level = 1 + (self._state.enhancement_level % 5)
            elif idx == 1:
                self._cycle_zoom()
            elif idx == 2:
                self._state.palette = (
                    scope_renderer.PALETTE_BLACKHOT
                    if self._state.palette == scope_renderer.PALETTE_WHITEHOT
                    else scope_renderer.PALETTE_WHITEHOT
                )
            elif idx == 3:
                self._state.outline_enabled = not self._state.outline_enabled
            elif idx == 4:
                self._state.frozen = not self._state.frozen
        self._rerender_last(force_reenhance=True)

    def _left_long(self) -> None:
        self._state.menu_open = False
        self._rerender_last()

    def _right_long(self) -> None:
        self._take_screenshot()

    def _cycle_zoom(self) -> None:
        zooms = [1, 2, 4, 8]
        self._state.zoom = zooms[(zooms.index(self._state.zoom) + 1) % len(zooms)]

    def _on_profile_changed(self) -> None:
        profile = self._current_profile()
        self._state.processing_profile = profile
        self._outline_processor.reset()
        if profile == outline_processing.PROFILE_VISIBLE:
            self._notify("Profile: Visible Demo (ordinary-camera edge demo, not thermal)")
        else:
            self._notify("Profile: Thermal Tiny1-C (raw/Y16 preferred)")
        self._rerender_last(force_reenhance=True)

    def _on_resolution_changed(self) -> None:
        self._outline_processor.reset()
        if self._capture is not None:
            self._notify(f"Resolution changed to {self._resolution_text()}, restarting capture")
            self._stop_capture()
            self._start_capture()
            return
        self._rerender_last(force_reenhance=True)

    def _on_tuning_changed(self) -> None:
        self._outline_processor.reset()
        self._notify(
            f"Outline tuning: {self._resolution_text()} / {self._detail_text()} / {self._smooth_text()}"
        )
        self._rerender_last(force_reenhance=True)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        key = event.key()
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if key in (Qt.Key_A, Qt.Key_Left):
            self._left_long() if shift else self._left_short()
        elif key in (Qt.Key_D, Qt.Key_Right):
            self._right_long() if shift else self._right_short()
        elif key == Qt.Key_S:
            self._take_screenshot()
        elif key == Qt.Key_Space:
            self._state.frozen = not self._state.frozen
            self._rerender_last()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _on_frame(self, raw14: np.ndarray, bgr_or_none, frame_id: int, fps: float) -> None:
        if self._state.frozen and self._last_scope is not None:
            self.video.set_frame(self._last_scope)
            return

        now = time.perf_counter()
        target_fps = 20.0 if self._state.outline_enabled else 30.0
        if self._last_scope is not None and now - self._last_process_ts < 1.0 / target_fps:
            return
        self._last_process_ts = now

        if not self._saved_source_this_run and self._is_uvc_item(self._active_source_data):
            _kind, idx = self._active_source_data
            self._save_last_uvc_index(int(idx))
            self._saved_source_this_run = True

        t0 = time.perf_counter()
        raw14_u16 = thermal_processing.to_raw14(raw14)
        self._last_raw = raw14_u16.copy()
        self._last_visible_frame = bgr_or_none.copy() if isinstance(bgr_or_none, np.ndarray) else None
        self._process_raw_to_components(raw14_u16, self._last_visible_frame, update_temporal=True)

        self._state.frame_id = frame_id
        self._state.fps = fps
        self._rerender_last()
        self._last_process_ms = (time.perf_counter() - t0) * 1000.0

    def _process_raw_to_components(
        self,
        raw14_u16: np.ndarray,
        visible_frame: Optional[np.ndarray] = None,
        update_temporal: bool = True,
    ) -> None:
        profile = self._current_profile()
        self._state.processing_profile = profile
        if profile == outline_processing.PROFILE_VISIBLE and self._state.outline_enabled:
            enhanced = np.zeros(raw14_u16.shape[:2], dtype=np.uint8)
        else:
            enhance_cfg = scope_enhancement.ScopeEnhanceConfig(level=self._state.enhancement_level)
            enhanced = scope_enhancement.enhance_scope_whitehot(raw14_u16, enhance_cfg)
        outline_cfg = self._build_outline_config(profile)
        outline = self._outline_processor.render(
            raw14_u16,
            outline_cfg,
            visible_frame=visible_frame,
            update_temporal=update_temporal,
        )
        self._state.outline_scale_x = outline.shape[1] / max(enhanced.shape[1], 1)
        self._state.outline_scale_y = outline.shape[0] / max(enhanced.shape[0], 1)
        self._state.outline_processing_scale = max(self._state.outline_scale_x, self._state.outline_scale_y)
        self._state.resolution_label = self._resolution_hud_text()

        if profile == outline_processing.PROFILE_VISIBLE:
            mask = np.zeros_like(enhanced, dtype=np.uint8)
            classifications = []
            hotspot = None
        else:
            hx, hy, hv = hotspot_detector.find_hotspot(enhanced)
            mask, _regions = hotspot_detector.find_candidates(enhanced, self._detect_cfg)
            contours = contour_overlay.extract_contours(mask, min_area=self._detect_cfg.min_area)
            classifications = tc.classify_all(contours, enhanced, mask, self._classify_cfg)
            hotspot = (hx, hy, hv)

        self._last_components = (enhanced, outline, mask, classifications, hotspot)

    def _rerender_last(self, force_reenhance: bool = False) -> None:
        if force_reenhance and self._last_raw is not None:
            self._process_raw_to_components(self._last_raw, self._last_visible_frame, update_temporal=False)
        if self._last_components is None:
            self._update_state_label()
            return
        enhanced, outline, mask, classifications, hotspot = self._last_components
        scope = scope_renderer.render_scope_frame(enhanced, outline, mask, classifications, hotspot, self._state)
        self._last_scope = scope
        self.video.set_frame(scope)
        if self._recorder.is_recording:
            self._recorder.write(scope)
        self._update_state_label()

    def _update_state_label(self) -> None:
        self._state.processing_profile = self._current_profile()
        self._state.resolution_label = self._resolution_hud_text()
        profile = "VISIBLE DEMO" if self._state.processing_profile == outline_processing.PROFILE_VISIBLE else "THERMAL"
        mode = "OUTLINE" if self._state.outline_enabled else (
            "BHOT" if self._state.palette == scope_renderer.PALETTE_BLACKHOT else "WHOT"
        )
        menu = f" MENU {self._state.menu_index + 1}" if self._state.menu_open else ""
        hold = " HOLD" if self._state.frozen else ""
        outline = "" if self._state.outline_enabled else " NO-OUT"
        perf = f"  P{self._last_process_ms:.0f}ms" if self._last_process_ms > 0 else ""
        self.lbl_state.setText(
            f"{profile}  {mode}  {self._resolution_hud_text()}  "
            f"{self._detail_text()}  {self._smooth_text()}  ENH {self._state.enhancement_level}  "
            f"ZOOM {self._state.zoom}x{outline}{hold}{menu}{perf}"
        )

    # ------------------------------------------------------------------
    # Screenshot / recording
    # ------------------------------------------------------------------
    def _take_screenshot(self) -> None:
        if self._last_scope is None:
            self._notify("No scope frame to save")
            return
        try:
            path = save_screenshot(self._last_scope, SCREENSHOT_DIR)
        except Exception as exc:
            self._notify(f"Screenshot failed: {exc}")
            return
        self._notify(f"Screenshot saved: {path}")

    # Kept for Recorder compatibility; no visible record button in scope mode.
    def _toggle_recording(self) -> None:
        if self._last_scope is None:
            self._notify("No frame to record")
            return
        if self._recorder.is_recording:
            path = self._recorder.stop()
            self._notify(f"Recording stopped: {path}")
            return
        h, w = self._last_scope.shape[:2]
        path = self._recorder.start(RECORDING_DIR, (w, h))
        self._notify(f"Recording started: {path}")

    def _notify(self, msg: str) -> None:
        self.log_line.emit(msg)

    @staticmethod
    def _is_uvc_item(data) -> bool:
        return isinstance(data, tuple) and len(data) == 2 and data[0] == SOURCE_UVC

    def _current_profile(self) -> str:
        if not hasattr(self, "cmb_profile"):
            return outline_processing.PROFILE_THERMAL
        profile = self.cmb_profile.currentData()
        if profile in outline_processing.ALL_PROFILES:
            return profile
        return outline_processing.PROFILE_THERMAL

    def _profile_text(self) -> str:
        return PROFILE_NAMES.get(self._current_profile(), "Thermal Tiny1-C")

    def _current_resolution_key(self) -> str:
        if not hasattr(self, "cmb_resolution"):
            return "tiny_x2"
        key = self.cmb_resolution.currentData()
        return key if key in RESOLUTION_PRESETS else "tiny_x2"

    def _current_detail_key(self) -> str:
        if not hasattr(self, "cmb_detail"):
            return "fine"
        key = self.cmb_detail.currentData()
        return key if key in DETAIL_PRESETS else "fine"

    def _current_smooth_key(self) -> str:
        if not hasattr(self, "cmb_smooth"):
            return "low"
        key = self.cmb_smooth.currentData()
        return key if key in SMOOTH_PRESETS else "low"

    def _current_capture_size(self) -> tuple[int, int]:
        preset = RESOLUTION_PRESETS[self._current_resolution_key()]
        width, height = preset["capture"]
        return int(width), int(height)

    def _current_processing_scale(self) -> float:
        res = RESOLUTION_PRESETS[self._current_resolution_key()]
        detail = DETAIL_PRESETS[self._current_detail_key()]
        return float(max(res["processing_scale"], detail["processing_scale"]))

    def _resolution_text(self) -> str:
        return RESOLUTION_PRESETS[self._current_resolution_key()]["label"]

    def _resolution_hud_text(self) -> str:
        return RESOLUTION_PRESETS[self._current_resolution_key()]["hud"]

    def _detail_text(self) -> str:
        return DETAIL_PRESETS[self._current_detail_key()]["label"]

    def _smooth_text(self) -> str:
        return SMOOTH_PRESETS[self._current_smooth_key()]["label"]

    def _build_outline_config(self, profile: str) -> outline_processing.OutlineConfig:
        detail_key = self._current_detail_key()
        smooth_key = self._current_smooth_key()
        detail = DETAIL_PRESETS[detail_key]
        smooth = SMOOTH_PRESETS[smooth_key]

        thermal_density = 0.036 * detail["density"]
        visible_density = 0.020 * detail["density"]
        high_percentile = 93.0 + detail["high_delta"]
        if profile == outline_processing.PROFILE_VISIBLE:
            high_percentile += 1.2

        return outline_processing.OutlineConfig(
            level=self._state.enhancement_level,
            profile=profile,
            processing_scale=self._current_processing_scale(),
            detail_mode=detail_key,
            smooth_mode=smooth_key,
            gaussian_ksize=int(smooth["gaussian"]),
            high_percentile=float(high_percentile),
            low_ratio=0.48 if detail_key == "clean" else 0.42,
            bridge_strength_ratio=0.68 if detail_key == "clean" else 0.54,
            bridge_max_gap=int(detail["bridge"]),
            glow_gain=0.04,
            glow_sigma=0.55,
            edge_hardness=float(detail["hardness"]),
            glow_mode=str(smooth["glow"]),
            temporal_alpha=float(smooth["temporal"]),
            thermal_edge_density=float(thermal_density),
            visible_edge_density=float(visible_density),
            visible_min_component_area=12 if detail_key in ("clean", "balanced") else 7,
            visible_bilateral_d=int(smooth["bilateral"]),
        )

    def _load_last_uvc_index(self) -> Optional[int]:
        try:
            if not LAST_SOURCE_FILE.exists():
                return None
            payload = json.loads(LAST_SOURCE_FILE.read_text(encoding="utf-8"))
            if payload.get("source_kind") == SOURCE_UVC:
                idx = payload.get("camera_index")
                if isinstance(idx, int):
                    return idx
        except Exception:
            return None
        return None

    def _save_last_uvc_index(self, index: int) -> None:
        try:
            LAST_SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            if LAST_SOURCE_FILE.exists():
                payload = json.loads(LAST_SOURCE_FILE.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            payload["source_kind"] = SOURCE_UVC
            payload["camera_index"] = int(index)
            LAST_SOURCE_FILE.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        except Exception:
            # Persistence failure should never block capture.
            pass

    def closeEvent(self, event) -> None:
        if self._capture is not None:
            self._capture.requestInterruption()
            self._capture.wait(2000)
        if self._recorder.is_recording:
            self._recorder.stop()
        super().closeEvent(event)
