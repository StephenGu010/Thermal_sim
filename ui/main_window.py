"""Thermal-scope PC simulation main window.

The UI intentionally behaves like a scope, not an algorithm workbench: a small
tool strip for source control, a dark device body, a 16:9 scope screen, and two
virtual physical buttons.
"""
from __future__ import annotations

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
    SCENE_PERSON,
    SOURCE_MOCK,
    SOURCE_UVC,
    CaptureConfig,
    CaptureThread,
    prioritize_uvc_indices,
    probe_uvc_indices,
    realtek_camera_present,
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
PROBE_MAX_INDEX = 12
MANUAL_FALLBACK_INDICES = tuple(range(8))


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
        self._detect_cfg = hotspot_detector.DetectConfig(percentile=89.0, min_area=14)
        self._classify_cfg = tc.ClassifyConfig()
        self._last_scope: Optional[np.ndarray] = None
        self._last_raw: Optional[np.ndarray] = None
        self._last_components: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, list, tuple[int, int, int]]
        ] = None
        self._button_down_at: dict[str, float] = {}

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
        self._populate_sources()
        self.cmb_scene = QComboBox()
        for scene in ALL_SCENES:
            self.cmb_scene.addItem(SCENE_NAMES.get(scene, scene), scene)
        self.cmb_scene.setCurrentIndex(list(ALL_SCENES).index(SCENE_PERSON))

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
        indices = probe_uvc_indices(max_index=PROBE_MAX_INDEX)
        is_realtek = realtek_camera_present()
        ordered = prioritize_uvc_indices(indices) if indices else []
        for pos, idx in enumerate(ordered):
            if is_realtek and pos == 0:
                label = f"UVC {idx} (Realtek preferred)"
            else:
                label = f"UVC {idx}"
            self.cmb_source.addItem(label, (SOURCE_UVC, idx))
        if not indices:
            # Keep manual options so users can try indices even when auto-probe is blocked.
            for idx in MANUAL_FALLBACK_INDICES:
                self.cmb_source.addItem(f"UVC {idx} (manual)", (SOURCE_UVC, idx))
        elif is_realtek and self.cmb_source.count() > 1:
            # Default-select the likely Realtek camera entry.
            self.cmb_source.setCurrentIndex(1)

    def _refresh_sources(self) -> None:
        current = self.cmb_source.currentData()
        self._populate_sources()
        for i in range(self.cmb_source.count()):
            if self.cmb_source.itemData(i) == current:
                self.cmb_source.setCurrentIndex(i)
                break
        n_cams = max(0, self.cmb_source.count() - 1)
        self._notify(f"Camera list refreshed: {n_cams} UVC options")

    def _start_capture(self) -> None:
        if self._capture is not None:
            return
        source = self.cmb_source.currentData()
        scene = self.cmb_scene.currentData()
        if source == SOURCE_MOCK:
            cfg = CaptureConfig(source_kind=SOURCE_MOCK, width=256, height=192, scene=scene)
        else:
            kind, idx = source
            cfg = CaptureConfig(source_kind=kind, camera_index=idx, width=256, height=192, scene=scene)

        self._capture = CaptureThread(cfg)
        self._capture.frame_ready.connect(self._on_frame)
        self._capture.status.connect(self._notify)
        self._capture.error.connect(self._on_capture_error)
        self._capture.finished.connect(self._on_capture_finished)
        self._capture.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._notify(f"Capture started: {self.cmb_source.currentText()}")

    def _stop_capture(self) -> None:
        if self._capture is None:
            return
        self._capture.requestInterruption()
        self._capture.wait(2000)

    def _on_capture_finished(self) -> None:
        self._capture = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._recorder.is_recording:
            self._recorder.stop()
        self._notify("Capture stopped")

    def _on_capture_error(self, msg: str) -> None:
        self._notify(f"Error: {msg}")
        QMessageBox.warning(self, "Capture error", msg)

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
    def _on_frame(self, raw14: np.ndarray, _bgr_or_none, frame_id: int, fps: float) -> None:
        if self._state.frozen and self._last_scope is not None:
            self.video.set_frame(self._last_scope)
            return

        raw14_u16 = thermal_processing.to_raw14(raw14)
        self._last_raw = raw14_u16.copy()
        self._process_raw_to_components(raw14_u16)

        self._state.frame_id = frame_id
        self._state.fps = fps
        self._rerender_last()

    def _process_raw_to_components(self, raw14_u16: np.ndarray) -> None:
        enhance_cfg = scope_enhancement.ScopeEnhanceConfig(level=self._state.enhancement_level)
        enhanced = scope_enhancement.enhance_scope_whitehot(raw14_u16, enhance_cfg)
        outline_cfg = outline_processing.OutlineConfig(level=self._state.enhancement_level)
        outline = outline_processing.render_outline(raw14_u16, outline_cfg)

        hx, hy, hv = hotspot_detector.find_hotspot(enhanced)
        mask, _regions = hotspot_detector.find_candidates(enhanced, self._detect_cfg)
        contours = contour_overlay.extract_contours(mask, min_area=self._detect_cfg.min_area)
        classifications = tc.classify_all(contours, enhanced, mask, self._classify_cfg)

        self._last_components = (enhanced, outline, mask, classifications, (hx, hy, hv))

    def _rerender_last(self, force_reenhance: bool = False) -> None:
        if force_reenhance and self._last_raw is not None:
            self._process_raw_to_components(self._last_raw)
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
        mode = "OUTLINE" if self._state.outline_enabled else (
            "BHOT" if self._state.palette == scope_renderer.PALETTE_BLACKHOT else "WHOT"
        )
        menu = f" MENU {self._state.menu_index + 1}" if self._state.menu_open else ""
        hold = " HOLD" if self._state.frozen else ""
        outline = "" if self._state.outline_enabled else " NO-OUT"
        self.lbl_state.setText(
            f"{mode}  ENH {self._state.enhancement_level}  ZOOM {self._state.zoom}x{outline}{hold}{menu}"
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

    def closeEvent(self, event) -> None:
        if self._capture is not None:
            self._capture.requestInterruption()
            self._capture.wait(2000)
        if self._recorder.is_recording:
            self._recorder.stop()
        super().closeEvent(event)
