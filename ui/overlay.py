import math
import os
import subprocess
import time
from pathlib import Path

import PyQt5
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QRadialGradient, QCursor
from PyQt5.QtWidgets import QApplication, QWidget


def _fix_qt_plugin_path():
    """
    OpenCV sets QT_QPA_PLATFORM_PLUGIN_PATH to its own bundled cv2/qt/plugins
    directory, which contains a stale xcb plugin that fails to initialize.
    Override it to point at the REAL PyQt5 platform plugins so QApplication
    can start. Must run before QApplication() is constructed.
    """
    plugins_dir = Path(PyQt5.__file__).parent / "Qt5" / "plugins"
    if plugins_dir.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_dir)


_fix_qt_plugin_path()

# Neon blue — Tony Stark HUD colour
NEON_BLUE = (66, 200, 255)  # (R, G, B) → BGR-ish blue-ish cyan
AMBER = (255, 180, 40)


class NeonCursorOverlay(QWidget):
    """
    A SMALL, borderless, always-on-top targeting reticle that REPLACES the
    system cursor while Telekinesis is active.

    It renders an Iron-Man style neon-blue corner-bracket frame around the
    cursor position and reacts to gesture state:
      - point:   steady neon-blue brackets (aiming)
      - pinch:   brackets contract + white-hot core + pulse rings (click/drag)
      - fist:    brackets turn amber (grab)
      - two_fingers: brackets open vertically with arrows (scroll)
      - neutral: ghosted / dim

    Because the system cursor is hidden (xsetroot) during active control and
    restored on deactivation, this reticle fully replaces the pointer — it is
    never drawn "on top of" the real cursor.
    """

    NEUTRAL = "neutral"
    POINT = "point"
    PINCH = "pinch"
    FIST = "fist"
    SCROLL = "two_fingers"

    # Reticle size in pixels (the ring extent)
    SIZE = 44

    def __init__(self, screen_w: int, screen_h: int, parent=None):
        super().__init__(parent)
        self._sw = screen_w
        self._sh = screen_h
        self._size = self.SIZE

        # ── Frameless, always-on-top, click-through, transparent tool window ──
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(self._size, self._size)

        # ── State ───────────────────────────────────────────────────────────────
        self._active = False
        self._hand_detected = False
        self._cx = screen_w // 2
        self._cy = screen_h // 2
        self._gesture = self.NEUTRAL
        self._confidence = 0.0
        self._fps = 0.0
        self._control_enabled = True
        self._on_escape = lambda: None
        self._on_quit = lambda: None

        # ── Timing ──────────────────────────────────────────────────────────────
        self._t0 = time.monotonic()
        self._start_time = time.monotonic()
        self._fade_alpha = 0.0

        # ── Timer for continuous redraw (~60 fps) ──────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(16)
        self._close_requested = False

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _qcolor(r, g, b, a=255):
        return QColor(r, g, b, a)

    def _center_on_cursor(self):
        """Snap the small window so its centre sits exactly at the cursor."""
        half = self._size / 2
        x = max(0, min(self._sw - self._size, int(self._cx - half)))
        y = max(0, min(self._sh - self._size, int(self._cy - half)))
        self.move(x, y)

    def _on_tick(self):
        # Repaint only (for fade / pulse animation). The window position is
        # driven by update_state(), once per inference frame — repositioning on
        # a 60 Hz timer caused visible lag and ghosting on the compositor.
        self.update()

    # ── public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Begin Telekinesis control: hide system cursor, show the reticle."""
        if self._active:
            return
        self._close_requested = False
        self._active = True
        self._hand_detected = False
        self._fade_alpha = 0.0
        self._center_on_cursor()
        self.show()
        self._hide_system_cursor()

    def stop(self):
        """End Telekinesis control: restore system cursor, hide the reticle."""
        if not self._active:
            return
        self._active = False
        self._hand_detected = False
        self.hide()
        self._show_system_cursor()

    def update_state(
        self,
        cursor_x: int,
        cursor_y: int,
        gesture: str,
        confidence: float,
        hand_detected: bool,
    ):
        self._cx = cursor_x
        self._cy = cursor_y
        self._gesture = gesture
        self._confidence = confidence
        self._hand_detected = hand_detected
        # Snap window to the cursor here (once per frame) — NOT on the timer.
        self._center_on_cursor()
        self.update()

    def update_status(self, fps: float, control_enabled: bool):
        self._fps = fps
        self._control_enabled = control_enabled

    def set_escape_callback(self, cb):
        self._on_escape = cb

    def set_quit_callback(self, cb):
        self._on_quit = cb

    # ── cursor hide / show (REPLACE, not overlay) ──────────────────────────────

    @staticmethod
    def _hide_system_cursor():
        """
        Hide the system cursor using PyQt5 (works on X11 and Wayland).
        Also tries X11-specific methods as fallback.
        """
        # Method 1: PyQt5 native (works on X11 and Wayland)
        try:
            QApplication.instance().setOverrideCursor(QCursor(Qt.BlankCursor))
        except Exception:
            pass
        
        # Method 2: xsetroot (X11 only, will fail silently on Wayland)
        try:
            subprocess.run(
                ["xsetroot", "-cursor_name", "none"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Method 3: unclutter (hide cursor after idle)
        try:
            subprocess.Popen(
                ["unclutter", "-root"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass
    
    @staticmethod
    def _show_system_cursor():
        """Restore the system cursor (works on X11 and Wayland)."""
        # Method 1: PyQt5 native (works on X11 and Wayland)
        try:
            QApplication.instance().restoreOverrideCursor()
        except Exception:
            pass
        
        # Method 2: xsetroot (X11 only, will fail silently on Wayland)
        try:
            subprocess.run(
                ["xsetroot", "-cursor_name", "left_ptr"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Method 3: Kill unclutter if it was started
        try:
            subprocess.run(
                ["pkill", "-f", "unclutter"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    # ── painting ───────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        now = time.monotonic()
        dt = min(now - self._t0, 0.1)
        self._t0 = now
        elapsed = now - self._start_time

        # Fade in/out based on active + hand presence
        if self._active and self._hand_detected:
            self._fade_alpha = min(1.0, self._fade_alpha + dt * 6.0)
        else:
            self._fade_alpha = max(0.0, self._fade_alpha - dt * 4.0)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = self._fade_alpha
        half = self._size / 2

        action = self._gesture in (self.PINCH, self.FIST)
        pulse = (math.sin(elapsed * 7.0) + 1.0) / 2.0 if action else 0.0

        # Colour: amber for grab, neon blue otherwise
        col = AMBER if self._gesture == self.FIST else NEON_BLUE
        r, g, b = col
        # Scroll tint shifts toward the scroll colour while keeping neon blue
        if self._gesture == self.SCROLL:
            r, g, b = 66, 160, 255

        # ── Outer soft glow (stronger while an action is active) ──────────────
        glow_radius = half + 4.0 + pulse * 8.0
        glow_strength = int((150 + pulse * 190) * a)
        grad = QRadialGradient(half, half, glow_radius)
        grad.setColorAt(0.35, self._qcolor(r, g, b, glow_strength))
        grad.setColorAt(0.75, self._qcolor(r, g, b, int(glow_strength * 0.35)))
        grad.setColorAt(1.0, self._qcolor(r, g, b, 0))
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(half - glow_radius), int(half - glow_radius),
                      int(glow_radius * 2), int(glow_radius * 2))

        # ── Thick circular neon ring ──────────────────────────────────────────
        ring_r = half - 5.0          # radius of the ring centreline
        ring_w = 4.0 + pulse * 2.5   # thick boundary, widens on action
        # Ring shrinks slightly when clicked (contract) and brightens on action
        ring_r_use = ring_r + (0.5 if not action else -pulse * 2.0)
        pen = QPen(self._qcolor(r, g, b, int(255 * a)), ring_w)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(int(half - ring_r_use), int(half - ring_r_use),
                      int(ring_r_use * 2), int(ring_r_use * 2))

        # ── White-hot centre dot (aim point) ──────────────────────────────────
        core = 3.0 + pulse * 2.0
        p.setBrush(self._qcolor(255, 255, 255, int(235 * a)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(half - core), int(half - core), int(core * 2), int(core * 2))

        # ── Gesture decorations ────────────────────────────────────────────────
        if self._gesture == self.PINCH:
            # Expanding pulse rings on click (from the ring outward)
            for i in range(3):
                phase = (elapsed * 4.0 + i * 0.7) % 1.0
                if phase < 1.0:
                    rr = ring_r_use + 4.0 + phase * 12.0
                    ring_a = max(0.0, (1.0 - phase) * a)
                    p.setPen(QPen(self._qcolor(r, g, b, int(ring_a * 190)), 1.6))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(int(half - rr), int(half - rr), int(rr * 2), int(rr * 2))
        elif self._gesture == self.SCROLL:
            # Up / down arrows outside the ring
            pen_s = QPen(self._qcolor(66, 160, 255, int(220 * a)), 2.4)
            p.setPen(pen_s)
            p.setBrush(Qt.NoBrush)
            cx, cy = int(half), int(half)
            o = int(ring_r_use) + 6
            p.drawLine(cx, cy - o - 8, cx - 5, cy - o - 2)
            p.drawLine(cx, cy - o - 8, cx + 5, cy - o - 2)
            p.drawLine(cx, cy + o + 8, cx - 5, cy + o + 2)
            p.drawLine(cx, cy + o + 8, cx + 5, cy + o + 2)

        p.end()

    # ── events ────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._on_escape()
            event.accept()
        elif event.key() == Qt.Key_Q:
            self._on_quit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._show_system_cursor()
        super().closeEvent(event)


class VideoPreview(QWidget):
    """
    A small, frameless, always-on-top camera preview window that displays
    the (flipped and skeleton-drawn) webcam frame. Replaces cv2.imshow,
    avoiding the conflict between OpenCV's bundled Qt and PyQt's xcb
    platform.

    Frameless + a painted neon-blue border to match the HUD aesthetic of
    the cursor reticle. Since there's no title bar, the whole window is
    click-and-drag draggable, and right-click closes it (Ctrl+C from the
    terminal still quits the whole app either way).
    """

    BORDER_W = 3          # neon border thickness in pixels
    CORNER_RADIUS = 10    # rounded-corner radius

    def __init__(self, width: int = 240, height: int = 180, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Telekinesis  (Ctrl+C quits, right-click to hide)")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(width, height)
        self._frame: QImage | None = None

        # ── drag-to-move state ─────────────────────────────────────────────
        self._drag_offset: "PyQt5.QtCore.QPoint | None" = None

    def show_frame(self, bgr: "object"):
        """Convert an OpenCV BGR frame to QImage and schedule a repaint."""
        import numpy as np

        arr = np.asarray(bgr)
        if arr.dtype != np.uint8 or arr.ndim != 3:
            return
        h, w, _ = arr.shape
        rgb = np.ascontiguousarray(arr[:, :, ::-1])  # BGR -> RGB
        self._frame = QImage(
            rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888
        ).copy()
        self.update()

    def paintEvent(self, _event):
        from PyQt5.QtGui import QPainterPath

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = self.rect()
        inner = rect.adjusted(
            self.BORDER_W, self.BORDER_W, -self.BORDER_W, -self.BORDER_W
        )

        # Rounded clip path so the video frame respects the rounded corners
        path = QPainterPath()
        path.addRoundedRect(
            float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()),
            self.CORNER_RADIUS, self.CORNER_RADIUS,
        )
        p.setClipPath(path)

        # Video frame (or a dark placeholder before the first frame arrives)
        if self._frame is not None:
            p.drawImage(inner, self._frame)
        else:
            p.fillRect(rect, QColor(10, 14, 18, 235))

        p.setClipping(False)

        # ── Neon-blue HUD border ────────────────────────────────────────────
        r, g, b = NEON_BLUE
        border_pen = QPen(QColor(r, g, b, 235), self.BORDER_W)
        p.setPen(border_pen)
        p.setBrush(Qt.NoBrush)
        half = self.BORDER_W / 2.0
        p.drawRoundedRect(
            rect.adjusted(int(half), int(half), -int(half), -int(half)),
            self.CORNER_RADIUS, self.CORNER_RADIUS,
        )

        p.end()

    # ── drag-to-move (frameless window has no title bar) ───────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.RightButton:
            self.hide()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        event.accept()

    def closeEvent(self, event):
        super().closeEvent(event)