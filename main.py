import subprocess
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import control.os_control as osc
from features.extractor import extract_feature
from interaction.engine import InteractionEngine
from ml.model import GESTURE_CLASSES, load_model
from ui.overlay import NeonCursorOverlay, VideoPreview
from vision.hand_tracker import HandTracker

from PyQt5.QtWidgets import QApplication

# ── Show debug window every N inference frames (1 = always, 3 = ~10fps display)
DISPLAY_EVERY_N = 2


# ── Probability smoother ──────────────────────────────────────────────────────


class _ProbSmoother:
    def __init__(
        self,
        num_classes: int,
        alpha_rise: float = 0.35,
        alpha_fall: float = 0.15,
    ):
        self._alpha_rise = alpha_rise
        self._alpha_fall = alpha_fall
        self._smooth = np.zeros(num_classes, dtype=np.float32)
        self._initialised = False

    def update(self, probs: np.ndarray) -> np.ndarray:
        """probs: (num_classes,) numpy array from softmax.  Returns smoothed probs."""
        if not self._initialised:
            self._smooth = probs.copy()
            self._initialised = True
            return self._smooth

        # Choose per-class alpha: fast for rising, slow for falling
        alpha = np.where(probs > self._smooth, self._alpha_rise, self._alpha_fall)
        self._smooth = alpha * probs + (1.0 - alpha) * self._smooth
        return self._smooth


# ── Camera thread ─────────────────────────────────────────────────────────────


class _CameraReader(threading.Thread):
    """
    Reads from V4L2 continuously and stores the most recent frame.
    The inference loop always gets the freshest image regardless of
    how long inference took — no stale-frame buffering.
    """

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name="camera-reader")
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ok: bool = False
        self._frame_count: int = 0

    def run(self):
        while True:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._ok = ok
                    self._frame = frame
                    self._frame_count += 1

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_screen_resolution() -> tuple[int, int]:
    try:
        out = subprocess.check_output(["xrandr"]).decode()
        for line in out.splitlines():
            if " connected" in line and "x" in line:
                for token in line.split():
                    if "x" in token and token[0].isdigit():
                        w, h = token.split("+")[0].split("x")
                        return int(w), int(h)
    except Exception:
        pass
    print("Could not detect screen resolution — defaulting to 1920x1080.")
    return 1920, 1080


def _open_camera() -> cv2.VideoCapture:
    """Open camera with settings tuned for minimum latency."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # explicit V4L2 backend
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # kernel delivers at most 1 queued frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # MediaPipe works well at 640×480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # MJPEG: camera compresses on-chip → faster USB transfer → lower latency
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    # ── Qt must initialise FIRST, before mediapipe/opencv spin up their own
    #    EGL/OpenGL contexts. Constructing it later crashes QApplication with
    #    "Could not load the Qt platform plugin xcb".
    app = QApplication.instance() or QApplication([])

    screen_w, screen_h = get_screen_resolution()
    print(f"Detected screen resolution: {screen_w}x{screen_h}")

    osc.init_cursor_backend(screen_w, screen_h)

    model = load_model("models/gesture_mlp.pt")
    cap = _open_camera()
    camera = _CameraReader(cap)
    camera.start()
    tracker = HandTracker(max_hands=2)
    engine = InteractionEngine(screen_w, screen_h)
    smoother = _ProbSmoother(num_classes=len(GESTURE_CLASSES))

    # ── Neon reticle (replaces the system cursor while active) ───────────────
    overlay = NeonCursorOverlay(screen_w, screen_h)
    overlay_started = False
    running = True

    # ── Debug: set to True to print coordinates ──────────────────────────────
    DEBUG_COORDS = False

    # ── Camera preview window (replaces cv2.imshow, no Qt conflict) ──────────
    preview = VideoPreview()
    preview.show()

    def _on_quit():
        nonlocal running
        running = False

    def _on_escape():
        engine.emergency_stop()
        print("EMERGENCY STOP. Press Ctrl+C to exit.")

    overlay.set_escape_callback(_on_escape)
    overlay.set_quit_callback(_on_quit)

    # Wait for the camera thread to deliver its first frame
    print("Waiting for camera…")
    while camera.frame_count == 0:
        time.sleep(0.01)
    print("Running. Ctrl+C quits. ESC = emergency stop.")

    loop_iter = 0
    prev_time = time.monotonic()

    try:
        while running:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)

            results, hands_lms = tracker.process(frame)

            gesture_label = "neutral"
            confidence = 0.0

            if hands_lms:
                feats = extract_feature(hands_lms[0])
                x_in = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x_in)
                    raw_probs = F.softmax(logits, dim=1).squeeze(0).numpy()

                # Asymmetric EMA: fast to recognise new gestures, slow to drop them.
                # Prevents 1-3 frame classifier hiccups from interrupting drag/scroll/fist.
                smooth_probs = smoother.update(raw_probs)

                pred_idx = int(smooth_probs.argmax())
                gesture_label = GESTURE_CLASSES[pred_idx]
                confidence = float(smooth_probs[pred_idx])
                engine.process_frame(gesture_label, confidence, hands_lms)

            # ── Neon reticle: replaces the cursor while control is enabled ───
            if engine.control_enabled:
                if not overlay_started:
                    overlay.start()
                    overlay_started = True
                cx, cy = engine.get_cursor_position()
                if DEBUG_COORDS and loop_iter % 10 == 0:
                    print(f"[DEBUG] Overlay: ({cx}, {cy})")
                overlay.update_state(
                    cx,
                    cy,
                    gesture_label,
                    confidence,
                    hand_detected=bool(hands_lms),
                )
            else:
                if overlay_started:
                    overlay.stop()
                    overlay_started = False

            # ── Camera preview + HUD status (every N frames, no cv2.imshow) ──
            loop_iter += 1
            if loop_iter % DISPLAY_EVERY_N == 0:
                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6) * DISPLAY_EVERY_N
                prev_time = now
                overlay.update_status(fps, control_enabled=engine.control_enabled)
                # Draw skeleton overlay onto the frame and show it in preview
                disp = tracker.draw(frame.copy(), results)
                preview.show_frame(disp)

            app.processEvents()
    except KeyboardInterrupt:
        print("\nQuit requested.")
    finally:
        if engine.control_enabled:
            engine.emergency_stop()
        if overlay_started:
            overlay.stop()
        preview.close()
        cap.release()
        print("Cleaned up.")


if __name__ == "__main__":
    main()
