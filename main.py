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
from vision.hand_tracker import HandTracker

# ── Show debug window every N inference frames (1 = always, 3 = ~10fps display)
DISPLAY_EVERY_N = 2
VIDEO_WINDOW_NAME = "AI Telekinesis  (ESC = emergency stop)"
VIDEO_COMPACT_SIZE = (360, 270)
VIDEO_EXPANDED_SIZE = (640, 480)


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


def _draw_preview_ui(
    frame: np.ndarray,
    gesture_label: str,
    confidence: float,
    fps: float,
    hand_count: int,
    control_enabled: bool,
) -> np.ndarray:
    """Draw a compact status HUD over the camera frame."""
    height, width = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (width, 104), (12, 18, 25), -1)
    cv2.rectangle(panel, (0, height - 32), (width, height), (12, 18, 25), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)

    accent = (70, 220, 255) if control_enabled else (80, 90, 255)
    status = "ACTIVE" if control_enabled else "STOPPED"
    cv2.circle(frame, (18, 22), 6, accent, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "TELEKINESIS",
        (32, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 242, 248),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (width - 94, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        accent,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        gesture_label.replace("_", " ").upper(),
        (16, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        accent,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{confidence:.0%}  |  {hand_count} hand{'s' if hand_count != 1 else ''}",
        (width - 185, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (215, 225, 235),
        1,
        cv2.LINE_AA,
    )

    bar_left, bar_right = 16, width - 16
    bar_y = 88
    cv2.rectangle(frame, (bar_left, bar_y), (bar_right, bar_y + 5), (45, 58, 68), -1)
    cv2.rectangle(
        frame,
        (bar_left, bar_y),
        (bar_left + int((bar_right - bar_left) * confidence), bar_y + 5),
        accent,
        -1,
    )

    cv2.putText(
        frame,
        f"{fps:.0f} FPS",
        (14, height - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 195, 205),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "C  size   H  hide   Q  quit",
        (width - 174, height - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 195, 205),
        1,
        cv2.LINE_AA,
    )
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
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

    # Wait for the camera thread to deliver its first frame
    print("Waiting for camera…")
    while camera.frame_count == 0:
        time.sleep(0.01)
    print("Running. ESC in the video window = emergency stop.")

    # Keep the preview useful without letting it occupy the whole desktop.
    # WINDOW_NORMAL lets the user resize it freely from the title bar/corners.
    cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(VIDEO_WINDOW_NAME, *VIDEO_COMPACT_SIZE)
    video_visible = True
    video_expanded = False

    loop_iter = 0
    prev_time = time.monotonic()

    while True:
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

        # Feed empty-hand frames to the engine too, so its grace periods can
        # absorb brief tracking dropouts instead of freezing the gesture state.
        engine.process_frame(gesture_label, confidence, hands_lms)

        # ── Display (every N frames to keep render from bottlenecking control) ──
        loop_iter += 1
        if loop_iter % DISPLAY_EVERY_N == 0:
            now = time.monotonic()
            fps = 1.0 / max(now - prev_time, 1e-6) * DISPLAY_EVERY_N
            prev_time = now

            frame = tracker.draw(frame, results)

            frame = _draw_preview_ui(
                frame,
                gesture_label,
                confidence,
                fps,
                len(hands_lms),
                engine.control_enabled,
            )

            if video_visible:
                cv2.imshow(VIDEO_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            engine.emergency_stop()
            print("EMERGENCY STOP. Close the window to exit.")
        elif key == ord("q"):
            break
        elif key == ord("c"):
            video_expanded = not video_expanded
            size = VIDEO_EXPANDED_SIZE if video_expanded else VIDEO_COMPACT_SIZE
            cv2.resizeWindow(VIDEO_WINDOW_NAME, *size)
        elif key == ord("h"):
            video_visible = not video_visible
            cv2.setWindowProperty(
                VIDEO_WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
                1 if video_visible else 0,
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
