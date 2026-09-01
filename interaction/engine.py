import math
import time

import numpy as np

import control.os_control as osc
from config.cursor_calibration import ACTIVE_ZONE, CURSOR_OFFSET_X, CURSOR_OFFSET_Y
from interaction.geometry import (
    CursorMapper,
    OneEuroFilter,
    ScrollPositionTracker,
    SwipeTracker,
    ZoomTracker,
)
from interaction.state_machine import GestureStateMachine

# ── Tunables ──────────────────────────────────────────────────────────────────
DRAG_THRESHOLD_PX = 18  # pixels before pinch becomes a drag
FIST_DRAG_SCALE = 1.2  # wrist-movement amplification for window drag
CURSOR_DEAD_ZONE_PX = 4  # ignore moves smaller than this to suppress micro-jitter

# Landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0


def _pinch_midpoint(lm: np.ndarray) -> tuple[float, float]:
    mp = (lm[THUMB_TIP, :2] + lm[INDEX_TIP, :2]) / 2.0
    return float(mp[0]), float(mp[1])


class InteractionEngine:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # Use calibration config for active zone
        self.cursor_mapper = CursorMapper(screen_w, screen_h, active_zone=ACTIVE_ZONE)
        self.zoom_tracker = ZoomTracker()
        self.swipe_tracker = SwipeTracker()
        self.scroll_tracker = ScrollPositionTracker()

        # exit_frames_required = grace period (frames) for classifier noise / hand loss
        self.pinch_sm = GestureStateMachine(
            stable_frames_required=4, exit_frames_required=5, cooldown_sec=0.25
        )
        self.scroll_sm = GestureStateMachine(
            stable_frames_required=3, exit_frames_required=8, cooldown_sec=0.10
        )
        self.fist_sm = GestureStateMachine(
            stable_frames_required=5, exit_frames_required=8, cooldown_sec=0.30
        )

        # Pinch / drag state
        self._pinch_live: bool = False
        self._dragging: bool = False
        self._drag_start: tuple[int, int] = (0, 0)

        # Fist / window-grab state  (incremental delta design)
        self._fist_window_id: int | None = None
        self._fist_win_pos: list[int] = [0, 0]  # current window x,y
        self._fist_prev_wrist: np.ndarray | None = None  # wrist position last frame

        # Smoothing filters for fist-drag wrist position (normalised coords)
        self._fist_wrist_fx = OneEuroFilter(min_cutoff=1.0, beta=2.0)
        self._fist_wrist_fy = OneEuroFilter(min_cutoff=1.0, beta=2.0)

        # Dead-zone: last pixel position actually sent to the OS
        self._last_cursor_px: int | None = None
        self._last_cursor_py: int | None = None

        # Current calculated cursor position (from hand landmarks, with or without dead-zone)
        # This is the source-of-truth for overlay positioning — eliminates OS cursor latency.
        self._current_cursor_x: int = screen_w // 2
        self._current_cursor_y: int = screen_h // 2

        # Grace-period: last successfully tracked hand landmarks
        self._last_primary: np.ndarray | None = None

        self.control_enabled = True
        self._was_pointing = False

        # Screen-pixel-per-normalised-unit scale (mirrors cursor active zone)
        az = self.cursor_mapper
        self._win_scale_x = screen_w / max(az.az_x1 - az.az_x0, 1e-6)
        self._win_scale_y = screen_h / max(az.az_y1 - az.az_y0, 1e-6)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def emergency_stop(self):
        if self._dragging:
            osc.mouse_up()
        self._pinch_live = False
        self._dragging = False
        self._fist_window_id = None
        self.control_enabled = False

    def get_cursor_position(self) -> tuple[int, int]:
        """Return the position the orb should be drawn at.

        Returns the REAL, polled OS pointer position (osc.get_cursor_position())
        rather than our internally-computed target. Now that the evdev backend
        moves the OS cursor with absolute positioning (see os_control.py), the
        target and the real pointer should always agree — but reading the
        polled ground-truth here still means the orb self-corrects for free
        if it's ever pushed by something outside our control (backend
        fallback to ydotool, another app moving the mouse, etc.), at the
        cost of at most one poll interval (~30ms) of latency.
        """
        return osc.get_cursor_position()

    def resume(self):
        self.control_enabled = True

    # ── per-frame entry ───────────────────────────────────────────────────────

    def process_frame(
        self, gesture_label: str, confidence: float, hands_landmarks: list
    ) -> None:
        if not self.control_enabled:
            if self._dragging:
                osc.mouse_up()
                self._dragging = False
            self._fist_window_id = None
            return

        has_hands = bool(hands_landmarks)

        if has_hands:
            # Normal frame — update cached landmarks
            primary = hands_landmarks[0]
            self._last_primary = primary.copy()
        else:
            # Hand temporarily lost (left frame, bad lighting, etc.)
            # Use frozen last-known landmarks so state machines can count
            # down their exit_frames_required grace period rather than
            # immediately killing ongoing drag / scroll / fist.
            if self._last_primary is None:
                return  # No history yet — nothing to preserve
            primary = self._last_primary
            gesture_label = "neutral"  # no gesture, but don't hard-cancel yet
            confidence = 0.0

        self._handle_cursor(gesture_label, primary)
        self._handle_pinch(gesture_label, confidence, primary)
        self._handle_scroll(gesture_label, confidence, primary)
        self._handle_fist(gesture_label, confidence, primary)
        self._handle_zoom(hands_landmarks)
        self._handle_swipe(primary)

    # ── cursor ────────────────────────────────────────────────────────────────

    def _handle_cursor(self, gesture_label: str, primary: np.ndarray) -> None:
        """
        Always calculate and update cursor position based on current gesture.
        The orb position is continuously synced to the calculated position.
        
        For OS cursor movement (click/drag), we apply dead-zone only during
        POINT and active PINCH to prevent accidental clicks during other gestures.
        """
        is_pointing = gesture_label == "point"
        is_pinching = gesture_label == "pinch"
        is_fisting = gesture_label == "fist"
        is_scrolling = gesture_label == "two_fingers"

        # Always calculate cursor position based on current gesture
        # This ensures the orb is never frozen during any action
        if is_pointing:
            if not self._was_pointing:
                self.cursor_mapper.reset()
                self._last_cursor_px = None
                self._last_cursor_py = None
            x, y = self.cursor_mapper.update(primary)
            self._move_cursor_deadzoned(x, y)

        elif is_pinching and self._pinch_live:
            mp = _pinch_midpoint(primary)
            x, y = self.cursor_mapper.update(primary, xy_norm=mp)
            self._move_cursor_deadzoned(x, y)

            if not self._dragging:
                dx, dy = x - self._drag_start[0], y - self._drag_start[1]
                if math.hypot(dx, dy) > DRAG_THRESHOLD_PX:
                    self._dragging = True
                    osc.mouse_down()

        elif is_fisting:
            # During fist, keep orb tracking wrist but don't move OS cursor
            wrist_pos = primary[WRIST, :2]
            x, y = self.cursor_mapper.update(primary, xy_norm=tuple(wrist_pos))
            self._update_overlay_position_only(x, y)

        elif is_scrolling:
            # During scroll, keep orb tracking wrist but don't move OS cursor
            wrist_pos = primary[WRIST, :2]
            x, y = self.cursor_mapper.update(primary, xy_norm=tuple(wrist_pos))
            self._update_overlay_position_only(x, y)

        else:
            # NEUTRAL or unknown: keep orb at last calculated position
            # (already in _current_cursor_x/y from previous frame)
            pass

        self._was_pointing = is_pointing

    def _move_cursor_deadzoned(self, x: int, y: int) -> None:
        """Move OS cursor to exact position.
        The orb position and OS cursor position MUST match exactly,
        with no dead-zone filtering, so they appear as one unified pointer.
        """
        # Apply calibration offset
        x_cal = x + CURSOR_OFFSET_X
        y_cal = y + CURSOR_OFFSET_Y
        
        # Always update BOTH overlay and OS cursor to the same position (no filtering)
        self._current_cursor_x = x_cal
        self._current_cursor_y = y_cal
        
        # Move OS cursor directly to the calculated position (NO dead-zone)
        # This ensures the system cursor and orb are always at the same coordinates
        osc.move_cursor_absolute(x_cal, y_cal)

    def _update_overlay_position_only(self, x: int, y: int) -> None:
        """Update overlay and OS cursor position during FIST/SCROLL.
        The cursor should always follow the hand, even during these gestures,
        so the orb and system cursor remain aligned.
        """
        # Apply calibration offset
        x_cal = x + CURSOR_OFFSET_X
        y_cal = y + CURSOR_OFFSET_Y
        
        # Update BOTH overlay and OS cursor position
        self._current_cursor_x = x_cal
        self._current_cursor_y = y_cal
        
        # Move OS cursor to keep it aligned with orb
        osc.move_cursor_absolute(x_cal, y_cal)

    # ── pinch → click / drag ─────────────────────────────────────────────────

    def _handle_pinch(
        self, gesture_label: str, confidence: float, primary: np.ndarray
    ) -> None:
        self.pinch_sm.update(gesture_label == "pinch", confidence)

        if self.pinch_sm.just_activated():
            mp = _pinch_midpoint(primary)
            sx, sy = self.cursor_mapper.update(primary, xy_norm=mp)
            self._drag_start = (sx, sy)
            self._pinch_live = True
            self._dragging = False

        if self.pinch_sm.just_released():
            if self._dragging:
                osc.mouse_up()
            else:
                osc.click_left()
            self._pinch_live = False
            self._dragging = False

    # ── two-finger scroll ─────────────────────────────────────────────────────

    def _handle_scroll(
        self, gesture_label: str, confidence: float, primary: np.ndarray
    ) -> None:
        self.scroll_sm.update(gesture_label == "two_fingers", confidence)

        if self.scroll_sm.just_activated():
            self.scroll_tracker.reset()

        if self.scroll_sm.is_held():
            notches = self.scroll_tracker.update(primary)
            if notches != 0.0:
                osc.scroll_wheel(notches)

    # ── fist → window grab + drag ─────────────────────────────────────────────

    def _handle_fist(
        self, gesture_label: str, confidence: float, primary: np.ndarray
    ) -> None:
        """
        FIST grabs the window under the cursor and moves it with the hand.

        Incremental delta design
        ─────────────────────────
        Instead of computing (wrist_now - wrist_at_grab) each frame, we
        accumulate per-frame deltas:
            win_pos += (wrist_this_frame - wrist_last_frame) × scale

        Benefit: if the hand briefly leaves the camera frame (wrist goes to
        edge) and returns, the previous frame's wrist IS the correct reference.
        No jump, no lost position — exactly one frame of zero delta (the frame
        the hand reappears), then drag resumes smoothly.

        The wrist position is One-Euro filtered before computing the delta so
        that small hand tremor does not jitter the dragged window.
        """
        self.fist_sm.update(gesture_label == "fist", confidence)

        if self.fist_sm.just_activated():
            # Reset wrist smoothing so the very first filtered sample == raw sample
            # (no history to pull from — avoids a phantom jump on activation).
            self._fist_wrist_fx._x_prev = None
            self._fist_wrist_fy._x_prev = None

            wid, wx, wy = osc.get_window_under_cursor()
            if wid is not None:
                self._fist_window_id = wid
                self._fist_win_pos = [wx, wy]
                raw = primary[WRIST, :2]
                now = time.monotonic()
                sx = self._fist_wrist_fx(float(raw[0]), t=now)
                sy = self._fist_wrist_fy(float(raw[1]), t=now)
                self._fist_prev_wrist = np.array([sx, sy], dtype=np.float64)
                print(f"  [fist] grabbed window {wid} at ({wx},{wy})")
            else:
                self._fist_window_id = None
                self._fist_prev_wrist = None
                print(
                    "  [fist] no moveable window under cursor (pure-Wayland surface?)"
                )

        if self.fist_sm.is_held() and self._fist_window_id is not None:
            raw = primary[WRIST, :2]
            now = time.monotonic()
            sx = self._fist_wrist_fx(float(raw[0]), t=now)
            sy = self._fist_wrist_fy(float(raw[1]), t=now)
            wrist_now = np.array([sx, sy], dtype=np.float64)

            if self._fist_prev_wrist is not None:
                # Per-frame delta on smoothed wrist → accumulate into window position
                dx_norm = float(wrist_now[0] - self._fist_prev_wrist[0])
                dy_norm = float(wrist_now[1] - self._fist_prev_wrist[1])
                self._fist_win_pos[0] += int(
                    dx_norm * self._win_scale_x * FIST_DRAG_SCALE
                )
                self._fist_win_pos[1] += int(
                    dy_norm * self._win_scale_y * FIST_DRAG_SCALE
                )
                # Keep window title bar on-screen (GNOME refuses to place it above y=0)
                self._fist_win_pos[1] = max(0, self._fist_win_pos[1])
                osc.move_window(
                    self._fist_window_id, self._fist_win_pos[0], self._fist_win_pos[1]
                )

            # Always update prev_wrist — during grace period this stays frozen,
            # so delta = 0 (window holds) until the hand reappears.
            self._fist_prev_wrist = wrist_now

        if self.fist_sm.just_released():
            self._fist_window_id = None
            self._fist_prev_wrist = None

    # ── zoom ──────────────────────────────────────────────────────────────────

    def _handle_zoom(self, hands_landmarks: list) -> None:
        result = self.zoom_tracker.update(hands_landmarks)
        if result == "in":
            osc.zoom_in()
        elif result == "out":
            osc.zoom_out()

    # ── swipe ─────────────────────────────────────────────────────────────────

    def _handle_swipe(self, primary: np.ndarray) -> None:
        result = self.swipe_tracker.update(primary)
        if result == "right":
            osc.swipe_next()
        elif result == "left":
            osc.swipe_prev()