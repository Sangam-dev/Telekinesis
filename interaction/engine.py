import math

import numpy as np

import control.os_control as osc
from interaction.geometry import (
    CursorMapper,
    ScrollPositionTracker,
    SwipeTracker,
    ZoomTracker,
)
from interaction.state_machine import GestureStateMachine

# ── Tunables ──────────────────────────────────────────────────────────────────
DRAG_THRESHOLD_PX = 18  # pixels before pinch becomes a drag
FIST_DRAG_SCALE = 1.2  # wrist-movement amplification for window drag

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

        self.cursor_mapper = CursorMapper(screen_w, screen_h)
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
        Moves cursor only during POINT (fingertip) or active PINCH (midpoint).
        Frozen for all other gestures → no hot-corner triggers.
        """
        is_pointing = gesture_label == "point"
        is_pinching = gesture_label == "pinch"

        if is_pointing:
            if not self._was_pointing:
                self.cursor_mapper.reset()
            x, y = self.cursor_mapper.update(primary)
            osc.move_cursor_absolute(x, y)

        elif is_pinching and self._pinch_live:
            mp = _pinch_midpoint(primary)
            x, y = self.cursor_mapper.update(primary, xy_norm=mp)
            osc.move_cursor_absolute(x, y)

            if not self._dragging:
                dx, dy = x - self._drag_start[0], y - self._drag_start[1]
                if math.hypot(dx, dy) > DRAG_THRESHOLD_PX:
                    self._dragging = True
                    osc.mouse_down()

        self._was_pointing = is_pointing

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
        """
        self.fist_sm.update(gesture_label == "fist", confidence)

        if self.fist_sm.just_activated():
            wid, wx, wy = osc.get_window_under_cursor()
            if wid is not None:
                self._fist_window_id = wid
                self._fist_win_pos = [wx, wy]
                self._fist_prev_wrist = primary[WRIST, :2].copy()
                print(f"  [fist] grabbed window {wid} at ({wx},{wy})")
            else:
                self._fist_window_id = None
                self._fist_prev_wrist = None
                print(
                    "  [fist] no moveable window under cursor (pure-Wayland surface?)"
                )

        if self.fist_sm.is_held() and self._fist_window_id is not None:
            wrist_now = primary[WRIST, :2]

            if self._fist_prev_wrist is not None:
                # Per-frame delta → accumulate into window position
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
            self._fist_prev_wrist = wrist_now.copy()

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
