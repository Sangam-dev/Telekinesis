import math
import time
from collections import deque

import numpy as np

INDEX_FINGERTIP_IDX = 8
WRIST_IDX = 0


# ── One-Euro Filter (1D) ─────────────────────────────────────────────────────


class OneEuroFilter:
    """
    One-Euro Filter for a single scalar signal.
    Reference: Casiez et al. CHI 2012.
    """

    def __init__(
        self, min_cutoff: float = 0.8, beta: float = 0.4, d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float | None = None) -> float:
        now = t if t is not None else time.monotonic()
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = now
            return x

        dt = max(now - self._t_prev, 1e-6)
        self._t_prev = now

        # Filter the derivative to estimate speed
        raw_dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx = a_d * raw_dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx

        # Adapt cutoff to speed: fast motion → high cutoff → low lag
        cutoff = self.min_cutoff + self.beta * abs(dx)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


# ── CursorMapper ─────────────────────────────────────────────────────────────


class CursorMapper:
    """
    Maps normalized index-fingertip position to screen pixels.

    Pipeline per frame:
      1. Raw landmark x/y (normalized [0..1] after camera flip).
      2. Remap from active_zone to [0..1] — dead-zones at camera edges.
      3. One-Euro filter in x and y independently.
      4. Scale to screen pixels.
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        active_zone: tuple[float, float, float, float] = (0.10, 0.90, 0.05, 0.95),
        min_cutoff: float = 0.5,
        beta: float = 1.5,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        # active_zone = (x_lo, x_hi, y_lo, y_hi) in normalized camera coords
        self.az_x0, self.az_x1, self.az_y0, self.az_y1 = active_zone
        self._fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def update(
        self, landmarks_xyz: np.ndarray, xy_norm: tuple[float, float] | None = None
    ) -> tuple[int, int]:
        if xy_norm is not None:
            raw_x, raw_y = float(xy_norm[0]), float(xy_norm[1])
        else:
            tip = landmarks_xyz[INDEX_FINGERTIP_IDX]  # normalized [0..1]
            raw_x, raw_y = float(tip[0]), float(tip[1])

        # Remap active zone → [0..1] and clamp
        norm_x = (raw_x - self.az_x0) / max(self.az_x1 - self.az_x0, 1e-6)
        norm_y = (raw_y - self.az_y0) / max(self.az_y1 - self.az_y0, 1e-6)
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # One-Euro filter in normalised [0,1] space.
        # Filtering before scaling keeps beta's units in (normalised/s),
        # which is resolution-independent and prevents tremor velocity in
        # pixel-space from opening the filter wide.
        now = time.monotonic()
        fx = self._fx(norm_x, t=now) * self.screen_w
        fy = self._fy(norm_y, t=now) * self.screen_h

        return int(fx), int(fy)

    def reset(self):
        """Call when the cursor control gesture is re-entered after a pause."""
        self._fx._x_prev = None
        self._fy._x_prev = None


# ── ZoomTracker ───────────────────────────────────────────────────────────────


def palm_center(landmarks_xyz: np.ndarray) -> np.ndarray:
    """Approximate palm center as the average of wrist + middle-finger MCP."""
    return (landmarks_xyz[WRIST_IDX] + landmarks_xyz[9]) / 2.0


class ZoomTracker:
    """Tracks two-hand palm distance over time; returns 'in'/'out'/None."""

    def __init__(self, min_delta: float = 0.02):
        self.prev_distance: float | None = None
        self.min_delta = min_delta

    def update(self, hands_landmarks: list) -> str | None:
        if len(hands_landmarks) < 2:
            self.prev_distance = None
            return None

        p1 = palm_center(hands_landmarks[0])
        p2 = palm_center(hands_landmarks[1])
        distance = float(np.linalg.norm(p1 - p2))

        result = None
        if self.prev_distance is not None:
            delta = distance - self.prev_distance
            if abs(delta) > self.min_delta:
                result = "in" if delta > 0 else "out"

        self.prev_distance = distance
        return result


# ── SwipeTracker ──────────────────────────────────────────────────────────────


class SwipeTracker:
    """Tracks index-fingertip x-position history; detects fast lateral swipes."""

    def __init__(self, history_len: int = 10, velocity_threshold: float = 1.5):
        self.history: deque[tuple[float, float]] = deque(maxlen=history_len)
        self.velocity_threshold = velocity_threshold
        self._last_swipe_time = 0.0
        self._swipe_cooldown = 0.8

    def update(self, landmarks_xyz: np.ndarray) -> str | None:
        now = time.time()
        x = float(landmarks_xyz[INDEX_FINGERTIP_IDX][0])
        self.history.append((now, x))

        if len(self.history) < self.history.maxlen:
            return None
        if now - self._last_swipe_time < self._swipe_cooldown:
            return None

        t0, x0 = self.history[0]
        t1, x1 = self.history[-1]
        dt = max(t1 - t0, 1e-6)
        velocity = (x1 - x0) / dt

        if velocity > self.velocity_threshold:
            self._last_swipe_time = now
            self.history.clear()
            return "right"
        elif velocity < -self.velocity_threshold:
            self._last_swipe_time = now
            self.history.clear()
            return "left"
        return None


# ── ScrollPositionTracker ─────────────────────────────────────────────────────

_SCROLL_WRIST = 0  # wrist landmark index — most stable point during scroll


class ScrollPositionTracker:
    """
    Position-based scroll: wrist Y position relative to the vertical center of
    the camera frame determines both DIRECTION and SPEED.

    You do not need to move your hand to keep scrolling — just hold it
    above or below center.  The further from center, the faster the scroll.

    Layout (camera Y: 0 = top, 1 = bottom)
    ───────────────────────────────────────
      Hand above center  (y < 0.5)  →  scroll UP   (positive notches)
      Hand at center     (|offset| < dead_zone)  →  no scroll (dead zone)
      Hand below center  (y > 0.5)  →  scroll DOWN (negative notches)

    Speed curve
    ───────────
      At the dead-zone boundary  →  0 notches / sec
      At the frame edge (y=0 or y=1)  →  max_speed notches / sec
      Curve is linear between the two.  Raise max_speed to scroll faster.

    Tuning
    ──────
      dead_zone   Camera fraction around centre with no scroll.  Default 0.12
                  (hand must be >12% above or below centre to start scrolling).
      max_speed   Notches/sec at maximum displacement.  Default 14.
    """

    def __init__(self, dead_zone: float = 0.12, max_speed: float = 14.0):
        self.dead_zone = dead_zone
        self.max_speed = max_speed
        self._prev_t: float = 0.0

    def reset(self):
        self._prev_t = 0.0

    def update(self, landmarks_xyz: np.ndarray) -> float:
        """
        Returns notches this frame (positive = up, negative = down).
        Called every frame the TWO_FINGER gesture is held.
        """
        now = time.monotonic()
        # Use real dt so speed is frame-rate independent
        dt = max(now - self._prev_t, 1e-6) if self._prev_t > 0.0 else 1.0 / 30.0
        self._prev_t = now

        y = float(landmarks_xyz[_SCROLL_WRIST][1])  # normalized [0..1]
        offset = y - 0.5  # signed: + = below centre

        if abs(offset) < self.dead_zone:
            return 0.0

        # Strip dead zone then normalise to [0..1]
        sign = 1.0 if offset > 0.0 else -1.0
        active = (abs(offset) - self.dead_zone) / max(0.5 - self.dead_zone, 1e-6)
        active = min(1.0, active)

        # notches this frame = speed × time
        notches_per_sec = active * self.max_speed
        return -sign * notches_per_sec * dt  # negate: y↓ = scroll down

    # Kept for callers that still reference ScrollVelocityTracker by name
    ScrollVelocityTracker = None  # type: ignore[assignment]  (see below)


# Alias so any code that still imports the old name keeps working
ScrollVelocityTracker = ScrollPositionTracker
