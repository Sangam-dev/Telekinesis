"""
Cursor Calibration Configuration

Adjust these values if the orb doesn't align with your finger position.
Each value represents a percentage of the camera frame (0.0 to 1.0).

Default active_zone: (0.10, 0.90, 0.05, 0.95)
  - x_min: 10% from left
  - x_max: 90% from left
  - y_min: 5% from top
  - y_max: 95% from top

If the orb is consistently offset:
  - Offset to the RIGHT: increase x_min and x_max (shift right)
  - Offset to the LEFT: decrease x_min and x_max (shift left)
  - Offset DOWN: increase y_min and y_max (shift down)
  - Offset UP: decrease y_min and y_max (shift up)

Example: if orb is 3-4 cm to the right and 2 cm down:
  original: (0.10, 0.90, 0.05, 0.95)
  adjusted: (0.15, 0.95, 0.07, 0.97)  # shift by ~5% right, ~2% down

NOTE: CURSOR_OFFSET_X/Y were previously set to (-110, -60) to paper over a
bug where the orb tracked a different position than the real OS cursor
(relative-motion drift in os_control.py's evdev backend — now fixed by
switching to absolute positioning). That offset was compensating for the
wrong problem, not a real hand-to-screen calibration error, so it's been
reset to (0, 0). Re-run calibrate_cursor.py / precise_calibration.py to
re-measure now that the underlying drift is fixed — you likely need a much
smaller value than before, if any.
"""

# Active zone for cursor mapping (x_min, x_max, y_min, y_max) in normalized [0,1] coords
# Expanded from default (0.10, 0.90, 0.05, 0.95) to reduce edge distortion
ACTIVE_ZONE = (0.05, 0.95, 0.02, 0.98)

# Scaling offset in pixels (for fine sub-pixel calibration)
# Adjust these based on visual measurement of orb offset from finger
# Positive = right/down, Negative = left/up
CURSOR_OFFSET_X = 0
CURSOR_OFFSET_Y = 0