# Trace: The Pointer-Control System — Deep Dive

This document traces **every piece of the pointer/OS-control subsystem** in
Telekinesis: the geometry, the state machines, the engine that wires gestures
to actions, and the OS backends (evdev / ydotool / xdotool). It explains **how
a gesture becomes an action** and — critically — **how continuity is maintained**
frame-to-frame so that drag, scroll, fist-window-grab, and cursor position don't
jitter, jump, or die mid-interaction.

> Scope note: This covers the **control** chain that runs *after* the ML
> classifier picks a gesture. The ML side (how `"pinch"` is recognised from
> landmarks) is covered in `course.md` / `course_ml.md`. Here we assume the
> classifier has already produced a `gesture_label` + `confidence` each frame,
> and we trace what happens next.

---

## Table of Contents

1. [Where the Control Chain Starts](#1-where-the-control-chain-starts)
2. [The Pipeline Top to Bottom](#2-the-pipeline-top-to-bottom)
3. [Every Tool in the Toolbox](#3-every-tool-in-the-toolbox)
4. [The Gesture State Machine — the Node That Holds Everything Together](#4-the-gesture-state-machine)
5. [The One-Euro Filter — Smoothing Movement](#5-the-one-euro-filter)
6. [The Cursor Mapper — Hand Position → Screen Pixels](#6-the-cursor-mapper)
7. [The Interaction Engine — Routing Gestures to Actions](#7-the-interaction-engine)
8. [Each Gesture, Traced End to End](#8-each-gesture-traced-end-to-end)
9. [Continuity & Stability — the Core Design Math](#9-continuity--stability)
10. [The OS Backends — How Input Actually Reaches the Computer](#10-the-os-backends)
11. [Emergency Stop & Safety](#11-emergency-stop--safety)
12. [A Worked Frame-by-Frame Trace](#12-a-worked-frame-by-frame-trace)
13. [Glossary of Control Concepts](#13-glossary)

---

## 1. Where the Control Chain Starts

In `main.py`'s inference loop, for every camera frame:

```python
engine.process_frame(gesture_label, confidence, hands_lms)
```

We pass three things into `InteractionEngine.process_frame`:

- **`gesture_label`** — a string: one of the 6 classes, OR the fallbacks
  `"neutral"` when no hand / no confident gesture.
- **`confidence`** — the (EMA-smoothed) softmax probability of that prediction.
- **`hands_lms`** — a list of `(21, 3)` landmark arrays; `hands_lms[0]` is the
  **primary hand** the control system reacts to.

Everything downstream is **deterministic geometry + state machines** — no ML.
The ML already did its job; now we turn landmark geometry into real OS input.

---

## 2. The Pipeline Top to Bottom

```
per-frame:
  gesture_label, confidence, hands_landmarks
        │
        ▼
  InteractionEngine.process_frame()
        │   ┌─ hand lost? → use frozen _last_primary, label="neutral", conf=0
        ├──► _handle_cursor()   (point / pinch-midpoint → move_cursor_absolute)
        ├──► _handle_pinch()    (pinch → click / drag)
        ├──► _handle_scroll()   (two_fingers → scroll_wheel)
        ├──► _handle_fist()     (fist → window grab + move)
        ├──► _handle_zoom()     (two hands → zoom in/out)
        └──► _handle_swipe()    (fast lateral move → page next/prev)
        │
        ▼
  control/os_control.py  (backends switch)
        ├── evdev uinput  (virtual relative mouse)  ── preferred, zero-latency
        └── ydotool       (threaded cursor worker)  ── fallback
        │
        ▼
  Real OS: cursor / click / drag / scroll / keys / window move
```

The **engine is called once per frame** from the main loop, and it fans out to
each gesture handler. Each handler uses its **own** state machine so gestutes
don't interfere with one another.

---

## 3. Every Tool in the Toolbox

| Tool | What it does | Where |
|------|--------------|-------|
| **OpenCV (`cv2`)** | Captures frames; renders the debug window; reads ESC/q keys; `cv2.flip` mirrors the image for natural interaction. | `main.py`, `collect_data.py`, `test_live.py` |
| **MediaPipe Hands** | Detects 21 landmarks per hand from a frame. | `vision/hand_tracker.py` |
| **NumPy** | Landmark arrays, vector/matrix math (pinch midpoint, palm center, distances, deltas). | everywhere |
| **PyTorch (`torch`)** | Runs the `GestureMLP` classifier; `F.softmax` for probabilities. | `ml/`, `main.py` |
| **`time` / `threading`** | Frame timing (real `dt` for speed), background worker threads for non-blocking cursor/window moves. | `main.py`, `os_control.py` |
| **evdev `UInput` + `ecodes`** | Creates a **virtual relative mouse** at `/dev/uinput` — the fast, zero-latency cursor backend. | `control/os_control.py` |
| **ydotool + ydotoold** | The fallback cursor/input backend (works even where evdev is denied). Runs `ydotool mousemove` / `click` / `key`. | `control/os_control.py` |
| **xdotool** | Queries the window under the cursor (`getwindowgeometry`) and moves X11/XWayland windows (`windowmove`); also reads current cursor pos. | `control/os_control.py` |
| **subprocess** | Shells out to `ydotool` and `xdotool` on the backend worker threads. | `control/os_control.py` |
| **`xrandr`** | Detects monitor resolution so coordinates map correctly. | `main.py`, `os_control.py` |
| **`collections.deque`** | Fixed-length sliding window for swipe velocity history. | `interaction/geometry.py` |
| **`matplotlib`** | (ML eval only) plots training curve / confusion matrix. | `ml/train.py` |

---

## 4. The Gesture State Machine

`interaction/state_machine.py` — the single most important device for both
**turn gesture→action** and **maintain continuity**.

### The states

```
IDLE ──confident──▶ DETECTING ──stable xN──▶ CONFIRMED ──▶ ACTIVE
                                                              │
IDLE ◀─cooldown── COOLDOWN ◀── RELEASED ◀─exit xM (unconfident)┘
```

### The two-step entry (strict)

1. From **IDLE**: the first confident frame (`gesture_active AND confidence ≥
   0.85`) moves to **DETECTING**.
2. In **DETECTING**, the gesture must stay confident for
   `stable_frames_required` consecutive frames before **CONFIRMED**.
   - **Any** unconfident frame in DETECTING resets straight back to IDLE.
   - This is the *strict* guard against accidentally triggering an action from
     a 1-frame flicker.

### The one-step activation

**CONFIRMED → ACTIVE** on the next frame, and **only on that exact frame** does
`just_activated()` return `True`.

> This is the mechanism by which a gesture "fires an action": the engine checks
> `just_activated()` in `_handle_pinch()` / `_handle_fist()` to run the
> *one-time* setup for that gesture (record drag-start, grab the window, etc.).

### The lenient exit (grace period)

Once **ACTIVE**, an unconfident frame does **NOT** immediately kill the gesture.
Instead it increments `_exit_count`. Only after `exit_frames_required`
consecutive unconfident frames does it transition **RELEASED** (and only that
one frame has `just_released() == True`).

- If confidence recovers within the grace window, `_exit_count` resets to 0 and
  the gesture continues uninterrupted.
- This is the **asymmetry**: hard to enter, easy to keep holding. That's the
  deliberate design so drag/scroll/fist don't drop out from classifier noise.

### The cooldown

**RELEASED → COOLDOWN**, which waits `cooldown_sec` before returning to IDLE.
This prevents immediate re-triggering (e.g., a pinch flicking on/off and
double-clicking).

### The per-gesture tunings

| Machine | stable (enter) | exit (grace) | cooldown | used for |
|---------|----------------|--------------|----------|----------|
| `pinch_sm` | 4 frames | 5 frames | 0.25 s | click / drag |
| `scroll_sm` | 3 frames | 8 frames | 0.10 s | two-finger scroll |
| `fist_sm` | 5 frames | 8 frames | 0.30 s | window grab |

Scroll and fist get **longer grace periods** because they are *sustained*
actions where dropping out mid-way would be disruptive; pinch activates faster
(3-4 frames) so clicks feel responsive.

---

## 5. The One-Euro Filter

`interaction/geometry.py::_OneEuro` — an adaptive low-pass filter (Casiez et
al., CHI 2012) that is the key to **smooth but lag-free** cursor motion.

### Why a normal low-pass filter is not enough

A standard low-pass filter (fixed cutoff) either:
- is **too slow** (high smoothing → cursor lags behind your hand when moving
  fast), or
- is **too jumpy** (low smoothing → jitter when hand is still).

### The One-Euro trick: cutoff adapts to speed

The filter smooths the signal, but **raises its cutoff frequency as you move
faster**:

```python
def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)

# per update:
raw_dx = (x - x_prev) / dt          # speed estimate
a_d = self._alpha(self.d_cutoff, dt)   # derivative filter coeff
dx = a_d*raw_dx + (1-a_d)*dx_prev      # smoothed velocity
cutoff = self.min_cutoff + self.beta*abs(dx)   # ← ADAPTIVE
a = self._alpha(cutoff, dt)
x_hat = a*x + (1-a)*x_prev
```

- **Still hand** (low speed) → low cutoff → strong smoothing → no jitter.
- **Fast hand** (high speed) → high cutoff → low lag → cursor keeps up.
- `beta` controls how aggressively the cutoff responds to speed.

### Why `reset()` matters for continuity

```python
def reset(self):
    self._fx._x_prev = None
    self._fy._x_prev = None
```

When the cursor-control gesture is *re-entered* after a pause (e.g., you stop
pointing and then point again), the filter's history is cleared. Otherwise the
filter would remember a stale old position and "catch up" with a visible jump.
`export`: re-entering pointing = fresh filter = no jump.

---

## 6. The Cursor Mapper

`interaction/geometry.py::CursorMapper` — maps a hand position to screen pixels.

```
raw landmark (0..1)
   │  remap active_zone → [0..1], clamp
   ▼
norm_x/norm_y (0..1)
   │  One-Euro filter (x and y independently)
   ▼
norm * screen_size
   │
   ▼
int (sx, sy)  →  move_cursor_absolute(sx, sy)
```

### The active (dead) zone

```python
active_zone = (0.10, 0.90, 0.05, 0.95)   # x_lo, x_hi, y_lo, y_hi
```

The camera's outer 10% (x) and 5% (y) edges are **dead zones**. The hand must be
inside the box `[0.10..0.90] x [0.05..0.95]` to control the cursor. The zone is
remapped linearly to the full `[0..1]`, then to screen pixels:

```python
norm_x = (raw_x - az_x0) / max(az_x1 - az_x0, 1e-6)
norm_y = (raw_y - az_y0) / max(az_y1 - az_y0, 1e-6)
```

The remap is the same one used later for the window-drag scale in the engine
(`_win_scale_x / _win_scale_y`), so the cursor and the window-drag share a
consistent mapping between camera-space and screen-space.

### Two cursor feeds

`update()` accepts either:
- the **index fingertip** (landmark 8) — used for POINT, or
- an explicit `xy_norm` midpoint — used for PINCH (thumb/Index midpoint).

This lets one mapper serve both gestures.

---

## 7. The Interaction Engine

`interaction/engine.py::InteractionEngine` is the **router**. Each frame it:

1. **Decides the "primary" hand.** Uses `hands_landmarks[0]`, or, if no hand,
   the frozen `_last_primary` with `label="neutral"`, `conf=0` (see
   [§9 Continuity](#9-continuity--stability)).
2. **Fans the frame out** to all six handlers in sequence.
3. Each handler consults **its own state machine** and calls into `osc.*`.

### Module-level tunables

```python
DRAG_THRESHOLD_PX = 18   # pixels the pinch must travel before it becomes a drag
FIST_DRAG_SCALE = 1.2    # amplify wrist motion ×1.2 for window drag
```

---

## 8. Each Gesture, Traced End to End

### A) POINT → cursor move

- `_handle_cursor`: `is_pointing = (label == "point")`.
- On the **first** pointing frame (`not self._was_pointing`) it calls
  `cursor_mapper.reset()` — fresh One-Euro filter → no jump on entry.
- `(x, y) = cursor_mapper.update(primary)` → fingertip (landmark 8) mapped to
  pixels → `osc.move_cursor_absolute(x, y)`.
- `_was_pointing` is updated each frame.
- Cursor moves **only during POINT** (or active PINCH; see below). For all other
  gestures it stays put — deliberately, to avoid accidental hot-corner
  triggers in GNOME.

### B) PINCH → click or drag

This is the richest interaction; it uses **two** systems: the cursor mapper for
motion and the pinch state machine for the click/drag lifecycle.

1. **Activation** (`pinch_sm.just_activated()`):
   ```python
   mp = _pinch_midpoint(primary)              # midpoint of thumb(4) & index(8)
   sx, sy = cursor_mapper.update(primary, xy_norm=mp)
   self._drag_start = (sx, sy)                # remember where pinch began
   self._pinch_live = True
   self._dragging = False
   ```
   The midpoint of thumb+index becomes the drag anchor.

2. **Held** (`_pinch_live` and label still `pinch`): `_handle_cursor` moves the
   cursor to the midpoint. If not yet dragging, it checks distance from
   `_drag_start`:
   ```python
   dx, dy = x - drag_start[0], y - drag_start[1]
   if math.hypot(dx, dy) > DRAG_THRESHOLD_PX:   # moved >18px
       self._dragging = True
       osc.mouse_down()
   ```
   So: **pinch-in-place = click; pinch-then-move = drag.** The 18px threshold
   distinguishes "pointing then pressing" from "dragging".

3. **Release** (`pinch_sm.just_released()`):
   ```python
   if self._dragging: osc.mouse_up()        # end the drag
   else:              osc.click_left()      # it never moved → a click
   self._pinch_live = False
   self._dragging = False
   ```

**Continuity:** the drag lives in `_pinch_live` / `_dragging` booleans and in
`drag_start`. If the hand flickers out of view for a few frames, the pinch state
machine stays ACTIVE (grace period), `_pinch_live` stays True, and the drag is
never dropped. The cursor just holds at the frozen position.

### C) TWO_FINGERS → scroll

`_handle_scroll`:

1. **Activation** (`scroll_sm.just_activated()`): `scroll_tracker.reset()`.
2. **Held** (`scroll_sm.is_held()`): call `scroll_tracker.update(primary)` →
   returns **notches for this frame**; if nonzero, `osc.scroll_wheel(notches)`.

The `ScrollPositionTracker` uses the **wrist Y**, not velocity (see the big
docstring in `geometry.py`):

- wrist Y relative to the camera's vertical center (`offset = y - 0.5`).
- Inside the `dead_zone` (12%) → 0 notches.
- Outside it → linear speed up to `max_speed` (14 notches/s) at the frame edge.

```python
notches_per_sec = active * self.max_speed
return -sign * notches_per_sec * dt    # negate: hand below centre = scroll down
```

- Uses **real dt** (from `time.monotonic`) so speed is independent of frame rate.
- You don't move the hand to scroll — you **hold** it above/below centre.

**Continuity:** scroll only emits while `is_held()`. The 8-frame grace period on
`scroll_sm` keeps it alive during brief hand losses. `scroll_tracker` is
stateless except for `_prev_t` (reset on activation) — no stale history to jump.

### D) FIST → grab and drag a window

`_handle_fist` uses the *incremental-delta* design:

1. **Activation** (`fist_sm.just_activated()`): capture the window under the
   cursor **once**:
   ```python
   wid, wx, wy = osc.get_window_under_cursor()
   self._fist_window_id = wid
   self._fist_win_pos = [wx, wy]               # window's top-left
   self._fist_prev_wrist = primary[WRIST, :2]  # wrist right now
   ```
   If no moveable window (pure Wayland surface / no xdotool) → `wid=None`,
   nothing to grab.

2. **Held & has window** (`is_held()` and `_fist_window_id` not None): compute
   the **per-frame wrist delta** and accumulate it into the window position:
   ```python
   dx_norm = wrist_now[0] - _fist_prev_wrist[0]
   dy_norm = wrist_now[1] - _fist_prev_wrist[1]
   _fist_win_pos[0] += int(dx_norm * _win_scale_x * FIST_DRAG_SCALE)
   _fist_win_pos[1] += int(dy_norm * _win_scale_y * FIST_DRAG_SCALE)
   _fist_win_pos[1] = max(0, _fist_win_pos[1])   # keep title bar on-screen
   osc.move_window(_fist_window_id, *_fist_win_pos)
   ```
   `_win_scale_x/y` convert normalized deltas into screen pixels (same active
   zone mapping as the cursor), then `FIST_DRAG_SCALE=1.2` amplifies.

3. **Release** (`fist_sm.just_released()`): clear `_fist_window_id` and
   `_fist_prev_wrist` — the window is no longer grabbed.

**Continuity (the key insight):** because we accumulate **deltas per frame**
rather than `wrist_now - wrist_at_grab`, a transient hand loss "freezes" the
previous wrist as the reference. When the hand reappears, delta = 0 for one
frame (window holds) and then resumes smoothly — **no jump, no lost position**.
Updating `_fist_prev_wrist = wrist_now` every held frame keeps the reference
current.

### E) Two hands apart/together → zoom

`_handle_zoom` calls `ZoomTracker.update(hands_landmarks)`:

- Needs **both hands** (`len >= 2`).
- Palm center = avg of wrist (0) and middle MCP (9) → Euclidean distance between
  the two palms.
- If `delta > min_delta` → `"in"`; if `< -min_delta` → `"out"`; then
  `osc.zoom_in()` / `osc.zoom_out()` (which press `ctrl+plus` / `ctrl+minus`).
- Discrete, thresholded — not continuous. `prev_distance=None` resets whenever
  fewer than 2 hands are present, so re-entry doesn't compare against a stale
  distance.

### F) Fast lateral swipe → next/prev

`_handle_swipe` calls `SwipeTracker.update(primary)`:

- Maintains a `deque` (maxlen 10) of `(time, fingertip_x)`.
- Computes **velocity** = `(x_last - x_first) / dt`.
- If `velocity > +1.5` → `"right"` (→ `osc.swipe_next()` → `Right` key);
  if `< -1.5` → `"left"` (→ `osc.swipe_prev()` → `Left` key).
- **Cooldown** `0.8s` between swipes and a `history.clear()` on a detected
  swipe prevent the same swipe firing repeatedly.

---

## 9. Continuity & Stability — the Core Design Math

Continuity in this system is engineered at **multiple independent layers**
because a single point of failure would ruin interactions. The layers:

### Layer 1 — Probability smoothing (in `main.py`, before the engine)
An **asymmetric EMA** on the classifier's softmax vector:
- rises fast (`alpha_rise=0.55`) → a new gesture is recognised quickly;
- falls slow (`alpha_fall=0.15`) → a 1–3 frame classifier hiccup won't drop the
  gesture.
This is the *first* gate: it stops noisy classifier outputs from even reaching
the state machines.

### Layer 2 — The strict-enter / lenient-exit state machine
As described in [§4](#4-the-gesture-state-machine): hard to *start* an action
(needs N stable frames), easy to *keep* it (M-frame grace). This converts a
frame-by-frame boolean into a **state** with hysteresis, so an action "latches"
once begun.

### Layer 3 — Frozen landmarks on hand loss
In `process_frame`:
```python
if has_hands:
    primary = hands_landmarks[0]
    self._last_primary = primary.copy()
else:
    if self._last_primary is None:
        return                      # nothing to preserve yet
    primary = self._last_primary    # reuse last known hand
    gesture_label = "neutral"
    confidence = 0.0
```
When the hand leaves the frame (out of view, bad lighting), the engine **reuses
the last known landmarks** and forces `gesture_active=False`. The state
machines then count down their grace periods using frozen data rather than
immediately cancelling. Result: a drag / scroll / fist survives brief hand loss
and resumes when the hand returns.

### Layer 4 — One-Euro filter + reset-on-entry
Continuous motion (cursor, pinch-midpoint) is smoothed by the adaptive One-Euro
filter, and the filter is **reset** whenever the control gesture is re-entered
so stale history can't cause a jump.

### Layer 5 — Incremental deltas (fist) instead of absolute positions
The fist window-drag accumulates per-frame wrist deltas instead of comparing to
the grab-time wrist. This makes it inherently robust to hand-loss (the reference
is always the previous frame).

### Layer 6 — Background worker threads that drop stale positions
Both the ydotool cursor worker and the xdotool window mover store only the
**latest** pending position/job. If the OS backend is slower than the inference
loop, old positions are overwritten, never queued — so the OS always ends up at
the most recent target, and the inference loop is never blocked.

### Why all six layers?
Individually, any one layer reduces jitter; together they make continuous,
sustained interactions (drag, scroll, window-grab) feel deliberate and stable
rather than flickery or jumpy. This is the heart of "controlling a computer
with your hand" feeling usable.

---

## 10. The OS Backends

`control/os_control.py` is the boundary between the engine and the real OS.

### Backend selection (`init_cursor_backend`)

At startup, `main.py` detects the screen size and calls:

```python
osc.init_cursor_backend(screen_w, screen_h)
```

Logic:
1. Try **evdev** (`_EVDEV_IMPORTABLE` + instantiate `_EvdevMouse`).
   - On success → "✓ Cursor backend: evdev uinput (zero-latency)".
   - On `PermissionError` (no `/dev/uinput` access) → prints how to fix.
   - On other errors → falls through.
2. If evdev unavailable → try **ydotool** (`_YdotoolCursorWorker` thread).
3. If neither → print error, no cursor backend.
4. **xdotool window mover** (`_WindowMoverWorker`) always starts if available.

### The evdev virtual mouse (`_EvdevMouse`)

Creates a uinput device declaring it can emit `REL_X, REL_Y, REL_WHEEL,`
(`REL_WHEEL_HI_RES` if available) and left/right/middle buttons.

It **tracks its own absolute cursor position** (`self._x, self._y`, seeded from
`xdotool getmouselocation` at startup or screen center), and converts an
absolute *target* into **relative deltas** — because that's what a physical
relative mouse produces:

```python
def move(self, x, y):
    dx, dy = x - self._x, y - self._y
    if dx == 0 and dy == 0: return
    if dx: self.ui.write(EV_REL, REL_X, dx)
    if dy: self.ui.write(EV_REL, REL_Y, dy)
    self.ui.syn()
    self._x, self._y = x, y
```

Buttons are `EV_KEY` `BTN_LEFT` press/release with a `syn()` to commit. A
click is down → sleep 40ms → up.

**Scroll** supports both legacy and modern apps:
```python
def scroll_wheel(self, notches):
    hi_res = int(notches * 120)               # 120 units = 1 notch (smooth)
    if hi_res and hasattr(ecodes, "REL_WHEEL_HI_RES"):
        ui.write(EV_REL, REL_WHEEL_HI_RES, hi_res)
    self._wheel_accum += notches              # accumulate fractional notches
    coarse = int(self._wheel_accum)
    if coarse:
        ui.write(EV_REL, REL_WHEEL, coarse)   # legacy whole-notch tick
        self._wheel_accum -= coarse
    self.ui.syn()
```
Modern GTK/Qt apps use `REL_WHEEL_HI_RES` for pixel-level smooth scrolling; a
fractional accumulator ensures `REL_WHEEL` still emits whole-notch ticks for
legacy apps. `self._wheel_accum` is **continuity state** for scroll: leftover
fractional notches carry across frames instead of being lost.

### The ydotool fallback (`_YdotoolCursorWorker`)

A `threading.Thread` that waits on an `Event`, picks up the single pending
`(x, y)`, and shells out to `ydotool mousemove`. Because subprocess calls are
slow compared to evdev, running them off the main thread keeps the inference
loop responsive, and only the **latest** position is kept (stale ones dropped).

### The window mover (`_WindowMoverWorker`)

A thread that waits on an `Event` and calls `xdotool windowmove <wid> <x> <y>`
with the latest `(wid, x, y)`. Non-blocking to the engine; drops stale updates;
`timeout=0.15s` so a slow xdotool can't stall anything.

### `get_window_under_cursor` — grabbing the right window

```python
xdotool getmouselocation --shell   # → parse WINDOW=...
xdotool getwindowgeometry --shell <wid>   # → parse X= / Y=
```
Returns `(wid, wx, wy)` — called **once**, at fist-gesture *entry*. Returns
`(None, 0, 0)` on failure or pure-Wayland surfaces (which xdotool can't see).

### Public API used by the engine

```python
move_cursor_absolute(x, y)   # clamps to hot-corner margin, then backend move
click_left()                 # full click
mouse_down() / mouse_up()    # drag press / release
scroll_wheel(notches)        # signed floats
zoom_in() / zoom_out()       # ctrl+plus / ctrl+minus
swipe_next() / swipe_prev()  # Right / Left keys
get_window_under_cursor()    # (wid, wx, wy)
move_window(wid, x, y)       # enqueue move
```

### The hot-corner guard

```python
HOT_CORNER_MARGIN = 20
def _clamp(x, y):
    return max(20, min(screen_w-20, x)), max(20, min(screen_h-20, y))
```
`move_cursor_absolute` clamps the cursor 20px away from every edge/corner so it
can't accidentally trigger GNOME's hot corner (which would grab focus).

---

## 11. Emergency Stop & Safety

Pressing **ESC** in the video window calls `engine.emergency_stop()`:

```python
def emergency_stop(self):
    if self._dragging:
        osc.mouse_up()          # release any held mouse button / drag
    self._pinch_live = False
    self._dragging = False
    self._fist_window_id = None
    self.control_enabled = False
```

While `control_enabled == False`, `process_frame` refuses all control:
- releases a live drag,
- clears the fist grab,
- returns immediately.

This is the safety valve: a single key instantly kills all OS input. Pressing
`q` exits the program.

---

## 12. A Worked Frame-by-Frame Trace

### Scenario: user performs a pinch-drag-release of a window slider

Suppose the classifier output is `pinch` with high confidence.

| Frame | State machine | Engine action |
|-------|---------------|----------------|
| 1 | IDLE → DETECTING (stable=1) | nothing yet |
| 2 | DETECTING (stable=2) | nothing yet |
| 3 | DETECTING (stable=3) | nothing yet |
| 4 | DETECTING (stable=4) → CONFIRMED | nothing yet |
| 5 | CONFIRMED → ACTIVE, `just_activated=True` | record `_drag_start`, `_pinch_live=True`; cursor moves to thumb/index midpoint |
| 6 | ACTIVE | moves cursor toward midpoint; hand hasn't moved >18px → not dragging |
| 7 | ACTIVE | hand moved >18px from `_drag_start` → `_dragging=True`, `osc.mouse_down()` |
| 8 | ACTIVE (hand flickers out, conf=0) | `_exit_count=1` — **still ACTIVE**; uses frozen landmarks; cursor holds |
| 9 | ACTIVE (confidence back) | `_exit_count=0`; drag continues |
| 10 | ACTIVE | keep moving + dragging |
| 11 | not confident ×5 → RELEASED, `just_released=True` | `_dragging` → `osc.mouse_up()` |
| 12 | RELEASED → COOLDOWN (0.25s) | `_pinch_live=False` |

**Continuity instantiated here:** the frame-8 flicker did *not* end the drag
because (a) smoothing kept the label stable, (b) the grace period absorbed it,
and (c) frozen landmarks kept geometry valid. Only a sustained loss (5 frames)
released it, and only then as an intentional `mouse_up`.

### Scenario: fist window grab

| Frame | Action |
|-------|--------|
| 1–5 | fist_sm climbs IDLE→ACTIVE (5 stable frames) |
| 6 | `just_activated`: grab window under cursor → `(_fist_window_id, _fist_win_pos)`, `_fist_prev_wrist = current wrist` |
| 7 | held: delta = wrist_now − wrist_prev → win_pos += delta×scale×1.2 → `move_window` |
| 8 | held: hand leaves frame → frozen wrist ⇒ delta=0 ⇒ window holds position |
| 9 | held: hand returns → delta resumes ⇒ drag continues (no jump) |
| 20 | sustained loss → RELEASED → `_fist_window_id=None`, grab released |

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **Gesture state machine** | hysteresis state machine (IDLE→DETECTING→CONFIRMED→ACTIVE→RELEASED→COOLDOWN) for one gesture. |
| **Stable frames** | consecutive confident frames required before activation; strict. |
| **Grace period** | consecutive unconfident frames tolerated while ACTIVE; lenient. |
| **Cooldown** | pause after release before the gesture can re-trigger. |
| **`just_activated()` / `just_released()`** | True for exactly the one frame the transition fires — used for one-time setup/teardown. |
| **`is_held()`** | True every frame a sustained gesture is ACTIVE. |
| **One-Euro filter** | adaptive low-pass: more smoothing when slow, less lag when fast. |
| **Active (dead) zone** | camera region that controls the cursor; edges ignored. |
| **Pinch midpoint** | mean of thumb(4) and index(8) landmarks; the drag/anchor point. |
| **Drag threshold (18px)** | movement needed before a pinch becomes a drag instead of a click. |
| **FIST_DRAG_SCALE (1.2)** | amplification of wrist motion when dragging a window. |
| **Incremental delta** | accumulate per-frame motion deltas rather than absolute positions. |
| **evdev UInput** | virtual relative mouse at `/dev/uinput`; fast, zero-latency. |
| **REL_WHEEL_HI_RES** | high-resolution scroll event (120 = 1 notch) for smooth scrolling. |
| **ydotool worker** | fallback cursor backend; threaded subprocess, drops stale pos. |
| **xdotool window mover** | threaded X11 window mover; used for fist drag. |
| **Hot-corner margin** | clamp distance from screen edges to avoid GNOME hot corner. |
| **`_last_primary`** | frozen last-known hand landmarks used during hand loss. |
| **EMA smoother** | asymmetric probability smoother in `main.py` (fast-rise/slow-fall). |
| **Control-enabled flag** | global gate; `False` after emergency stop. |
```

And so, in one sentence: **a gesture triggers an action only after it has been
confirmed for N stable frames (state machine), and because the engine retains
state — SMP probabilities, per-gesture states, frozen landmarks, adaptive
filters, incremental deltas, and threaded backends that never block — every
sustained interaction stays continuous across noise, hand-loss, and OS lag.**
```
